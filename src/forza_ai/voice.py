"""APEX voice co-pilot: "APEX, speed it up a bit" -> a bounded live tuning change.

Pipeline (all off the control thread):
  microphone -> utterance (energy VAD, or hold a wheel push-to-talk button)
             -> ElevenLabs speech-to-text -> wake word check ("APEX ...")
             -> Claude tool call -> ("tune", adjustments, reply) on the runtime's event queue
             -> the control loop applies it through forza_ai.tuning (ranges + step limits)
             -> spoken confirmation (ElevenLabs text-to-speech), or console only.

The voice path can only change driving-style settings or hand control BACK to the human; it can
never arm/engage the AI. Without ANTHROPIC_API_KEY a small keyword parser is used instead of Claude.

Try it without the game:
  python -m forza_ai.voice --text "APEX, speed it up a bit"     # parsing only, no mic/STT
  python -m forza_ai.voice --listen                              # mic + STT + parsing, prints results
"""

import argparse
import io
import json
import os
import queue
import re
import threading
import time
import wave

import numpy as np

from forza_ai.tuning import PARAMETERS

SAMPLE_RATE = 16_000
BLOCK = 480                                     # 30 ms
ELEVENLABS = "https://api.elevenlabs.io/v1"
ANTHROPIC = "https://api.anthropic.com"         # explicit: never inherit a proxy base URL from the env
DEFAULT_MODEL = "claude-haiku-4-5"              # fastest Claude; a command should land within a second or two
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"       # ElevenLabs premade "Rachel"; override with ELEVENLABS_VOICE_ID
WAKE = re.compile(r"\b(apex|a\s?pex|apecs|apex's|ape\s?x|aypex|a\s?packs)\b[\s,.!?:;-]*", re.I)


# ---------------------------------------------------------------- audio segmentation

class Segmenter:
    """Cut 30 ms int16 blocks into utterances by energy, against an adaptive noise floor."""

    def __init__(self, start_blocks=3, end_blocks=23, preroll_blocks=10, max_blocks=270, min_blocks=12,
                 min_rms=400.0):
        self.start_blocks, self.end_blocks, self.preroll = start_blocks, end_blocks, preroll_blocks
        self.max_blocks, self.min_blocks, self.min_rms = max_blocks, min_blocks, min_rms
        self.noise = min_rms / 3
        self._history, self._speech, self._loud, self._quiet = [], None, 0, 0

    def feed(self, block):
        rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
        loud = rms > max(self.min_rms, self.noise * 3)
        if self._speech is None:
            self._history = (self._history + [block])[-self.preroll:]
            self._loud = self._loud + 1 if loud else 0
            if not loud:
                self.noise += 0.05 * (rms - self.noise)          # track background (game audio, fans)
            if self._loud >= self.start_blocks:
                self._speech, self._quiet = list(self._history), 0
            return None
        self._speech.append(block)
        self._quiet = 0 if loud else self._quiet + 1
        if self._quiet >= self.end_blocks or len(self._speech) >= self.max_blocks:
            speech, self._speech, self._history, self._loud = self._speech, None, [], 0
            if len(speech) - self._quiet >= self.min_blocks:
                return np.concatenate(speech)
        return None


def wav_bytes(samples):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


def strip_wake(text, required=True):
    """Command text after the wake word; None if a required wake word is missing."""
    match = WAKE.search(text or "")
    if match is None:
        return None if required else (text or "").strip()
    return text[match.end():].strip() or None


# ---------------------------------------------------------------- speech services

class ElevenLabsSTT:
    def __init__(self, api_key, model=None, timeout=8.0):
        import httpx
        self.client = httpx.Client(timeout=timeout, headers={"xi-api-key": api_key})
        self.model = model or os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")

    def transcribe(self, samples):
        response = self.client.post(f"{ELEVENLABS}/speech-to-text",
                                    data={"model_id": self.model, "language_code": "en"},
                                    files={"file": ("speech.wav", wav_bytes(samples), "audio/wav")})
        response.raise_for_status()
        return response.json().get("text", "").strip()


class ElevenLabsTTS:
    def __init__(self, api_key, voice_id=None, timeout=8.0):
        import httpx
        self.client = httpx.Client(timeout=timeout, headers={"xi-api-key": api_key})
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)

    def say(self, text):
        import sounddevice
        response = self.client.post(f"{ELEVENLABS}/text-to-speech/{self.voice_id}",
                                    params={"output_format": "pcm_16000"},
                                    json={"text": text, "model_id": "eleven_flash_v2_5"})
        response.raise_for_status()
        sounddevice.play(np.frombuffer(response.content, dtype=np.int16), SAMPLE_RATE, blocking=True)


# ---------------------------------------------------------------- understanding

def _settings_text(settings):
    lines = []
    for name, value in settings.items():
        low, high, step, meaning = PARAMETERS[name]
        limit = f", at most {step:g} per command" if step else ""
        lines.append(f"- {name} = {value:g} (range {low:g}-{high:g}{limit}): {meaning}")
    return "\n".join(lines)


SYSTEM = """You are APEX, the voice co-pilot of an AI that drives a car in Forza Horizon 4. The human \
speaks to you while the AI drives. Turn each request into a tool call. You can only adjust the \
driving-style settings listed below, or hand control back to the human; you can never start the AI.

Current settings (only these are adjustable in this run):
{settings}

Guidance:
- "a bit"/"a little" = about a quarter of the per-command limit; plain request = half; "a lot"/"much" = the full limit.
- "Speed up"/"faster"/"more aggressive": raise throttle_cap and throttle_rate (if present); if a speed limit is on, raise it.
- "Slow down"/"calmer": the reverse. "Brake harder/earlier" -> brake_gain up. "Turn more"/"it's understeering" -> steer_gain up.
- An explicit number ("speed limit 150", "throttle 80 percent" = 0.8) -> mode "set". Relative requests -> mode "change".
- "Stop", "I've got it", "take over", "let me drive" -> take_over.
- Questions about the settings, or unclear requests -> answer, briefly.
- spoken_reply: at most 10 words, no settings jargon."""

TOOLS = [
    {"name": "adjust_driving",
     "description": "Change one or more driving-style settings of the AI driver.",
     "input_schema": {"type": "object", "properties": {
         "adjustments": {"type": "array", "items": {"type": "object", "properties": {
             "parameter": {"type": "string", "enum": sorted(PARAMETERS)},
             "mode": {"type": "string", "enum": ["set", "change"]},
             "value": {"type": "number", "description": "new value (set) or signed amount (change)"}},
             "required": ["parameter", "mode", "value"]}},
         "spoken_reply": {"type": "string"}},
         "required": ["adjustments", "spoken_reply"]}},
    {"name": "take_over",
     "description": "Disengage the AI immediately and give the car back to the human driver.",
     "input_schema": {"type": "object", "properties": {"spoken_reply": {"type": "string"}},
                      "required": ["spoken_reply"]}},
    {"name": "answer",
     "description": "Reply without changing anything (questions, unclear or unsupported requests).",
     "input_schema": {"type": "object", "properties": {"spoken_reply": {"type": "string"}},
                      "required": ["spoken_reply"]}},
]


class ClaudeBrain:
    def __init__(self, api_key, model=None, timeout=6.0):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key, base_url=ANTHROPIC, timeout=timeout, max_retries=0)
        self.model = model or DEFAULT_MODEL
        self.name = f"Claude ({self.model})"

    def interpret(self, text, settings):
        tools = [dict(t) for t in TOOLS]
        names = sorted(settings)
        schema = json.loads(json.dumps(tools[0]["input_schema"]))
        schema["properties"]["adjustments"]["items"]["properties"]["parameter"]["enum"] = names
        tools[0]["input_schema"] = schema
        message = self.client.messages.create(
            model=self.model, max_tokens=300, tools=tools, tool_choice={"type": "any"},
            system=SYSTEM.format(settings=_settings_text(settings)),
            messages=[{"role": "user", "content": text}])
        for block in message.content:
            if block.type == "tool_use":
                data = dict(block.input)
                return {"action": {"adjust_driving": "tune", "take_over": "take_over"}.get(block.name, "answer"),
                        "adjustments": data.get("adjustments", []), "reply": data.get("spoken_reply", "")}
        return {"action": "answer", "adjustments": [], "reply": "Sorry, say that again?"}


class KeywordBrain:
    """Offline fallback: a few fixed phrases, no API key needed."""
    name = "keyword parser (no ANTHROPIC_API_KEY)"
    RULES = (
        (r"\b(stop|take over|i've got it|i got it|let me drive|my car)\b", "take_over", ()),
        (r"\b(speed (it )?up|faster|quicker|more aggressive|push)\b", "tune",
         (("throttle_cap", .5), ("throttle_rate", .5))),
        (r"\b(slow (it )?down|slower|calm|gentle|less aggressive)\b", "tune",
         (("throttle_cap", -.5), ("throttle_rate", -.5))),
        (r"\b(brake (harder|more|earlier))\b", "tune", (("brake_gain", .5),)),
        (r"\b(brake (less|softer|later))\b", "tune", (("brake_gain", -.5),)),
        (r"\b(understeer\w*|turn (in )?more|steer more|sharper)\b", "tune", (("steer_gain", .5),)),
        (r"\b(oversteer\w*|turn (in )?less|steer less)\b", "tune", (("steer_gain", -.5),)),
    )

    def interpret(self, text, settings):
        lowered = text.lower()
        scale = .5 if re.search(r"\b(a (little )?bit|slightly|a little)\b", lowered) else \
            2.0 if re.search(r"\b(a lot|much|way)\b", lowered) else 1.0
        for pattern, action, changes in self.RULES:
            if re.search(pattern, lowered):
                adjustments = [{"parameter": name, "mode": "change",
                                "value": fraction * scale * PARAMETERS[name][2]}
                               for name, fraction in changes if name in settings]
                if action == "take_over":
                    return {"action": "take_over", "adjustments": [], "reply": "Your car."}
                if adjustments:
                    return {"action": "tune", "adjustments": adjustments, "reply": "Done."}
        return {"action": "answer", "adjustments": [], "reply": "I didn't catch a setting."}


def describe(results):
    """Short spoken summary of what actually changed (after ranges and step limits)."""
    words = {"throttle_cap": lambda v: f"throttle {v:.0%}", "throttle_rate": lambda v: f"throttle ramp {v:g}",
             "max_speed_kmh": lambda v: f"speed limit {v:g}" if v else "speed limit off",
             "brake_gain": lambda v: f"brakes times {v:g}",
             "corner_speed_kmh": lambda v: f"corner limit {v:g}" if v else "corner limit off",
             "steer_gain": lambda v: f"steering times {v:g}"}
    changed = [words[r["parameter"]](r["new"]) for r in results if r.get("applied")]
    if not changed:
        notes = [r.get("note") for r in results if r.get("note")]
        return "Already at the limit." if any("limit" in (n or "") or "clamp" in (n or "") for n in notes) \
            else "Nothing changed."
    return ", ".join(changed) + "."


# ---------------------------------------------------------------- the assistant

class VoiceAssistant:
    def __init__(self, stt, brain, speaker=None, *, wake_word=True, ptt_button=None, device=None, log=print):
        self.stt, self.brain, self.speaker = stt, brain, speaker
        self.wake_word = wake_word and ptt_button is None
        self.ptt_button, self.device, self.log = ptt_button, device, log
        self._events = self._tuning = None
        self._utterances = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._ptt = False
        self._threads = []
        self._stream = None

    # runtime hooks
    def start(self, events, tuning):
        import sounddevice
        self._events, self._tuning = events, tuning
        blocks = queue.Queue(maxsize=400)

        def callback(data, frames, timing, status):
            try:
                blocks.put_nowait(data[:, 0].copy())
            except queue.Full:
                pass

        self._stream = sounddevice.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                               blocksize=BLOCK, device=self.device, callback=callback)
        self._stream.start()
        for target, args in ((self._segment, (blocks,)), (self._work, ())):
            thread = threading.Thread(target=target, args=args, name="apex-voice", daemon=True)
            thread.start()
            self._threads.append(thread)
        how = f"hold wheel button {self.ptt_button} and talk" if self.ptt_button is not None \
            else 'say "APEX, ..."'
        self.log(f"[APEX] listening ({how}); understanding: {self.brain.name}; "
                 f"replies: {'spoken' if self.speaker else 'console only'}")

    def wheel_buttons(self, buttons):
        self._ptt = self.ptt_button is not None and self.ptt_button in buttons

    def close(self):
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        for thread in self._threads:
            thread.join(timeout=1.0)

    # threads
    def _segment(self, blocks):
        segmenter, held = Segmenter(), []
        while not self._stop.is_set():
            try:
                block = blocks.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._speaking.is_set():
                continue                                # don't listen to our own reply
            if self.ptt_button is not None:
                if self._ptt:
                    held.append(block)
                elif held:
                    if len(held) >= 10:                 # 0.3 s
                        self._offer(np.concatenate(held))
                    held = []
                continue
            utterance = segmenter.feed(block)
            if utterance is not None:
                self._offer(utterance)

    def _offer(self, samples):
        try:
            self._utterances.put_nowait(samples)
        except queue.Full:
            pass                                        # still busy with earlier speech; drop it

    def _work(self):
        while not self._stop.is_set():
            try:
                samples = self._utterances.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.handle_text(self.stt.transcribe(samples))
            except Exception as error:                  # network, API or audio trouble: never fatal
                self.log(f"[APEX] error: {type(error).__name__}: {error}")

    def handle_text(self, text):
        started = time.monotonic()
        command = strip_wake(text, required=self.wake_word)
        if command is None:
            return None                                 # chatter without the wake word
        self.log(f'[APEX] heard: "{text}"')
        decision = self.brain.interpret(command, self._tuning.snapshot())
        reply = decision["reply"]
        if decision["action"] == "take_over":
            self._events.put_nowait("manual")
        elif decision["action"] == "tune" and decision["adjustments"]:
            answer = queue.Queue(maxsize=1)
            self._events.put_nowait(("tune", decision["adjustments"], answer))
            try:
                reply = describe(answer.get(timeout=1.0))
            except queue.Empty:
                reply = "The driver didn't respond."
        self.log(f"[APEX] {reply}  ({(time.monotonic() - started) * 1000:.0f} ms)")
        self._say(reply)
        return decision

    def _say(self, text):
        if self.speaker is None or not text:
            return
        self._speaking.set()
        try:
            self.speaker.say(text)
        except Exception as error:
            self.log(f"[APEX] speech error: {type(error).__name__}: {error}")
        finally:
            time.sleep(0.2)                             # let the room echo die down
            self._speaking.clear()


def build(*, ptt_button=None, device=None, model=None, speak=True, wake_word=True, log=print):
    """Assistant from environment keys: ELEVENLABS_API_KEY (required), ANTHROPIC_API_KEY (optional)."""
    eleven = os.environ.get("ELEVENLABS_API_KEY")
    if not eleven:
        raise RuntimeError("voice needs ELEVENLABS_API_KEY (speech-to-text)")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    brain = ClaudeBrain(anthropic_key, model) if anthropic_key else KeywordBrain()
    speaker = ElevenLabsTTS(eleven) if speak else None
    return VoiceAssistant(ElevenLabsSTT(eleven), brain, speaker, wake_word=wake_word,
                          ptt_button=ptt_button, device=device, log=log)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Try the APEX voice co-pilot without the game.")
    parser.add_argument("--text", help="interpret this sentence (no microphone or speech-to-text)")
    parser.add_argument("--listen", action="store_true", help="use the microphone; Ctrl+C to stop")
    parser.add_argument("--device", help="microphone name or index (python -m sounddevice lists them)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-speak", action="store_true")
    args = parser.parse_args(argv)
    from forza_ai.tuning import LiveTuning
    tuning = LiveTuning(throttle_cap=0.7, throttle_rate=0.5, max_speed_kmh=0.0, brake_gain=1.5,
                        corner_speed_kmh=130.0, steer_gain=1.6)
    events = queue.Queue()

    def drain():                                        # stand-in for the control loop
        while True:
            event = events.get()
            if isinstance(event, tuple):
                event[2].put(tuning.apply(event[1]))
            else:
                print(f"[runtime] would disengage: {event}")

    threading.Thread(target=drain, daemon=True).start()
    if args.text:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        brain = ClaudeBrain(anthropic_key, args.model) if anthropic_key else KeywordBrain()
        assistant = VoiceAssistant(None, brain, wake_word=False)
        assistant._events, assistant._tuning = events, tuning
        decision = assistant.handle_text(args.text)
        print(json.dumps(decision, indent=2))
        print("settings now:", tuning.snapshot())
        return 0
    if not args.listen:
        parser.error("choose --text or --listen")
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    assistant = build(device=device, model=args.model, speak=not args.no_speak)
    assistant.start(events, tuning)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        assistant.close()
        print("settings now:", tuning.snapshot())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
