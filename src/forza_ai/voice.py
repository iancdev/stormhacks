"""APEX voice co-pilot: "APEX, speed it up a bit" -> a bounded live tuning change, via Gemini.

Pipeline (all off the control thread):
  microphone -> utterance (energy VAD, or hold a wheel push-to-talk button)
             -> ONE Gemini call with the audio clip + function tools: Gemini hears the speech,
                checks it is addressed to APEX and picks adjust_driving / take_over / answer / ignore
             -> ("tune", adjustments, reply) on the runtime's event queue
             -> the control loop applies it through forza_ai.tuning (ranges + step limits)
             -> spoken confirmation with Gemini text-to-speech (or console only).

The voice path can only change driving-style settings or hand control BACK to the human; it can
never arm/engage the AI. Needs GEMINI_API_KEY (Google AI Studio).

Try it without the game:
  python -m forza_ai.voice --text "APEX, speed it up a bit"     # Gemini (or keywords without a key)
  python -m forza_ai.voice --listen                              # mic + Gemini, prints results
"""

import argparse
import base64
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
DEFAULT_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-lite-tts"         # the fast TTS model, for a quick spoken confirmation
TTS_RATE = 24_000
TTS_VOICE = "Kore"
WAKE = re.compile(r"\b(apex|a\s?pex|apecs|apex's|ape\s?x|aypex|a\s?packs)\b[\s,.!?:;-]*", re.I)


# ---------------------------------------------------------------- audio segmentation

class Segmenter:
    """Cut 30 ms int16 blocks into utterances by energy, against an adaptive noise floor."""

    def __init__(self, start_blocks=3, end_blocks=23, preroll_blocks=10, max_blocks=270, min_blocks=12,
                 min_rms=150.0):
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


# ---------------------------------------------------------------- understanding

def _settings_text(settings):
    lines = []
    for name, value in settings.items():
        low, high, step, meaning = PARAMETERS[name]
        limit = f", at most {step:g} per command" if step else ""
        lines.append(f"- {name} = {value:g} (range {low:g}-{high:g}{limit}): {meaning}")
    return "\n".join(lines)


SYSTEM = """You are APEX, the voice co-pilot of an AI that drives a car in Forza Horizon 4. The human \
speaks to you while the AI drives; game audio may be in the background. Always answer with exactly \
one tool call. You can only adjust the driving-style settings listed below, or hand control back to \
the human; you can never start the AI.
{wake}
Current settings (only these are adjustable in this run):
{settings}

Guidance:
- "a bit"/"a little" = about a quarter of the per-command limit; plain request = half; "a lot"/"much" = the full limit.
- "Speed up"/"faster"/"more aggressive": raise throttle_cap and throttle_rate (if present); if a speed limit is on, raise it.
- "Slow down"/"calmer": the reverse. "Brake harder/earlier" -> brake_gain up. "Turn more"/"it's understeering" -> steer_gain up.
- An explicit number ("speed limit 150", "throttle 80 percent" = 0.8) -> mode "set". Relative requests -> mode "change".
- "Stop", "I've got it", "take over", "let me drive" -> take_over.
- Questions about the settings, or unclear requests -> answer, briefly.
- heard: what the human said, verbatim. spoken_reply: at most 10 words, no settings jargon."""

WAKE_RULE = """Only respond to speech addressed to you by name ("APEX, ..."; also accept close mishearings \
like "Apex", "Apecs", "A-pex"). Anything else (game audio, talking to someone else, no speech) -> ignore."""


def _tools(settings):
    heard = {"type": "string", "description": "what the human said, verbatim"}
    reply = {"type": "string", "description": "short spoken reply, at most 10 words"}
    return [
        {"type": "function", "name": "adjust_driving",
         "description": "Change one or more driving-style settings of the AI driver.",
         "parameters": {"type": "object", "properties": {
             "heard": heard,
             "adjustments": {"type": "array", "items": {"type": "object", "properties": {
                 "parameter": {"type": "string", "enum": sorted(settings)},
                 "mode": {"type": "string", "enum": ["set", "change"]},
                 "value": {"type": "number", "description": "new value (set) or signed amount (change)"}},
                 "required": ["parameter", "mode", "value"]}},
             "spoken_reply": reply},
             "required": ["heard", "adjustments", "spoken_reply"]}},
        {"type": "function", "name": "take_over",
         "description": "Disengage the AI immediately and give the car back to the human driver.",
         "parameters": {"type": "object", "properties": {"heard": heard, "spoken_reply": reply},
                        "required": ["heard", "spoken_reply"]}},
        {"type": "function", "name": "answer",
         "description": "Reply without changing anything (questions, unclear or unsupported requests).",
         "parameters": {"type": "object", "properties": {"heard": heard, "spoken_reply": reply},
                        "required": ["heard", "spoken_reply"]}},
        {"type": "function", "name": "ignore",
         "description": "The audio is not addressed to APEX, or contains no request.",
         "parameters": {"type": "object", "properties": {"heard": heard}, "required": ["heard"]}},
    ]


ACTIONS = {"adjust_driving": "tune", "take_over": "take_over", "answer": "answer", "ignore": "ignore"}


class GeminiBrain:
    """Hears the audio clip (or reads text) and picks a tool, in one Gemini call."""

    def __init__(self, api_key, model=None):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("APEX_GEMINI_MODEL", DEFAULT_MODEL)
        self.name = f"Gemini ({self.model})"
        self._fast = True                       # minimal thinking: a command should land in about a second

    def interpret(self, settings, *, audio=None, text=None, wake_required=True):
        if audio is not None:
            content = [{"type": "text", "text": "Voice command audio:"},
                       {"type": "audio", "data": base64.b64encode(wav_bytes(audio)).decode("ascii"),
                        "mime_type": "audio/wav"}]
        else:
            content = [{"type": "text", "text": text}]
        system = SYSTEM.format(wake=WAKE_RULE if wake_required else "", settings=_settings_text(settings))
        config = {"tool_choice": "any"}
        if self._fast:
            config["thinking_level"] = "minimal"
        try:
            interaction = self.client.interactions.create(
                model=self.model, input=content, system_instruction=system, tools=_tools(settings),
                generation_config=config, store=False)
        except Exception:
            if not self._fast:
                raise
            self._fast = False                  # model without thinking_level: retry once, plain
            config.pop("thinking_level")
            interaction = self.client.interactions.create(
                model=self.model, input=content, system_instruction=system, tools=_tools(settings),
                generation_config=config, store=False)
        for step in interaction.steps or ():
            if getattr(step, "type", None) == "function_call":
                data = step.arguments if isinstance(step.arguments, dict) else json.loads(step.arguments or "{}")
                return {"action": ACTIONS.get(step.name, "answer"), "heard": data.get("heard", ""),
                        "adjustments": data.get("adjustments", []), "reply": data.get("spoken_reply", "")}
        return {"action": "ignore", "heard": "", "adjustments": [], "reply": ""}


class GeminiTTS:
    def __init__(self, client, voice=None, device=None):
        self.client, self.device = client, device
        self.voice = voice or os.environ.get("APEX_VOICE", TTS_VOICE)

    def say(self, text):
        import sounddevice
        interaction = self.client.interactions.create(
            model=TTS_MODEL,
            input=[{"type": "user_input", "content": [{"type": "text", "text": text, "annotations": [
                {"type": "speech_metadata", "style": "calm, confident race engineer on the radio"}]}]}],
            response_format={"type": "audio", "mime_type": "audio/l16", "sample_rate": TTS_RATE},
            generation_config={"speech_config": [{"voice": self.voice}]}, store=False)
        pcm = np.frombuffer(base64.b64decode(interaction.output_audio.data), dtype="<i2")
        sounddevice.play(pcm, TTS_RATE, blocking=True, device=self.device)


class KeywordBrain:
    """Offline fallback for typed text: a few fixed phrases, no API key needed."""
    name = "keyword parser (no GEMINI_API_KEY)"
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

    def interpret(self, settings, *, audio=None, text=None, wake_required=True):
        if audio is not None:
            raise RuntimeError("the keyword parser needs text; set GEMINI_API_KEY for voice")
        command = strip_wake(text, required=wake_required)
        if command is None:
            return {"action": "ignore", "heard": text, "adjustments": [], "reply": ""}
        lowered = command.lower()
        scale = .5 if re.search(r"\b(a (little )?bit|slightly|a little)\b", lowered) else \
            2.0 if re.search(r"\b(a lot|much|way)\b", lowered) else 1.0
        for pattern, action, changes in self.RULES:
            if re.search(pattern, lowered):
                adjustments = [{"parameter": name, "mode": "change",
                                "value": fraction * scale * PARAMETERS[name][2]}
                               for name, fraction in changes if name in settings]
                if action == "take_over":
                    return {"action": "take_over", "heard": text, "adjustments": [], "reply": "Your car."}
                if adjustments:
                    return {"action": "tune", "heard": text, "adjustments": adjustments, "reply": "Done."}
        return {"action": "answer", "heard": text, "adjustments": [], "reply": "I didn't catch a setting."}


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
    def __init__(self, brain, speaker=None, *, wake_word=True, ptt_button=None, device=None, log=print):
        self.brain, self.speaker = brain, speaker
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
        mic = sounddevice.query_devices(self.device, kind="input")["name"]
        self.log(f"[APEX] listening on {mic} ({how}); {self.brain.name}; "
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
        self.log(f"[APEX] ...thinking ({len(samples) / SAMPLE_RATE:.1f} s of speech)")
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
                self.handle(audio=samples)
            except Exception as error:                  # network, API or audio trouble: never fatal
                self.log(f"[APEX] error: {type(error).__name__}: {error}")

    def handle(self, *, audio=None, text=None):
        started = time.monotonic()
        decision = self.brain.interpret(self._tuning.snapshot(), audio=audio, text=text,
                                        wake_required=self.wake_word)
        if decision["action"] == "ignore":              # chatter, game audio, not addressed to APEX
            if decision.get("heard"):
                self.log(f'[APEX] not for me: "{decision["heard"]}"')
            return None
        self.log(f'[APEX] heard: "{decision.get("heard", "")}"')
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


def _key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def build(*, ptt_button=None, device=None, model=None, speak=True, wake_word=True, log=print, output_device=None):
    """Assistant from GEMINI_API_KEY (Google AI Studio)."""
    key = _key()
    if not key:
        raise RuntimeError("voice needs GEMINI_API_KEY (Google AI Studio)")
    brain = GeminiBrain(key, model)
    return VoiceAssistant(brain, GeminiTTS(brain.client, device=output_device) if speak else None, wake_word=wake_word,
                          ptt_button=ptt_button, device=device, log=log)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Try the APEX voice co-pilot without the game.")
    parser.add_argument("--text", help="interpret this sentence (no microphone)")
    parser.add_argument("--listen", action="store_true", help="use the microphone; Ctrl+C to stop")
    parser.add_argument("--device", help="microphone name or index (python -m sounddevice lists them)")
    parser.add_argument("--output", help="speaker for replies, name or index (default: Windows default)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-speak", action="store_true")
    args = parser.parse_args(argv)
    output = int(args.output) if args.output and args.output.isdigit() else args.output
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
        brain = GeminiBrain(_key(), args.model) if _key() else KeywordBrain()
        speaker = GeminiTTS(brain.client, device=output) if _key() and not args.no_speak else None
        assistant = VoiceAssistant(brain, speaker)
        assistant._events, assistant._tuning = events, tuning
        decision = assistant.handle(text=args.text)
        print(json.dumps(decision, indent=2))
        print("settings now:", tuning.snapshot())
        return 0
    if not args.listen:
        parser.error("choose --text or --listen")
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    assistant = build(device=device, model=args.model, speak=not args.no_speak, output_device=output)
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
