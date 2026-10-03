"""
Forza AI wheel project - hardware check script.

Run ONE test at a time, from the activated venv:

    python check.py wheel      # live TMX axes + buttons (Ctrl+C to stop)
    python check.py ffb        # small force left/right on the TMX (hold wheel lightly!)
    python check.py vjoy       # sweep vJoy X axis + log any FFB packets sent to vJoy
    python check.py telemetry  # print Forza Data Out packets (Forza must be running)
    python check.py bind steer|brake|gas   # wiggle one vJoy axis for Forza's binding screen
"""
import sys
import time

# ---------------------------------------------------------------- helpers

def open_sdl():
    import sdl2
    # keep reading the wheel even when this window isn't focused
    sdl2.SDL_SetHint(sdl2.SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, b"1")
    if sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK | sdl2.SDL_INIT_HAPTIC) != 0:
        sys.exit("SDL init failed: " + sdl2.SDL_GetError().decode())
    return sdl2


def find_tmx(sdl2):
    n = sdl2.SDL_NumJoysticks()
    print(f"Found {n} controller(s):")
    tmx_index = None
    for i in range(n):
        name = (sdl2.SDL_JoystickNameForIndex(i) or b"?").decode(errors="replace")
        print(f"  [{i}] {name}")
        if tmx_index is None and "vjoy" not in name.lower() and (
            "tmx" in name.lower() or "thrustmaster" in name.lower()
        ):
            tmx_index = i
    if tmx_index is None:
        sys.exit("No Thrustmaster/TMX wheel found. Is it plugged in and powered?")
    js = sdl2.SDL_JoystickOpen(tmx_index)
    print(f"\nUsing [{tmx_index}]: {sdl2.SDL_JoystickNumAxes(js)} axes, "
          f"{sdl2.SDL_JoystickNumButtons(js)} buttons\n")
    return js


# ---------------------------------------------------------------- tests

def test_wheel():
    """Turn the wheel and press each pedal. Note which axis number moves."""
    sdl2 = open_sdl()
    js = find_tmx(sdl2)
    n_axes = sdl2.SDL_JoystickNumAxes(js)
    n_btn = sdl2.SDL_JoystickNumButtons(js)
    print("Turn the wheel and press each pedal. Ctrl+C to stop.\n")
    try:
        while True:
            sdl2.SDL_JoystickUpdate()
            axes = "  ".join(f"a{i}:{sdl2.SDL_JoystickGetAxis(js, i):+6d}" for i in range(n_axes))
            pressed = [i for i in range(n_btn) if sdl2.SDL_JoystickGetButton(js, i)]
            print(f"\r{axes}  buttons:{pressed}      ", end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nDone.")


def test_ffb():
    """Push the wheel gently left, then right, using a constant-force effect."""
    sdl2 = open_sdl()
    js = find_tmx(sdl2)
    haptic = sdl2.SDL_HapticOpenFromJoystick(js)
    if not haptic:
        sys.exit("Wheel has no haptic support via SDL: " + sdl2.SDL_GetError().decode())
    if not (sdl2.SDL_HapticQuery(haptic) & sdl2.SDL_HAPTIC_CONSTANT):
        sys.exit("Wheel does not report constant-force support.")

    LEVEL = 12000  # out of 32767 (~35%)

    def err():
        return sdl2.SDL_GetError().decode(errors="replace")

    q = sdl2.SDL_HapticQuery(haptic)
    print(f"Haptic: axes={sdl2.SDL_HapticNumAxes(haptic)} max_effects={sdl2.SDL_HapticNumEffects(haptic)} "
          f"gain_supported={bool(q & sdl2.SDL_HAPTIC_GAIN)} "
          f"autocenter_supported={bool(q & sdl2.SDL_HAPTIC_AUTOCENTER)}")

    if q & sdl2.SDL_HAPTIC_GAIN:
        print("SetGain(100) ->", sdl2.SDL_HapticSetGain(haptic, 100), err())
    if q & sdl2.SDL_HAPTIC_AUTOCENTER:
        print("SetAutocenter(0) ->", sdl2.SDL_HapticSetAutocenter(haptic, 0), err())

    def make(dir_type, level):
        eff = sdl2.SDL_HapticEffect()
        eff.type = sdl2.SDL_HAPTIC_CONSTANT
        eff.constant.type = sdl2.SDL_HAPTIC_CONSTANT
        eff.constant.direction.type = dir_type
        if dir_type == sdl2.SDL_HAPTIC_CARTESIAN:
            eff.constant.direction.dir[0] = 1
        else:  # polar: 9000 = east / along +x
            eff.constant.direction.dir[0] = 9000
        eff.constant.length = 1500  # ms
        eff.constant.level = level
        return eff

    input("Close the Thrustmaster control panel / joy.cpl if open.\n"
          "Hold the wheel LIGHTLY, then press Enter...")
    try:
        for dname, dtype in (("CARTESIAN", sdl2.SDL_HAPTIC_CARTESIAN),
                             ("POLAR", sdl2.SDL_HAPTIC_POLAR)):
            for label, level in (("+level", LEVEL), ("-level", -LEVEL)):
                eff = make(dtype, level)
                eid = sdl2.SDL_HapticNewEffect(haptic, eff)
                if eid < 0:
                    print(f"{dname} {label}: NewEffect failed: {err()}")
                    continue
                rc = sdl2.SDL_HapticRunEffect(haptic, eid, 1)
                print(f"{dname} {label}: run -> {rc} {err() if rc < 0 else ''}  "
                      "(note which way the wheel goes)")
                time.sleep(1.8)
                sdl2.SDL_HapticDestroyEffect(haptic, eid)
                time.sleep(0.5)

        if sdl2.SDL_HapticRumbleSupported(haptic) == 1:
            sdl2.SDL_HapticRumbleInit(haptic)
            print("Rumble test (wheel should vibrate) ->",
                  sdl2.SDL_HapticRumblePlay(haptic, 0.5, 1000), err())
            time.sleep(1.2)
    finally:
        sdl2.SDL_HapticStopAll(haptic)
        sdl2.SDL_HapticClose(haptic)
    print("Done. Tell Claude what moved (and which way) for each line, "
          "and paste the full output.")


def test_vjoy():
    """Sweep vJoy's X axis and log any force-feedback packets a game sends to it."""
    import pyvjoy
    j = pyvjoy.VJoyDevice(1)

    if j.ffb_supported():
        log = open("ffb_log.txt", "w")

        def ffbcb(data, reptype):
            # keep this tiny: it runs on vJoy's thread
            try:
                packetdict, ebi = pyvjoy.FFB_Effect_Manager.ffb_packet_to_dict(data, reptype)
                log.write(f"{time.time():.3f} ebi={ebi} {packetdict}\n")
            except Exception as e:  # never let the callback crash
                log.write(f"{time.time():.3f} parse error {e}\n")

        j.ffb_register_callback(ffbcb)
        print("FFB is enabled on vJoy device 1. Packets -> ffb_log.txt")
    else:
        print("WARNING: FFB not enabled on vJoy device 1 "
              "(Configure vJoy -> tick 'Enable Effects').")

    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(f"Sweeping X axis left/right for {seconds}s "
          "(watch joy.cpl -> vJoy Device, or the car's front wheels in Forza)")
    # Pedals: brake on Y, gas on Z, inverted like the TMX (0x8000 = released).
    j.set_axis(pyvjoy.HID_USAGE_Y, 0x8000)
    j.set_axis(pyvjoy.HID_USAGE_Z, 0x8000)
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            for v in (0x1, 0x4000, 0x8000, 0x4000):
                j.set_axis(pyvjoy.HID_USAGE_X, v)
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
    print("Done. vJoy axis works if it moved.")


def test_bind():
    """Move ONE vJoy axis on a timer so Forza's 'press input to bind' screen can catch it.

    python check.py bind steer   (X axis: centre -> full right -> centre)
    python check.py bind brake   (Y axis: released -> floored -> released)
    python check.py bind gas     (Z axis: released -> floored -> released)
    """
    import pyvjoy
    target = sys.argv[2] if len(sys.argv) > 2 else ""
    axes = {"steer": pyvjoy.HID_USAGE_X, "brake": pyvjoy.HID_USAGE_Y, "gas": pyvjoy.HID_USAGE_Z}
    if target not in axes:
        sys.exit("Usage: python check.py bind steer|brake|gas")
    j = pyvjoy.VJoyDevice(1)
    # rest position for everything: steering centred, pedals released
    j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
    j.set_axis(pyvjoy.HID_USAGE_Y, 0x8000)
    j.set_axis(pyvjoy.HID_USAGE_Z, 0x8000)

    start, end = (0x4000, 0x8000) if target == "steer" else (0x8000, 0x1)
    for s in range(6, 0, -1):
        print(f"\rSwitch to Forza and select the '{target}' row now... moving in {s}s ",
              end="", flush=True)
        time.sleep(1)
    print(f"\nMoving {target}...")
    steps = 40
    for i in range(steps + 1):          # slow push over ~1s
        j.set_axis(axes[target], int(start + (end - start) * i / steps))
        time.sleep(0.025)
    time.sleep(1.0)                     # hold
    j.set_axis(axes[target], start)     # back to rest
    print("Done. Check what Forza bound.")


def test_bindall():
    """Press vJoy buttons one after another, so you can bind Forza rows without alt-tabbing.

    python check.py bindall [first_button] [seconds_between]
    e.g. python check.py bindall 1 5   -> presses button 1, then 2, 3 ... every 5 s

    In Forza: highlight the first UNDEFINED row and press Enter before the first press.
    After each press, move down one row and press Enter again before the next one.
    Ctrl+C in the terminal stops it. Writes the button numbers it pressed to bind_log.txt.
    """
    import pyvjoy
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    gap = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    j = pyvjoy.VJoyDevice(1)
    j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
    j.set_axis(pyvjoy.HID_USAGE_Y, 0x8000)
    j.set_axis(pyvjoy.HID_USAGE_Z, 0x8000)
    j.reset_buttons()
    log = open("bind_log.txt", "a")
    for s in range(8, 0, -1):
        print(f"\rSwitch to Forza, highlight the first row and press Enter... {s}s ",
              end="", flush=True)
        time.sleep(1)
    print()
    b = first
    try:
        while b <= 32:
            print(f"Pressing button {b}")
            log.write(f"button {b}\n"); log.flush()
            j.set_button(b, 1)
            time.sleep(0.3)
            j.set_button(b, 0)
            b += 1
            time.sleep(gap)
    except KeyboardInterrupt:
        pass
    j.reset_buttons()
    print(f"Stopped. Last button pressed: {b - 1}")


def test_telemetry(port=9999):
    """Listen for Forza Horizon 4 Data Out packets and print a few fields."""
    import socket
    import struct
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    s.settimeout(5)
    print(f"Listening on 127.0.0.1:{port}. Drive in Forza. Ctrl+C to stop.\n")
    count, t0 = 0, time.time()
    try:
        while True:
            try:
                data, _ = s.recvfrom(1024)
            except socket.timeout:
                print("No packets for 5s. Check Data Out is On, IP 127.0.0.1, port 9999.")
                continue
            count += 1
            # FH4 'dash' layout (324 bytes): sled block, 12 Horizon-only bytes, dash block.
            # Verify: rpm and speed should match the in-game HUD.
            race_on = struct.unpack_from("<i", data, 0)[0]
            rpm = struct.unpack_from("<f", data, 16)[0]
            speed = struct.unpack_from("<f", data, 256)[0] * 3.6 if len(data) >= 260 else float("nan")
            steer = struct.unpack_from("<b", data, 320)[0] if len(data) >= 321 else 0
            rate = count / max(time.time() - t0, 1e-6)
            print(f"\rlen={len(data)}  on={race_on}  rpm={rpm:7.0f}  "
                  f"speed={speed:6.1f} km/h  steer={steer:+4d}  ({rate:5.1f} pkt/s)   ",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\nDone.")


TESTS = {"wheel": test_wheel, "ffb": test_ffb, "vjoy": test_vjoy,
         "bind": test_bind, "bindall": test_bindall, "telemetry": test_telemetry}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TESTS:
        print(__doc__)
        sys.exit(1)
    TESTS[sys.argv[1]]()
