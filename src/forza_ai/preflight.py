"""Read-only discovery before a Windows wheel-control acceptance test.

    python -m forza_ai.preflight --json --takeover-button 0

Exit 0 means the device discovery checks here passed; exit 1 means they
did not. Neither result certifies Forza bindings, HidHide, calibrated physical
direction, constant-force support, or motor stability. This command never opens
a joystick/haptic handle, acquires vJoy, runs an effect, or changes virtual inputs.
"""

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import sys


_PACKAGES = (
    ("PySDL2", "sdl2", "0.9.17"),
    ("pysdl2-dll", "sdl2dll", "2.32.10"),
    ("pyvjoyffb", "pyvjoy", "0.4"),
    ("dxcam", "dxcam", None),
    ("numpy", "numpy", None),
    ("Pillow", "PIL", None),
    ("torch", "torch", None),
)

_UNCHECKED = (
    "Physical axes/buttons, takeover-button range, attachment, and haptic support are unverified: SDL_JoystickOpen can enable autocenter, so this read-only command never opens the wheel.",
    "Forza wheel/axis bindings, assists, and Data Out settings live inside the game; this command does not inspect them.",
    "HidHide visibility/allowlist must be checked for both this Python process and Forza; device discovery alone cannot certify it.",
    "TMX rotation, pedal calibration, physical torque direction, and takeover-button identity require manual verification.",
    "Constant-force support, effect expiry, and steering stability require a separate bounded hardware acceptance test; no haptic handle is opened here.",
)


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def _packages() -> list[dict]:
    """Inspect metadata/module paths without importing heavy or native packages."""
    packages = []
    for distribution, module, expected in _PACKAGES:
        entry = {"distribution": distribution, "module": module, "version": None,
                 "module_found": None, "expected_version": expected}
        try:
            entry["version"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
        except Exception as error:
            entry["metadata_error"] = _error_text(error)
        try:
            entry["module_found"] = importlib.util.find_spec(module) is not None
        except Exception as error:
            entry["discovery_error"] = _error_text(error)
        packages.append(entry)
    return packages


def _sdl_error(sdl, operation: str) -> RuntimeError:
    detail = (sdl.SDL_GetError() or b"no SDL error detail").decode(errors="replace")
    return RuntimeError(f"{operation}: {detail}")


def _tmx_report(takeover_button: int | None) -> dict:
    report = {"checked": True, "ready": False, "devices": [], "selected_index": None,
              "errors": [], "haptic_check": "not checked: opening the joystick can alter force state",
              "readiness_note": "TMX name discovery only; physical counts and capabilities are unverified.",
              "known_mapping": {"steering_axis": 0, "brake_axis": 1, "throttle_axis": 2,
                                "steering_right_positive": True, "pedals_inverted": True,
                                "nominal_rotation_deg": 900},
              "mapping_source": "project calibration contract, not measured by this command",
              "takeover_button": takeover_button, "takeover_button_verified": False}
    sdl = None
    initialized = False
    try:
        sdl = importlib.import_module("sdl2")
        if sdl.SDL_InitSubSystem(sdl.SDL_INIT_JOYSTICK) < 0:
            raise _sdl_error(sdl, "SDL_InitSubSystem(JOYSTICK)")
        initialized = True
        count = sdl.SDL_NumJoysticks()
        if count < 0:
            raise _sdl_error(sdl, "SDL_NumJoysticks")
        for index in range(count):
            name_bytes = sdl.SDL_JoystickNameForIndex(index)
            if name_bytes is None:
                raise _sdl_error(sdl, "SDL_JoystickNameForIndex")
            name = name_bytes.decode(errors="replace")
            match = "tmx" in name.lower() and "vjoy" not in name.lower()
            device = {"index": index, "name": name, "tmx_match": match}
            report["devices"].append(device)
            if not match:
                continue
            if report["selected_index"] is None:
                report["selected_index"] = index  # Matches WindowsAdapter selection.
            # SDL2's DirectInput JoystickOpen acquires the device, resets force
            # feedback, and enables autocenter. Even capability-only checks
            # using a joystick handle therefore violate read-only discovery.
            # https://github.com/libsdl-org/SDL/blob/release-2.32.10/src/joystick/windows/SDL_dinputjoystick.c#L766-L864
            device.update(axes=None, buttons=None, attached=None, haptic_capable=None,
                          details_checked=False)
        if report["selected_index"] is None:
            report["errors"].append("No calibrated TMX found. Check power, USB, and Python's HidHide visibility.")
        else:
            report["ready"] = True
        if sum(d["tmx_match"] for d in report["devices"]) > 1:
            report["warning"] = "Multiple TMX devices found; runtime uses the first matching SDL index."
    except (Exception, SystemExit) as error:
        report["errors"].append(_error_text(error))
        report["ready"] = False
    finally:
        if initialized:
            try:
                sdl.SDL_QuitSubSystem(sdl.SDL_INIT_JOYSTICK)
            except Exception as error:
                report["errors"].append(_error_text(error))
                report["ready"] = False
    return report


def _vjoy_report(device_id: int) -> dict:
    report = {"checked": True, "ready": False, "device_id": device_id, "enabled": None,
              "status": "unknown", "status_code": None, "free": False, "exists": None,
              "axes": {"X": None, "Y": None, "Z": None}, "errors": []}
    try:
        pyvjoy = importlib.import_module("pyvjoy")
        sdk = importlib.import_module("pyvjoy._sdk")
        report["enabled"] = bool(sdk.vJoyEnabled())
        if not report["enabled"]:
            report["errors"].append("vJoy driver is not enabled")
            return report
        status = int(sdk.GetVJDStatus(device_id))
        report["status_code"] = status
        report["status"] = {0: "owned_by_this_process", 1: "free", 2: "busy", 3: "missing", 4: "unknown"}.get(status, "unknown")
        report["free"] = status == pyvjoy.VJD_STAT_FREE
        report["exists"] = True if status in (0, 1, 2) else False if status == 3 else None
        if not report["free"]:
            report["errors"].append(f"vJoy device {device_id} is {report['status']}; the adapter requires a free device")
        # pyvjoyffb 0.4 has no Python wrapper for this read-only SDK function.
        # Signature: BOOL GetVJDAxisExist(UINT rID, UINT Axis); integer arguments
        # only, no pointer outputs or changes to ctypes DLL function signatures.
        # https://github.com/jshafer817/vJoy/blob/master/SDK/inc/vjoyinterface.h
        for axis in ("X", "Y", "Z"):
            usage = getattr(pyvjoy, f"HID_USAGE_{axis}")
            present = bool(sdk._vj.GetVJDAxisExist(device_id, usage))
            report["axes"][axis] = present
            if not present:
                report["errors"].append(f"vJoy {axis} axis was not found; enable X steering, Y brake, and Z throttle")
        report["ready"] = report["free"] and all(report["axes"].values()) and not report["errors"]
    except (Exception, SystemExit) as error:
        report["errors"].append(_error_text(error))
    return report


def build_report(vjoy_device_id: int = 1, takeover_button: int | None = None) -> dict:
    """Return JSON-safe read-only discovery results; do not start control loops."""
    if type(vjoy_device_id) is not int or not 1 <= vjoy_device_id <= 16:
        raise ValueError("vjoy_device_id must be an integer from 1 to 16")
    if takeover_button is not None and (type(takeover_button) is not int or takeover_button < 0):
        raise ValueError("takeover_button must be a nonnegative SDL button index")
    windows = sys.platform == "win32"
    report = {
        "schema_version": 1, "ready": False, "scope": "read_only_device_discovery",
        "platform": {"system": platform.system(), "sys_platform": sys.platform,
                     "release": platform.release(), "machine": platform.machine(),
                     "python": platform.python_version(), "hardware_supported": windows},
        "packages": _packages(), "warnings": [], "not_checked": list(_UNCHECKED),
        "package_check_note": "Module discovery and installed versions do not prove importability; only SDL/vJoy probes import their packages on Windows.",
    }
    for package in report["packages"]:
        if package["expected_version"] and package["version"] not in (None, package["expected_version"]):
            report["warnings"].append(f"{package['distribution']} {package['version']} differs from the tested version {package['expected_version']}")
    if not windows:
        reason = "Physical TMX/vJoy control requires Windows; only package discovery ran on this OS."
        report["tmx"] = {"checked": False, "ready": False, "errors": [reason]}
        report["vjoy"] = {"checked": False, "ready": False, "errors": [reason]}
        report["summary"] = reason
        return report
    report["tmx"] = _tmx_report(takeover_button)
    report["vjoy"] = _vjoy_report(vjoy_device_id)
    report["ready"] = report["tmx"]["ready"] and report["vjoy"]["ready"]
    report["summary"] = ("Read-only discovery checks passed; TMX capabilities, manual configuration, and bounded movement acceptance remain unverified."
                         if report["ready"] else "Read-only device checks did not pass; inspect TMX/vJoy errors before movement testing.")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the complete JSON report")
    parser.add_argument("--vjoy-device-id", type=int, default=1)
    parser.add_argument("--takeover-button", type=int, help="record a candidate SDL index; range validation requires opening the wheel later")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.vjoy_device_id, args.takeover_button)
    except ValueError as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(report["summary"])
        print(f"OS: {report['platform']['system']} | Python {report['platform']['python']}")
        for package in report["packages"]:
            print(f"{package['distribution']}: version={package['version'] or 'not found'}, module_found={package['module_found']}")
        for section in ("tmx", "vjoy"):
            print(f"{section.upper()}: {json.dumps(report[section])}")
        for warning in report["warnings"]:
            print(f"Version note: {warning}")
        print("Still requires verification:")
        for item in report["not_checked"]:
            print(f"- {item}")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
