"""Read-only preflight discovery with no native libraries or connected devices."""

from contextlib import redirect_stdout
import importlib.metadata
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from forza_ai.preflight import build_report, main


class DiscoveryRig:
    def __init__(self):
        # Deliberately expose only permitted read operations; any acquisition,
        # haptic handle, effect, force/gain, or input write is an AttributeError.
        self.sdl = SimpleNamespace(
            SDL_INIT_JOYSTICK=0x200,
            SDL_InitSubSystem=Mock(return_value=0),
            SDL_QuitSubSystem=Mock(),
            SDL_NumJoysticks=Mock(return_value=2),
            SDL_JoystickNameForIndex=Mock(side_effect=lambda i: [b"vJoy", b"Thrustmaster TMX"][i]),
            SDL_GetError=Mock(return_value=b"test SDL failure"),
        )
        self.vjoy = SimpleNamespace(VJD_STAT_FREE=1, HID_USAGE_X=0x30, HID_USAGE_Y=0x31, HID_USAGE_Z=0x32)
        self.native = SimpleNamespace(GetVJDAxisExist=Mock(return_value=1))
        self.sdk = SimpleNamespace(vJoyEnabled=Mock(return_value=True),
                                   GetVJDStatus=Mock(return_value=1), _vj=self.native)
        self.modules = {"sdl2": self.sdl, "pyvjoy": self.vjoy, "pyvjoy._sdk": self.sdk}
        self.importer = Mock(side_effect=self.modules.__getitem__)

    def report(self, **kwargs):
        with patch("forza_ai.preflight.sys.platform", "win32"), \
             patch("forza_ai.preflight._packages", return_value=[]), \
             patch("forza_ai.preflight.importlib.import_module", self.importer):
            return build_report(**kwargs)


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.rig = DiscoveryRig()

    def test_ready_uses_only_name_enumeration_and_balanced_subsystem_lifetime(self):
        report = self.rig.report(takeover_button=4)
        self.assertTrue(report["ready"])
        self.assertEqual(report["scope"], "read_only_device_discovery")
        self.assertTrue(report["vjoy"]["exists"])
        self.assertEqual(report["vjoy"]["axes"], {"X": True, "Y": True, "Z": True})
        self.rig.sdl.SDL_InitSubSystem.assert_called_once_with(0x200)
        self.rig.sdl.SDL_QuitSubSystem.assert_called_once_with(0x200)
        self.assertEqual(self.rig.native.GetVJDAxisExist.call_args_list,
                         [call(1, 0x30), call(1, 0x31), call(1, 0x32)])
        self.assertIsNone(report["tmx"]["devices"][1]["buttons"])
        self.assertIsNone(report["tmx"]["devices"][1]["axes"])
        self.assertIsNone(report["tmx"]["devices"][1]["haptic_capable"])
        self.assertFalse(report["tmx"]["devices"][1]["details_checked"])
        self.assertFalse(report["tmx"]["takeover_button_verified"])
        self.assertEqual(report["tmx"]["known_mapping"]["throttle_axis"], 2)
        self.assertIn("Forza", " ".join(report["not_checked"]))
        self.assertIn("HidHide", " ".join(report["not_checked"]))
        self.assertIn("Constant-force", " ".join(report["not_checked"]))

    def test_non_windows_never_imports_hardware_packages(self):
        packages = [{"distribution": "torch", "version": "test", "expected_version": None}]
        with patch("forza_ai.preflight.sys.platform", "darwin"), \
             patch("forza_ai.preflight._packages", return_value=packages), \
             patch("forza_ai.preflight.importlib.import_module") as importer:
            report = build_report()
        importer.assert_not_called()
        self.assertFalse(report["ready"])
        self.assertFalse(report["tmx"]["checked"])
        self.assertFalse(report["vjoy"]["checked"])
        self.assertEqual(report["packages"], packages)
        self.assertIn("requires Windows", report["summary"])

    def test_other_thrustmaster_models_not_opened(self):
        self.rig.sdl.SDL_JoystickNameForIndex.side_effect = lambda i: [b"vJoy", b"Thrustmaster T150"][i]
        report = self.rig.report()
        self.assertFalse(report["ready"])
        self.assertIn("No calibrated TMX", report["tmx"]["errors"][0])

    def test_takeover_range_is_explicitly_unverified_even_with_candidate(self):
        report = self.rig.report(takeover_button=120)
        self.assertEqual(report["tmx"]["takeover_button"], 120)
        self.assertFalse(report["tmx"]["takeover_button_verified"])
        self.assertIn("takeover-button range", " ".join(report["not_checked"]))

    def test_enumeration_error_closes_subsystem(self):
        self.rig.sdl.SDL_NumJoysticks.return_value = -1
        report = self.rig.report()
        self.assertFalse(report["ready"])
        self.assertIn("test SDL failure", report["tmx"]["errors"][0])
        self.rig.sdl.SDL_QuitSubSystem.assert_called_once()

    def test_failed_init_does_not_quit_uninitialized_subsystem(self):
        self.rig.sdl.SDL_InitSubSystem.return_value = -1
        report = self.rig.report()
        self.assertFalse(report["ready"])
        self.rig.sdl.SDL_QuitSubSystem.assert_not_called()
        self.assertTrue(report["vjoy"]["ready"])

    def test_subsystem_cleanup_failure_is_not_ready(self):
        self.rig.sdl.SDL_QuitSubSystem.side_effect = RuntimeError("quit failed")
        report = self.rig.report()
        self.assertFalse(report["ready"])
        self.rig.sdl.SDL_QuitSubSystem.assert_called_once()
        self.assertIn("quit failed", report["tmx"]["errors"][0])

    def test_wheel_open_cannot_be_called_even_if_api_is_available(self):
        self.rig.sdl.SDL_JoystickOpen = Mock(side_effect=AssertionError("Opening can engage centering torque"))
        self.rig.sdl.SDL_HapticOpenFromJoystick = Mock(side_effect=AssertionError("No haptic acquisition"))
        self.assertTrue(self.rig.report()["ready"])
        self.rig.sdl.SDL_JoystickOpen.assert_not_called()
        self.rig.sdl.SDL_HapticOpenFromJoystick.assert_not_called()

    def test_import_system_exit_from_missing_vjoy_dll_becomes_report_error(self):
        def import_module(name):
            if name == "pyvjoy":
                raise SystemExit("missing SDK DLL")
            return self.rig.modules[name]
        self.rig.importer.side_effect = import_module
        report = self.rig.report()
        self.assertFalse(report["ready"])
        self.assertTrue(report["tmx"]["ready"])
        self.assertIn("missing SDK DLL", report["vjoy"]["errors"][0])

    def test_missing_sdl_does_not_prevent_vjoy_diagnostics(self):
        def import_module(name):
            if name == "sdl2":
                raise ImportError("SDL missing")
            return self.rig.modules[name]
        self.rig.importer.side_effect = import_module
        report = self.rig.report()
        self.assertFalse(report["ready"])
        self.assertTrue(report["vjoy"]["ready"])

    def test_busy_missing_and_unknown_vjoy_statuses_are_not_ready(self):
        for code, expected_exists in ((0, True), (2, True), (3, False), (4, None), (99, None)):
            with self.subTest(code=code):
                self.rig.sdk.GetVJDStatus.return_value = code
                report = self.rig.report()
                self.assertFalse(report["ready"])
                self.assertEqual(report["vjoy"]["exists"], expected_exists)
                self.assertFalse(report["vjoy"]["free"])

    def test_absent_axis_is_reported_without_writing_it(self):
        self.rig.native.GetVJDAxisExist.side_effect = lambda device, axis: axis != 0x31
        report = self.rig.report(vjoy_device_id=2)
        self.assertFalse(report["ready"])
        self.assertFalse(report["vjoy"]["axes"]["Y"])
        self.rig.sdk.GetVJDStatus.assert_called_once_with(2)
        self.assertIn("Y axis", report["vjoy"]["errors"][0])

    def test_driver_disabled_does_not_query_nonexistent_device(self):
        self.rig.sdk.vJoyEnabled.return_value = False
        report = self.rig.report()
        self.assertFalse(report["ready"])
        self.rig.sdk.GetVJDStatus.assert_not_called()
        self.rig.native.GetVJDAxisExist.assert_not_called()

    def test_cli_json_and_exit_codes_match_report(self):
        for ready in (True, False):
            output = io.StringIO()
            report = {"ready": ready, "scope": "read_only_device_discovery"}
            with patch("forza_ai.preflight.build_report", return_value=report) as build, redirect_stdout(output):
                exit_code = main(["--json", "--vjoy-device-id", "2", "--takeover-button", "4"])
            self.assertEqual(exit_code, 0 if ready else 1)
            self.assertEqual(json.loads(output.getvalue()), report)
            build.assert_called_once_with(2, 4)

    def test_package_discovery_never_imports_modules(self):
        from forza_ai.preflight import _packages
        with patch("forza_ai.preflight.importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError), \
             patch("forza_ai.preflight.importlib.util.find_spec", return_value=None), \
             patch("forza_ai.preflight.importlib.import_module") as importer:
            packages = _packages()
        importer.assert_not_called()
        self.assertTrue(all(p["version"] is None and not p["module_found"] for p in packages))
        self.assertIn("pyvjoyffb", {p["distribution"] for p in packages})

    def test_invalid_arguments_fail_before_any_discovery(self):
        for kwargs in ({"vjoy_device_id": 0}, {"vjoy_device_id": 17},
                       {"takeover_button": -1}, {"takeover_button": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_report(**kwargs)


if __name__ == "__main__":
    unittest.main()
