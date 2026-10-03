"""Foreground gating without Windows, focus changes, or native device access."""

import ctypes
from ctypes import wintypes
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from forza_ai.foreground import ForegroundGameGuard, _Win32ForegroundApi


class Win32Rig:
    def __init__(self):
        self.window = 0x123456780001
        self.handle = 0x234567890002
        self.process_id = 1234
        self.path = r"C:\Games\ForzaHorizon4.exe"
        self.user32 = SimpleNamespace(
            GetForegroundWindow=Mock(return_value=self.window),
            GetWindowThreadProcessId=Mock(side_effect=self.process),
        )
        self.kernel32 = SimpleNamespace(
            OpenProcess=Mock(return_value=self.handle),
            QueryFullProcessImageNameW=Mock(side_effect=self.query),
            CloseHandle=Mock(return_value=1),
        )
        self.native = _Win32ForegroundApi(user32=self.user32, kernel32=self.kernel32)

    def process(self, window, process_id):
        ctypes.cast(process_id, ctypes.POINTER(wintypes.DWORD)).contents.value = self.process_id
        return 4321

    def query(self, handle, flags, buffer, length):
        buffer.value = self.path
        ctypes.cast(length, ctypes.POINTER(wintypes.DWORD)).contents.value = len(self.path)
        return 1

    def guard(self, name="ForzaHorizon4.exe"):
        with patch("forza_ai.foreground.sys.platform", "win32"), \
             patch("forza_ai.foreground._Win32ForegroundApi", return_value=self.native):
            return ForegroundGameGuard(name)


class ForegroundTests(unittest.TestCase):
    def setUp(self):
        self.rig = Win32Rig()

    def test_game_executable_matches_case_insensitively(self):
        self.rig.path = r"C:\Games\FORZAHORIZON4.EXE"
        self.assertTrue(self.rig.guard().is_active())
        self.rig.kernel32.OpenProcess.assert_called_once_with(0x1000, False, 1234)
        self.rig.kernel32.CloseHandle.assert_called_once_with(self.rig.handle)

    def test_only_exact_executable_basename_matches(self):
        guard = self.rig.guard()
        for path in (r"C:\ForzaHorizon4.exe\explorer.exe", r"C:\Games\NotForzaHorizon4.exe",
                     r"C:\Games\ForzaHorizon4.exe.exe", r"C:\Games\ApplicationFrameHost.exe"):
            with self.subTest(path=path):
                self.rig.path = path
                self.assertFalse(guard.is_active())
        self.assertEqual(self.rig.kernel32.CloseHandle.call_count, 4)

    def test_explicit_alternative_game_executable(self):
        self.rig.path = r"C:\Games\ForzaHorizon5.exe"
        self.assertTrue(self.rig.guard("forzahorizon5.exe").is_active())

    def test_no_foreground_window_does_not_open_process(self):
        self.rig.user32.GetForegroundWindow.return_value = None
        self.assertFalse(self.rig.guard().is_active())
        self.rig.user32.GetWindowThreadProcessId.assert_not_called()
        self.rig.kernel32.OpenProcess.assert_not_called()

    def test_unknown_process_does_not_open_handle(self):
        self.rig.process_id = 0
        self.assertFalse(self.rig.guard().is_active())
        self.rig.kernel32.OpenProcess.assert_not_called()
        self.rig.kernel32.CloseHandle.assert_not_called()

    def test_failed_window_query_ignores_output_process_id(self):
        self.rig.user32.GetWindowThreadProcessId.side_effect = lambda window, pid: 0
        self.assertFalse(self.rig.guard().is_active())
        self.rig.kernel32.OpenProcess.assert_not_called()

    def test_inaccessible_process_is_inactive_without_closing_null_handle(self):
        self.rig.kernel32.OpenProcess.return_value = None
        self.assertFalse(self.rig.guard().is_active())
        self.rig.kernel32.QueryFullProcessImageNameW.assert_not_called()
        self.rig.kernel32.CloseHandle.assert_not_called()

    def test_failed_image_query_always_closes_open_handle(self):
        self.rig.kernel32.QueryFullProcessImageNameW.side_effect = lambda *args: 0
        self.assertFalse(self.rig.guard().is_active())
        self.rig.kernel32.CloseHandle.assert_called_once_with(self.rig.handle)

    def test_native_query_error_fails_closed_and_closes_handle(self):
        self.rig.kernel32.QueryFullProcessImageNameW.side_effect = OSError("process exited")
        self.assertFalse(self.rig.guard().is_active())
        self.rig.kernel32.CloseHandle.assert_called_once_with(self.rig.handle)

    def test_empty_or_truncated_successful_query_is_inactive_and_closes(self):
        def query_with_length(size):
            def query(handle, flags, buffer, length):
                ctypes.cast(length, ctypes.POINTER(wintypes.DWORD)).contents.value = size
                return 1
            return query
        guard = self.rig.guard()
        for size in (0, self.rig.native.MAX_IMAGE_PATH):
            self.rig.kernel32.QueryFullProcessImageNameW.side_effect = query_with_length(size)
            self.assertFalse(guard.is_active())
        self.assertEqual(self.rig.kernel32.CloseHandle.call_count, 2)

    def test_alt_tab_during_query_fails_closed(self):
        self.rig.user32.GetForegroundWindow.side_effect = [self.rig.window, self.rig.window + 1]
        self.assertFalse(self.rig.guard().is_active())
        self.rig.kernel32.CloseHandle.assert_called_once_with(self.rig.handle)

    def test_reused_window_with_changed_process_fails_closed(self):
        def query_and_change_pid(*args):
            result = self.rig.query(*args)
            self.rig.process_id += 1
            return result
        self.rig.kernel32.QueryFullProcessImageNameW.side_effect = query_and_change_pid
        self.assertFalse(self.rig.guard().is_active())
        self.rig.kernel32.CloseHandle.assert_called_once_with(self.rig.handle)

    def test_every_check_requeries_process_no_stale_cache(self):
        guard = self.rig.guard()
        self.assertTrue(guard.is_active())
        self.rig.path = r"C:\Windows\explorer.exe"
        self.assertFalse(guard.is_active())
        self.assertEqual(self.rig.kernel32.OpenProcess.call_count, 2)
        self.assertEqual(self.rig.kernel32.CloseHandle.call_count, 2)

    def test_native_prototypes_preserve_handle_width(self):
        self.assertIs(self.rig.user32.GetForegroundWindow.restype, wintypes.HWND)
        self.assertIs(self.rig.kernel32.OpenProcess.restype, wintypes.HANDLE)
        self.assertEqual(ctypes.sizeof(wintypes.HANDLE), ctypes.sizeof(ctypes.c_void_p))
        self.assertEqual(self.rig.kernel32.OpenProcess.argtypes,
                         [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD])
        self.assertEqual(self.rig.kernel32.QueryFullProcessImageNameW.argtypes,
                         [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)])
        self.assertEqual(self.rig.kernel32.CloseHandle.argtypes, [wintypes.HANDLE])

    def test_invalid_names_fail_before_native_loading(self):
        names = ("", " ", ".exe", "Forza", "ForzaHorizon4.exe ", " Forza.exe", None, 1,
                 r"C:\Games\ForzaHorizon4.exe", "Games/ForzaHorizon4.exe", "*.exe", "x\x00.exe")
        with patch("forza_ai.foreground._Win32ForegroundApi") as native:
            for name in names:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "basename"):
                    ForegroundGameGuard(name)
        native.assert_not_called()

    def test_non_windows_error_precedes_native_loading(self):
        with patch("forza_ai.foreground.sys.platform", "darwin"), \
             patch("forza_ai.foreground._Win32ForegroundApi") as native, \
             self.assertRaisesRegex(RuntimeError, "requires Windows"):
            ForegroundGameGuard()
        native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
