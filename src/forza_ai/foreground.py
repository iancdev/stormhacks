"""Read-only Windows foreground-process guard for live driving observations.

No window is activated or moved. An unavailable or changing foreground process
fails closed, so an Alt-Tabbed desktop image cannot authorize AI steering.
"""

import ctypes
from ctypes import wintypes
import ntpath
import sys


class _Win32ForegroundApi:
    """Small native boundary; DLLs are loaded only when a guard is created."""

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    MAX_IMAGE_PATH = 32768

    def __init__(self, *, user32=None, kernel32=None):
        self.user32 = user32 if user32 is not None else ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = kernel32 if kernel32 is not None else ctypes.WinDLL("kernel32", use_last_error=True)

        # Explicit handle return types prevent the ctypes default (C int) from
        # truncating HWND/HANDLE values in a 64-bit Python process.
        self.user32.GetForegroundWindow.argtypes = []
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel32.OpenProcess.restype = wintypes.HANDLE
        self.kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL

    def _process_id(self, window) -> int:
        process_id = wintypes.DWORD()
        if not self.user32.GetWindowThreadProcessId(window, ctypes.byref(process_id)):
            return 0
        return process_id.value

    def foreground_executable(self) -> str | None:
        window = self.user32.GetForegroundWindow()
        if not window:
            return None
        process_id = self._process_id(window)
        if not process_id:
            return None
        handle = self.kernel32.OpenProcess(self.PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
        if not handle:
            return None
        try:
            path = ctypes.create_unicode_buffer(self.MAX_IMAGE_PATH)
            length = wintypes.DWORD(len(path))
            if not self.kernel32.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(length)):
                return None
            if not 0 < length.value < len(path):
                return None
            # Focus can change during the process query. Do not accept an old
            # game window after the user has switched applications.
            if self.user32.GetForegroundWindow() != window or self._process_id(window) != process_id:
                return None
            return path[:length.value]
        finally:
            self.kernel32.CloseHandle(handle)


class ForegroundGameGuard:
    """Require the foreground window's executable to match a configured game.

    This is an observation guard, not an executable authenticity check. The
    caller must still enforce command expiry, takeover, and local motor limits.
    """

    def __init__(self, process_name: str = "ForzaHorizon4.exe"):
        if (not isinstance(process_name, str) or len(process_name) <= 4
                or process_name != process_name.strip()
                or not process_name.casefold().endswith(".exe")
                or any(character in '<>:"/\\|?*' or ord(character) < 32 for character in process_name)):
            raise ValueError("process_name must be a nonempty executable basename ending in .exe, without a path")
        if sys.platform != "win32":
            raise RuntimeError("ForegroundGameGuard requires Windows")
        self.process_name = process_name
        self._expected_name = process_name.casefold()
        self._native = _Win32ForegroundApi()

    def is_active(self) -> bool:
        """Return False if focus is absent, inaccessible, or not the game."""
        try:
            path = self._native.foreground_executable()
        except OSError:
            return False
        return bool(path and ntpath.basename(path).casefold() == self._expected_name)
