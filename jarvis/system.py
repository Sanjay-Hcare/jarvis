"""Checks of the computer's state that explain slow replies before they happen."""

from __future__ import annotations

import ctypes
import os

# Voice mode needs roughly this much free memory of its own to run smoothly.
LOW_MEMORY_GB = 3.0


def on_battery() -> bool | None:
    """True when running on battery, False on mains power, None if unknown."""
    if os.name != "nt":
        return None

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    # 0 = on battery, 1 = plugged in, 255 = unknown (for example a desktop).
    return {0: True, 1: False}.get(status.ACLineStatus)


def free_memory_gb() -> float | None:
    """Physical memory currently free, in gigabytes, or None if unknown."""
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPhys / 1024**3


def performance_warnings(battery: bool | None, free_gb: float | None) -> list[str]:
    """Plain-language reasons the replies may be slow right now."""
    warnings = []
    if battery:
        warnings.append(
            "Running on battery: the laptop slows its graphics card and processor down, so replies "
            "can take several times longer. Plug in the charger for full speed."
        )
    if free_gb is not None and free_gb < LOW_MEMORY_GB:
        warnings.append(
            f"Only {free_gb:.1f} GB of memory is free, so replies may be slow while Windows makes room. "
            "Closing other apps helps."
        )
    return warnings


def current_warnings() -> list[str]:
    try:
        return performance_warnings(on_battery(), free_memory_gb())
    except (OSError, AttributeError, ValueError):
        # A warning is a courtesy; never let it stop the assistant starting.
        return []
