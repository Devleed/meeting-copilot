# =============================================================================
# COMPONENT: audio_device.py
#
# Locates the BlackHole virtual audio device by scanning the system's audio
# device list. BlackHole is a macOS virtual audio device that mirrors system
# audio output into a virtual input channel that Python can capture with
# sounddevice.
#
# Single Responsibility: find the integer device index of the BlackHole input
# device. Nothing else — no stream management, no audio processing.
#
# SOLID notes:
#   S — sole job: detect the BlackHole device index.
#   O — the fallback index is injected; changing it requires no code edit.
#   D — depends on sounddevice (an external lib) and a fallback index injected
#       at construction; no module-level globals.
# =============================================================================

import sounddevice as sd  # sounddevice — wraps PortAudio; sd.query_devices() lists all audio devices


class BlackHoleDeviceLocator:
    """
    Scans the system's audio devices and returns the index of the BlackHole
    virtual input device.

    Responsibility: iterate over sounddevice's device list, find the first
    device whose name contains "blackhole" (case-insensitive) and which has
    at least one input channel, and return its integer index.

    Falls back to a configurable default index if BlackHole is not found,
    so the pipeline degrades gracefully on systems where BlackHole is not
    installed (e.g. CI, non-macOS machines).

    Usage:
        locator = BlackHoleDeviceLocator(fallback_index=1)
        device_index = locator.find()
    """

    def __init__(self, fallback_index: int) -> None:
        # fallback_index — the device index to use if BlackHole is not detected.
        # Stored as a private attribute; used only inside find().
        self._fallback_index: int = fallback_index

    def find(self) -> int:
        """
        Return the sounddevice index of the BlackHole input device.

        Returns
        -------
        int
            Index of the BlackHole device, or the fallback index if not found.
        """
        # sd.query_devices() — returns a list of dicts, one per audio device on
        # the system. Each dict contains keys such as:
        #   "name"               — human-readable device name string
        #   "max_input_channels" — number of input channels (0 for output-only devices)
        #   "max_output_channels"— number of output channels
        # In TS this would be a call to the Web Audio API or a native module.
        devices = sd.query_devices()

        # enumerate(devices) — yields (index, device_dict) pairs, like Array.entries() in TS.
        # `i` is the integer device index that sounddevice uses to open a stream.
        for i, device in enumerate(devices):
            # device["name"].lower() — lowercase the name for case-insensitive comparison.
            # "blackhole" in ... — substring membership test. True if "blackhole" appears
            # anywhere in the name. Catches "BlackHole 2ch", "BlackHole 16ch", etc.
            # In TS: device.name.toLowerCase().includes("blackhole")
            #
            # device["max_input_channels"] > 0 — only consider devices that can receive
            # audio input. BlackHole appears as both an input device and an output device;
            # we need the input side so sounddevice can read system audio from it.
            name_matches: bool = "blackhole" in device["name"].lower()
            has_input: bool = device["max_input_channels"] > 0

            if name_matches and has_input:
                return i  # i — the integer device index to pass to sd.InputStream

        # BlackHole was not found in the device list (not installed, or unusual name).
        # Warn the user but continue with the fallback index.
        print(f"BlackHole not found — falling back to device index {self._fallback_index}")
        return self._fallback_index
