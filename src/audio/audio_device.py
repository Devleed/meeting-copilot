import platform
import threading
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly


def _find_blackhole(fallback_index: int) -> int:
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if "blackhole" in device["name"].lower() and device["max_input_channels"] > 0:
            print(f"Found BlackHole device: '{device['name']}' (index {i})")
            return i
    print(f"BlackHole not found — falling back to device index {fallback_index}")
    return fallback_index


def _find_pulse_device() -> int | str:
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if "pulse" in device["name"].lower() and device["max_input_channels"] > 0:
            print(f"Found PulseAudio device: '{device['name']}' (index {i})")
            return i
    print("PulseAudio device not found — falling back to 'default'")
    return "default"


class AudioDeviceLocator:
    """
    Locates the appropriate system-audio loopback input device for the current OS.

    macOS  → BlackHole virtual audio device
    Linux  → PulseAudio .monitor source
    other  → not supported; returns the fallback index with a warning
    """

    def __init__(self, fallback_index: int) -> None:
        self._fallback_index = fallback_index

    def find(self) -> int | str:
        os_name = platform.system()
        if os_name == "Darwin":
            return _find_blackhole(self._fallback_index)
        elif os_name == "Linux":
            return _find_pulse_device()
        else:
            print(
                f"Audio loopback capture is not supported on {os_name}. "
                f"Using fallback device index {self._fallback_index}."
            )
            return self._fallback_index


# Keep the old name as an alias so nothing outside this module breaks
BlackHoleDeviceLocator = AudioDeviceLocator


_WHISPER_RATE = 16000


def test_audio_capture(
    device_index: int | str,
    sample_rate: int = 16000,
    duration: float = 5.0,
    model_size: str = "base",
) -> bool:
    """
    Record `duration` seconds from `device_index`, print signal statistics,
    resample to 16 kHz if needed, and transcribe with Whisper.

    Intended to be called from the CLI with --test-audio before starting
    the full pipeline, so the user can verify their loopback device works
    and confirm that transcription produces sensible output.
    """
    print(f"\n[ Audio capture test — device {device_index}, {duration:.0f}s ]")
    print("  Make sure audio is playing on your system, then wait...\n")

    frames: list = []
    done = threading.Event()
    target_samples = int(sample_rate * duration)

    def _callback(indata, _frame_count, _time_info, _status):
        frames.append(indata.copy())
        if sum(f.shape[0] for f in frames) >= target_samples:
            done.set()

    try:
        with sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=sample_rate,
            dtype="float32",
            blocksize=1024,
            callback=_callback,
        ):
            done.wait(timeout=duration + 2.0)
    except Exception as exc:
        print(f"  FAIL — could not open device {device_index}: {exc}")
        _print_available_input_devices()
        return False

    if not frames:
        print("  FAIL — no audio frames received from device.")
        _print_available_input_devices()
        return False

    audio = np.concatenate([f.flatten() for f in frames])
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    print(f"  Samples captured : {len(audio)}")
    print(f"  RMS level        : {rms:.6f}")
    print(f"  Peak level       : {peak:.6f}")

    # 1e-4 ≈ -80 dBFS — anything below this is effectively digital silence
    if rms <= 1e-4:
        print(
            "\n  WARN — signal is very low or completely silent.\n"
            "         Check that audio is playing and the loopback device is\n"
            "         set as your system output (macOS: Audio MIDI Setup multi-output;\n"
            "         Linux: set the monitor source as the default input in pavucontrol).\n"
        )
        return False

    print("  PASS — audio signal detected.\n")

    # Resample to Whisper's required 16 kHz if the capture rate differs.
    if sample_rate != _WHISPER_RATE:
        g = gcd(sample_rate, _WHISPER_RATE)
        audio = resample_poly(audio, _WHISPER_RATE // g, sample_rate // g).astype(np.float32)

    from audio.speech_transcriber import SpeechTranscriber
    transcriber = SpeechTranscriber(model_size=model_size)

    print("[ Transcription ]")
    segments = transcriber.transcribe(audio)
    if segments:
        for line in segments:
            print(f"  {line}")
    else:
        print("  (no speech detected)")
    print()

    return True


def _print_available_input_devices() -> None:
    print("\n  Available input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"    [{i:2d}] {dev['name']}  (channels: {dev['max_input_channels']})")
    print()
