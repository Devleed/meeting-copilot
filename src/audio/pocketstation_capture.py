"""Capture one desktop application's audio with PocketStation."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Self

import numpy as np
from scipy.signal import resample_poly

AudioCallback = Callable[[np.ndarray, int, object | None, object | None], None]


class PocketStationCapture:
    """Present PocketStation application audio as the callback used by the app."""

    _SOURCE_RATE_HZ = 48_000
    _CLOSE_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        application: str,
        sample_rate: int,
        channels: int,
        chunk_size: int,
    ) -> None:
        application = application.strip()
        if not application:
            raise ValueError("application must not be empty")
        if sample_rate != 16_000:
            raise ValueError(
                "PocketStation capture currently requires a 16000 Hz output rate"
            )
        if channels != 1:
            raise ValueError("PocketStation capture currently produces mono audio")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        self._application: str | int = self._parse_application(application)
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._capture: Any | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._failure: Exception | None = None

    @staticmethod
    def _parse_application(value: str) -> str | int:
        process_id = value.removeprefix("pid:")
        if process_id == value:
            return value
        try:
            parsed = int(process_id)
        except ValueError as error:
            raise ValueError(
                "application process IDs must use pid:<positive integer>"
            ) from error
        if parsed <= 0:
            raise ValueError("application process ID must be greater than zero")
        return parsed

    @property
    def failure(self) -> Exception | None:
        """Return the reader failure, if capture ended unexpectedly."""
        return self._failure

    def open(self, callback: AudioCallback) -> PocketStationCapture:
        """Start application capture and deliver 16 kHz mono chunks."""
        if self._capture is not None:
            raise RuntimeError("PocketStation capture is already open")

        try:
            import pocketstation
        except ImportError as error:
            raise RuntimeError(
                "PocketStation capture is not installed; run "
                "`uv sync --extra pocketstation`"
            ) from error

        capture = pocketstation.capture(application=self._application)
        capture.start()
        self._capture = capture
        self._stop.clear()
        self._failure = None
        self._worker = threading.Thread(
            target=self._read_audio,
            args=(callback,),
            name="meeting-copilot-pocketstation",
        )
        try:
            self._worker.start()
        except Exception:
            capture.close()
            self._worker = None
            self._capture = None
            raise
        return self

    def _read_audio(self, callback: AudioCallback) -> None:
        input_samples_per_chunk = (
            self._chunk_size * self._SOURCE_RATE_HZ // self._sample_rate
        )
        pending = np.empty(0, dtype=np.float32)

        try:
            capture = self._capture
            if capture is None:
                return
            for frame in capture.audio:
                if self._stop.is_set():
                    break
                if frame.sample_rate_hz != self._SOURCE_RATE_HZ:
                    raise RuntimeError(
                        "PocketStation returned an unexpected sample rate: "
                        f"{frame.sample_rate_hz} Hz"
                    )
                samples = np.frombuffer(frame.samples_f32le, dtype="<f4")
                if frame.channel_count > 1:
                    samples = samples.reshape(-1, frame.channel_count).mean(axis=1)
                pending = np.concatenate((pending, samples))

                while len(pending) >= input_samples_per_chunk:
                    source = pending[:input_samples_per_chunk]
                    pending = pending[input_samples_per_chunk:]
                    output = resample_poly(
                        source,
                        self._sample_rate,
                        self._SOURCE_RATE_HZ,
                    ).astype(np.float32, copy=False)
                    callback(
                        output.reshape(-1, self._channels),
                        len(output),
                        None,
                        None,
                    )
        except Exception as error:  # noqa: BLE001 -- retain provider failures for the caller
            if not self._stop.is_set():
                self._failure = error

    def close(self) -> None:
        """Stop capture and wait for the reader thread to finish."""
        capture = self._capture
        if capture is None:
            return
        self._stop.set()
        close_error: Exception | None = None
        try:
            capture.close()
        except Exception as error:  # noqa: BLE001 -- join still runs after provider failure
            close_error = error
        worker = self._worker
        if worker is not None:
            worker.join(timeout=self._CLOSE_TIMEOUT_SECONDS)
            if worker.is_alive():
                raise RuntimeError(
                    "PocketStation capture did not stop within 5 seconds"
                )
        self._worker = None
        self._capture = None
        if close_error is not None:
            raise close_error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def test_pocketstation_capture(
    application: str,
    *,
    sample_rate: int,
    chunk_size: int,
    duration: float = 3.0,
) -> bool:
    """Capture briefly and report whether the selected application is audible."""
    received: list[np.ndarray] = []
    complete = threading.Event()
    target_samples = int(sample_rate * duration)

    def receive(samples: np.ndarray, _frames, _time, _status) -> None:
        received.append(samples.copy())
        if sum(len(chunk) for chunk in received) >= target_samples:
            complete.set()

    capture = PocketStationCapture(application, sample_rate, 1, chunk_size)
    try:
        with capture.open(receive):
            complete.wait(timeout=duration + 3.0)
    except Exception as error:  # noqa: BLE001 -- this command reports capture failures
        print(f"PocketStation capture failed: {error}")
        return False

    if capture.failure is not None:
        print(f"PocketStation capture failed: {capture.failure}")
        return False
    if not received:
        print("No audio arrived from the selected application.")
        return False

    audio = np.concatenate(received).reshape(-1)
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    print(f"Samples captured: {len(audio)}")
    print(f"RMS level: {rms:.6f}")
    print(f"Peak level: {peak:.6f}")
    if rms <= 1e-4:
        print("The selected application was silent. Start playback and try again.")
        return False
    print("Application audio is available.")
    return True
