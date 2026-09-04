from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from audio.pocketstation_capture import PocketStationCapture


class FakeCapture:
    def __init__(self, frames) -> None:
        self.audio = frames
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True


class PocketStationCaptureTests(unittest.TestCase):
    def test_application_name_and_process_id_are_preserved(self) -> None:
        named = PocketStationCapture("Zoom", 16_000, 1, 512)
        process = PocketStationCapture("pid:1234", 16_000, 1, 512)

        self.assertEqual(named._application, "Zoom")
        self.assertEqual(process._application, 1234)

    def test_invalid_process_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PocketStationCapture("pid:not-a-number", 16_000, 1, 512)
        with self.assertRaises(ValueError):
            PocketStationCapture("pid:0", 16_000, 1, 512)

    def test_reader_delivers_the_existing_vad_chunk_shape(self) -> None:
        source = np.linspace(-0.25, 0.25, 1_536, dtype=np.float32)
        frames = [
            SimpleNamespace(
                sample_rate_hz=48_000,
                channel_count=1,
                samples_f32le=source[:960].tobytes(),
            ),
            SimpleNamespace(
                sample_rate_hz=48_000,
                channel_count=1,
                samples_f32le=source[960:].tobytes(),
            ),
        ]
        fake_capture = FakeCapture(frames)
        received: list[np.ndarray] = []
        ready = threading.Event()

        def callback(samples, frame_count, _time, _status) -> None:
            self.assertEqual(frame_count, 512)
            received.append(samples)
            ready.set()

        fake_module = SimpleNamespace(capture=lambda **_kwargs: fake_capture)
        with patch.dict(sys.modules, {"pocketstation": fake_module}):
            capture = PocketStationCapture("Zoom", 16_000, 1, 512)
            with capture.open(callback):
                self.assertTrue(ready.wait(timeout=2.0))

        self.assertTrue(fake_capture.started)
        self.assertTrue(fake_capture.closed)
        self.assertIsNone(capture.failure)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].shape, (512, 1))


if __name__ == "__main__":
    unittest.main()
