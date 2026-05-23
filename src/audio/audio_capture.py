# =============================================================================
# COMPONENT: audio_capture.py
#
# Manages the sounddevice InputStream that captures audio from the BlackHole
# virtual audio device. Provides a clean context-manager interface so the
# stream is always closed when the caller's with-block exits, even on exceptions.
#
# Single Responsibility: open, hold, and close a sounddevice InputStream.
# Nothing about what to do with the audio — that is the pipeline's concern.
# The callback is injected so the caller decides how to process each audio block.
#
# SOLID notes:
#   S — sole job: lifecycle management of one sounddevice InputStream.
#   O — sample rate, channels, chunk size, and device index are all injected;
#       no hard-coded audio constants.
#   D — depends on sounddevice (external library) and a callback function
#       injected at start(); the capture class never knows what happens to audio.
# =============================================================================

from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly


class AudioCapture:
    """
    Opens and manages a sounddevice InputStream for capturing system audio.

    Responsibility: create and hold a sounddevice InputStream pointed at a
    specific input device; close it cleanly on exit. The class is a context
    manager (supports `with AudioCapture(...) as cap:`) to guarantee cleanup.

    Usage:
        capture = AudioCapture(
            device_index=blackhole_index,
            sample_rate=16000,
            channels=1,
            chunk_size=512,
        )
        with capture.open(callback=my_audio_callback):
            # audio is flowing; callback fires for each block
            ...
        # stream is now closed
    """

    def __init__(
        self,
        device_index: int | str,  # sounddevice integer index or "default"
        sample_rate: int,         # downstream rate in Hz — what VAD and Whisper expect (16 000)
        channels: int,            # number of audio channels (1 = mono)
        chunk_size: int,          # block size in samples at sample_rate (e.g. 512)
        capture_sample_rate: int | None = None,  # native device rate; None means same as sample_rate
    ) -> None:
        self._device_index = device_index
        self._sample_rate: int = sample_rate
        self._channels: int = channels
        self._chunk_size: int = chunk_size
        self._capture_sample_rate: int = capture_sample_rate if capture_sample_rate is not None else sample_rate
        self._stream = None

    def open(self, callback):
        """
        Open the audio input stream and register the callback.

        The callback is called by sounddevice on a dedicated high-priority audio
        thread every time a new block of `chunk_size` samples is available.
        The callback signature must be:

            def callback(indata, frames, time, status):
                ...

        where:
          indata  — numpy float32 array of shape (frames, channels)
          frames  — number of samples in this block (equals chunk_size)
          time    — sounddevice CFFI time info struct (rarely needed)
          status  — sounddevice StatusFlags indicating overflows/errors

        Returns self so the caller can use this method in a `with` statement:
            with capture.open(callback=fn):
                ...

        Parameters
        ----------
        callback : callable
            Function called for each audio block. Must be fast and non-blocking.

        Returns
        -------
        AudioCapture
            self — enables use as a context manager.
        """
        needs_resample = self._capture_sample_rate != self._sample_rate
        if needs_resample:
            _g = gcd(self._sample_rate, self._capture_sample_rate)
            _up = self._sample_rate // _g
            _down = self._capture_sample_rate // _g
            # Capture blocksize scaled so resampled output is exactly chunk_size samples.
            capture_blocksize = self._chunk_size * self._capture_sample_rate // self._sample_rate

            def _resampling_callback(indata, frames, time, status):
                resampled = resample_poly(indata, _up, _down, axis=0).astype(np.float32)
                callback(resampled, resampled.shape[0], time, status)

            stream_callback = _resampling_callback
        else:
            capture_blocksize = self._chunk_size
            stream_callback = callback

        self._stream = sd.InputStream(
            device=self._device_index,
            channels=self._channels,
            samplerate=self._capture_sample_rate,
            dtype="float32",
            blocksize=capture_blocksize,
            callback=stream_callback,
        )

        # Enter the InputStream's own context manager to start the stream.
        # This opens the PortAudio device and begins delivering audio blocks.
        self._stream.__enter__()
        return self  # return self so `with capture.open(cb) as cap:` works

    def close(self) -> None:
        """
        Stop the audio stream and release the PortAudio device.

        Safe to call multiple times — no-op if the stream is already closed.
        """
        if self._stream is not None:
            # __exit__(None, None, None) — tell the context manager to close cleanly
            # (None, None, None means "no exception was raised").
            # This stops the audio thread and releases the PortAudio device handle.
            self._stream.__exit__(None, None, None)
            self._stream = None  # clear the reference so close() is idempotent

    # ── Context manager protocol ────────────────────────────────────────────────
    # Implementing __enter__ and __exit__ lets callers write:
    #   with capture.open(callback=fn):
    #       ...
    # Python calls __enter__ on entering the with-block and __exit__ on leaving it.

    def __enter__(self):
        # __enter__ is called when the with-block is entered.
        # We already started the stream in open(), so just return self.
        return self  # return self so `as cap` in `with ... as cap:` refers to this object

    def __exit__(self, exc_type, exc_val, exc_tb):
        # __exit__ is called when the with-block exits, with or without an exception.
        # exc_type, exc_val, exc_tb — exception info if an exception occurred, else all None.
        # We always close the stream regardless of whether an exception was raised.
        self.close()
        # Return None (falsy) — do not suppress any exception that propagated out.
        # Returning True here would silence exceptions, which we don't want.
