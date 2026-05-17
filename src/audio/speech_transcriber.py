# =============================================================================
# COMPONENT: speech_transcriber.py
#
# Converts raw float32 audio buffers into text using the faster-whisper library
# (a CTranslate2-optimised port of OpenAI Whisper).
#
# Single Responsibility: load the Whisper model once and expose a single
# transcribe(audio) method that returns a list of text strings. No audio
# capture, no VAD, no threading — just speech-to-text inference.
#
# Why faster-whisper?
#   faster-whisper is significantly faster than the original openai-whisper
#   on CPU and Apple Silicon (MPS) because it uses CTranslate2 for optimised
#   inference with int8 quantisation. Accuracy is equivalent for meeting audio.
#
# SOLID notes:
#   S — sole job: transcribe an audio buffer to a list of text strings.
#   O — model size, device, and compute type are injected; changing them
#       requires no code modification.
#   D — depends on the WhisperModel class from faster-whisper, but the model
#       object is created inside __init__ from injected parameters, allowing
#       tests to substitute a mock if needed.
# =============================================================================

import numpy as np  # numpy — audio buffers are numpy float32 arrays

# WhisperModel — the main class in the faster-whisper library.
# It wraps a CTranslate2-converted Whisper model and exposes a .transcribe() method.
from faster_whisper import WhisperModel


class SpeechTranscriber:
    """
    Wraps the faster-whisper WhisperModel to transcribe audio buffers to text.

    Responsibility: load the Whisper model at construction time and expose a
    transcribe() method that accepts a raw float32 numpy array and returns a
    list of non-empty text strings (one per Whisper segment).

    The model is loaded synchronously in __init__ so the first call to
    transcribe() has no cold-start latency. Loading takes a few seconds at
    startup but runs only once.

    Usage:
        transcriber = SpeechTranscriber(model_size="base")
        texts = transcriber.transcribe(audio_array)
        # texts == ["Can you walk me through", "the architecture?"]
    """

    def __init__(
        self,
        model_size: str,            # model_size   — "tiny", "base", "small", "medium", "large-v2"
        beam_size: int = 5,         # beam_size    — beam search width; higher = slightly more accurate
        device: str = "auto",       # device       — "auto" lets CTranslate2 pick CUDA > MPS > CPU
        compute_type: str = "int8", # compute_type — quantisation; "int8" is fast with minimal loss
        model=None,                 # model        — optional pre-loaded model for testing
    ) -> None:
        self._beam_size: int = beam_size

        if model is not None:
            # Accept an externally provided model so tests can inject a mock.
            self._model = model
        else:
            print("Loading Whisper model...")
            # WhisperModel(model_size, device=..., compute_type=...) — download (or load
            # from the local CTranslate2 cache) and initialise the Whisper model.
            #
            # model_size   — controls accuracy vs. speed trade-off:
            #                "base" is recommended for meeting audio (fast, good accuracy).
            #
            # device="auto"  — CTranslate2 automatically picks the fastest available device:
            #                  CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.
            #
            # compute_type="int8" — 8-bit integer quantisation reduces memory usage and
            #                       speeds up inference with negligible accuracy loss for
            #                       speech-to-text tasks. ("float16" requires MPS/CUDA.)
            self._model = WhisperModel(
                model_size,               # which Whisper model variant to load
                device=device,            # hardware device to run inference on
                compute_type=compute_type,  # numeric precision / quantisation type
            )
            print("Whisper model loaded.")

    def transcribe(self, audio: np.ndarray) -> list:
        """
        Transcribe a raw audio buffer and return a list of text strings.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 numpy array of audio samples at 16 kHz.
            Typically the accumulated speech buffer from one utterance.

        Returns
        -------
        list[str]
            List of non-empty text strings, one per Whisper segment.
            An empty list is returned if Whisper produces no text (silence,
            noise, or audio shorter than a complete word).
        """
        # self._model.transcribe(audio, beam_size=...) — run the Whisper pipeline.
        # Returns a tuple: (segments_generator, TranscriptionInfo).
        # `segments_generator` is a generator (lazy); iterating it triggers inference.
        # `TranscriptionInfo` contains language, duration, etc. — we discard it with `_`.
        # In TS: the equivalent would be an async iterator of segment objects.
        segments, _ = self._model.transcribe(
            audio,                       # the raw float32 audio buffer to transcribe
            beam_size=self._beam_size,   # beam search width; 5 is a good balance
        )

        # Iterate over the lazy segments generator and collect non-empty text strings.
        #
        # segment.text  — the transcribed text string for one Whisper segment.
        # .strip()      — remove leading/trailing whitespace from the segment text.
        # if text       — skip empty strings (whitespace-only segments after stripping).
        #
        # List comprehension with walrus operator (:=):
        #   text := segment.text.strip()  — assign AND use the value in one expression.
        #   `if text`                     — filter out empty strings.
        # This avoids computing .strip() twice (once to assign, once to check).
        # In TS: segments.flatMap(s => s.text.trim() ? [s.text.trim()] : [])
        return [
            text                                 # include this segment's text
            for segment in segments              # iterate over Whisper segments
            if (text := segment.text.strip())   # assign stripped text; skip if empty
        ]
