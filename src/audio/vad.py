# =============================================================================
# COMPONENT: vad.py
#
# Voice Activity Detection (VAD) using the Silero VAD neural network model.
# Determines whether a given audio chunk contains speech.
#
# Single Responsibility: load the Silero model once and expose a single
# is_speech(chunk) method that returns a boolean. No audio I/O, no threading,
# no transcription — just VAD inference.
#
# Why Silero?
#   Silero VAD is a lightweight PyTorch model trained specifically on human
#   speech detection. It processes fixed-size audio chunks (512 samples at
#   16 kHz = 32 ms) and returns a probability in [0, 1]. It is faster and
#   more accurate than energy-threshold approaches for real-world audio.
#
# SOLID notes:
#   S — sole job: score one audio chunk as speech or not-speech.
#   O — threshold and sample rate are injected; tuning them needs no code change.
#   D — depends on torch (external library) and a pre-loaded model optionally
#       injected at construction to support unit testing with mock models.
# =============================================================================

import torch  # torch — PyTorch; required to run the Silero VAD neural network
import numpy as np  # numpy — numeric array library; audio chunks arrive as numpy arrays


class VoiceActivityDetector:
    """
    Wraps the Silero VAD model to classify audio chunks as speech or silence.

    Responsibility: load the Silero model from the PyTorch Hub on first use,
    hold it in inference mode, and expose a single is_speech() method that
    accepts a float32 numpy array and returns True if the chunk contains speech.

    The model is loaded in __init__ (synchronous, at construction time) so the
    first call to is_speech() has no cold-start latency.

    Usage:
        vad = VoiceActivityDetector(sample_rate=16000, threshold=0.2)
        if vad.is_speech(audio_chunk):
            speech_buffer.extend(audio_chunk.tolist())
    """

    def __init__(
        self,
        sample_rate: int,   # sample_rate — audio sample rate in Hz; must match the stream
        threshold: float,   # threshold   — minimum speech probability to classify as speech
        model=None,         # model       — optional pre-loaded model; None = load from Hub
    ) -> None:
        self._sample_rate: int = sample_rate
        self._threshold: float = threshold

        if model is not None:
            # Accept an externally provided model (useful for unit tests that supply
            # a mock so the real Silero model is never downloaded in CI).
            self._model = model
        else:
            # Load the Silero VAD model from the PyTorch Hub the first time.
            # torch.hub.load(...) — downloads (or loads from local cache) a model
            # hosted on a public GitHub repository.
            #
            # repo_or_dir="snakers4/silero-vad"
            #   GitHub shorthand: "username/repository". PyTorch Hub fetches the
            #   hubconf.py from this repo and calls the specified entry point.
            #
            # model="silero_vad"
            #   The entry point name defined in the repo's hubconf.py.
            #
            # force_reload=False
            #   Use the locally cached version if available (avoids re-downloading).
            #
            # trust_repo=True
            #   Required since PyTorch 1.12 to explicitly allow running code from a
            #   third-party repository. Analogous to npm's package lock mechanism.
            #
            # Returns a tuple (model, utilities); we only need the model object.
            print("Loading VAD model...")
            vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",  # GitHub repo shorthand
                model="silero_vad",                 # entry point in hubconf.py
                force_reload=False,                 # use cached version if present
                trust_repo=True,                    # allow third-party code execution
            )

            # .eval() — switch PyTorch to inference mode.
            # Disables training-specific layers (dropout, batch normalisation updates).
            # Always call .eval() on a model being used for prediction, never training.
            # In ONNX / TensorFlow: equivalent to setting the model to inference session.
            vad_model.eval()
            self._model = vad_model
            print("VAD model loaded.\n")

    def is_speech(self, chunk: np.ndarray) -> bool:
        """
        Return True if the audio chunk is classified as containing speech.

        Parameters
        ----------
        chunk : np.ndarray
            1-D float32 array of exactly 512 samples (32 ms at 16 kHz).
            Must match the VAD_CHUNK_SIZE used by the pipeline.

        Returns
        -------
        bool
            True if Silero's speech probability ≥ threshold; False otherwise.
        """
        # torch.tensor(chunk, dtype=torch.float32) — convert the numpy float32 array
        # to a PyTorch 1-D tensor of the same dtype. PyTorch cannot consume numpy arrays
        # directly for model inference; conversion is required.
        # In TS/ONNX: Float32Array → inference session input tensor.
        tensor = torch.tensor(chunk, dtype=torch.float32)

        # torch.no_grad() — context manager that disables gradient tracking.
        # During inference we do not need gradients (those are only for backpropagation
        # during training). Disabling them saves memory and speeds up the forward pass.
        with torch.no_grad():
            # self._model(tensor, self._sample_rate) — run one VAD forward pass.
            # The model takes the audio tensor and the sample rate as arguments.
            # Returns a scalar tensor containing the speech probability in [0, 1].
            #
            # .item() — extract the Python float from the scalar tensor.
            # Without .item() we'd have a 0-dim tensor, not a plain float.
            speech_prob: float = self._model(tensor, self._sample_rate).item()

        # Compare the probability to the configured threshold.
        # >= threshold — True if speech detected; False if silence.
        # In TS: speechProb >= this._threshold
        return speech_prob >= self._threshold
