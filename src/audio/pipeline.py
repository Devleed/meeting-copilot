# =============================================================================
# COMPONENT: pipeline.py
#
# The AudioPipeline orchestrates the end-to-end flow from captured audio to
# AI suggestions. It is the central coordinator of the runtime loop but
# delegates every specific task to collaborating classes.
#
# High-level data flow:
#
#   Configured audio source → capture callback → audio_queue
#                                         ↓  (VAD thread)
#                                   VoiceActivityDetector
#                                         ↓  (speech detected → silence)
#                                   SpeechTranscriber
#                                         ↓  (text segments)
#                                   FlushBuffer  ←── debounce timer
#                                         ↓  (timer fires)
#                                   SuggestionGenerator (on daemon thread)
#                                         ↓
#                                   terminal output
#
# Modes:
#   "auto"   — VAD detects end-of-utterance automatically (default).
#   "manual" — audio accumulates in a buffer; pressing Enter triggers
#              transcription + suggestion.
#
# Single Responsibility: coordinate the audio capture thread, the VAD loop,
# mode switching, and manual-mode keyboard commands. It owns none of the
# heavy logic — it calls the appropriate collaborator for each step.
#
# SOLID notes:
#   S — sole job: wire the pipeline components together and run the event loop.
#   O — adding a new mode or processing step means adding a collaborator and
#       a branch here; existing collaborators are unaffected.
#   D — all heavy-lifting collaborators (VAD, transcriber, buffer, generator)
#       are injected, making the pipeline testable with mock components.
# =============================================================================

import sys        # sys — read from stdin (command loop) and check for EOF
import queue      # queue.Queue — thread-safe FIFO used to pass audio from callback to VAD thread
import threading  # threading.Thread — runs the VAD loop on a background thread

import numpy as np  # numpy — audio buffers are numpy float32 arrays

# Local collaborating classes — all injected or constructed here.
from audio.audio_device import AudioDeviceLocator
from audio.audio_capture import AudioCapture
from audio.vad import VoiceActivityDetector
from audio.speech_transcriber import SpeechTranscriber
from audio.flush_buffer import FlushBuffer
from audio.pocketstation_capture import PocketStationCapture


class AudioPipeline:
    """
    Orchestrates audio capture, VAD, transcription, and AI suggestion generation.

    Responsibility: manage the runtime loop that ties together audio capture,
    voice activity detection, speech transcription, debounced flushing, and
    suggestion generation. Handle both auto and manual operating modes and
    support toggling between them at runtime.

    All heavy processing is delegated to injected collaborators:
      suggestion_generator — called (on a daemon thread) when text is flushed.
      config               — supplies all tunable parameters.

    Internal components created by this class:
      AudioDeviceLocator     — finds the audio input device index.
      AudioCapture           — manages the sounddevice InputStream.
      PocketStationCapture   — captures one selected desktop application.
      VoiceActivityDetector  — Silero VAD; classifies audio chunks as speech/silence.
      SpeechTranscriber      — faster-whisper; converts audio buffers to text.
      FlushBuffer            — debounces text segments and fires the AI callback.
      queue.Queue            — decouples the real-time audio callback from the VAD thread.

    Usage:
        pipeline = AudioPipeline(config, suggestion_generator)
        pipeline.run()   # blocks until Ctrl+C
    """

    def __init__(self, config, suggestion_generator, chat_history_service=None) -> None:
        # config — AppConfig instance; provides all audio and VAD parameters.
        self._config = config

        # suggestion_generator — SuggestionGenerator instance; called with the
        # final transcribed text when the flush timer fires or Enter is pressed.
        self._suggestion_gen = suggestion_generator

        # chat_history_service — ChatHistoryService | None; used to display and
        # export the session history when the user ends the session with "q".
        self._chat_history = chat_history_service

        # ── Thread-safe audio queue ────────────────────────────────────────────
        # The audio callback (called on a high-priority audio thread) pushes
        # numpy arrays onto this queue. The VAD loop (on a background thread)
        # pops them. Using a Queue decouples the real-time thread from slow work.
        # In TS: equivalent to a concurrent async queue or a SharedArrayBuffer approach.
        self._audio_queue: queue.Queue = queue.Queue()

        # ── Audio device locator ───────────────────────────────────────────────
        # Used in run() to find the BlackHole device index before opening the stream.
        self._device_locator = AudioDeviceLocator(
            fallback_index=config.blackhole_device_index
        )

        # ── VAD component ──────────────────────────────────────────────────────
        # Loads the Silero model at construction time (a few seconds at startup).
        self._vad = VoiceActivityDetector(
            sample_rate=config.sample_rate,
            threshold=config.speech_probability_threshold,
        )

        # ── Transcription component ────────────────────────────────────────────
        # Loads the Whisper model at construction time.
        self._transcriber = SpeechTranscriber(model_size=config.model_size)

        # ── Flush buffer ───────────────────────────────────────────────────────
        # When the flush timer fires, call _on_flush which spawns a daemon thread
        # to run suggestion_generator.generate() without blocking the VAD loop.
        self._flush_buffer = FlushBuffer(
            wait_seconds=config.flush_wait_seconds,
            on_flush=self._on_flush,
        )

        # ── Operating mode ─────────────────────────────────────────────────────
        # Current mode: "auto" or "manual". Mutable — toggled by the command loop.
        self._mode: str = config.mode

        # ── Manual mode audio buffer ───────────────────────────────────────────
        # In manual mode, raw audio samples accumulate here instead of going
        # through VAD. The user presses Enter to transcribe everything in the buffer.
        self._manual_buffer: list = []

        # threading.Lock() — mutual exclusion primitive.
        # Protects _manual_buffer from race conditions between the VAD thread
        # (which writes audio samples) and the main thread (which reads and clears it).
        self._manual_lock: threading.Lock = threading.Lock()

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the pipeline and block until Ctrl+C or EOF.

        Steps:
          1. Choose the configured audio source.
          2. Start the VAD background thread.
          3. Open the audio capture stream.
          4. Run the keyboard command loop on the main thread.
        """
        # Step 1: choose the configured audio source.
        if self._config.capture_backend == "pocketstation":
            if not self._config.capture_application:
                raise ValueError(
                    "CAPTURE_APPLICATION is required when CAPTURE_BACKEND=pocketstation"
                )
            capture = PocketStationCapture(
                application=self._config.capture_application,
                sample_rate=self._config.sample_rate,
                channels=1,
                chunk_size=self._config.vad_chunk_size,
            )
            print(
                "Using PocketStation application audio: "
                f"{self._config.capture_application}\n"
            )
        else:
            device_index = self._device_locator.find()
            print(f"Using audio input device index: {device_index}\n")
            capture = AudioCapture(
                device_index=device_index,
                sample_rate=self._config.sample_rate,
                channels=1,
                chunk_size=self._config.vad_chunk_size,
                capture_sample_rate=self._config.capture_sample_rate,
            )

        # Step 2: start the VAD loop on a daemon background thread.
        # daemon=True — this thread is killed automatically when the main thread exits.
        # Without daemon=True the process would hang on Ctrl+C waiting for the thread.
        vad_thread = threading.Thread(
            target=self._vad_loop,  # the function to run on the background thread
            daemon=True,            # die automatically when the main thread exits
        )
        vad_thread.start()  # .start() — launch the thread; returns immediately

        # Print the initial mode banner before entering the stream context.
        print("[ MODE: AUTO — VAD will detect end of speech ]")
        print("[ Type m + Enter to toggle mode | Press Enter in manual mode to transcribe ]")
        print("[ Type q + Enter to end session and view chat history ]\n")

        # Open the selected source and keep it alive for the duration of the
        # command loop. Both capture implementations provide the same callback.
        with capture.open(callback=self._audio_callback):
            print("Listening... Press Ctrl+C to stop\n")
            # Step 4: run the keyboard command loop on the main thread.
            self._command_loop()

    # ── Audio callback ────────────────────────────────────────────────────────

    def _audio_callback(
        self,
        indata,    # indata  — numpy float32 array of shape (frames, channels)
        frames,    # frames  — number of samples delivered (equals VAD_CHUNK_SIZE)
        time,      # time    — sounddevice CFFI time info (arrival/current/output times)
        status,    # status  — sounddevice StatusFlags; non-zero signals overflow/underflow
    ) -> None:
        """
        Called by the selected capture implementation for each audio block.

        Must return quickly — no heavy computation here. Just copy the buffer
        (sounddevice reuses it) and push it onto the queue.
        """
        # Copy before enqueueing because sounddevice reuses its input buffer.
        # Keeping the same rule for every capture implementation also gives the
        # queue ownership of every array it receives.
        # In TS: Float32Array.from(indata) or structuredClone(indata)
        self._audio_queue.put(indata.copy())

    # ── VAD background thread ──────────────────────────────────────────────────

    def _vad_loop(self) -> None:
        """
        Background thread: pull audio from the queue, run VAD, transcribe utterances.

        Runs forever (while True) until the process is killed. Each iteration:
          - Dequeues one audio block (blocks until data is available).
          - In manual mode: accumulates raw audio without VAD processing.
          - In auto mode: runs VAD per 512-sample chunk; commits utterances on silence.
        """
        speech_buffer: list = []    # accumulates float32 samples during a speech segment
        silence_chunks: int = 0     # consecutive silent chunks counted after speech ends
        is_speaking: bool = False   # True while a speech segment is in progress

        # leftover — samples that did not fill a complete VAD_CHUNK_SIZE slice.
        # Each audio block from sounddevice may not align exactly with VAD_CHUNK_SIZE;
        # leftover carries the remainder into the next iteration.
        # dtype=np.float32 — must match the dtype produced by sounddevice.
        leftover: np.ndarray = np.array([], dtype=np.float32)

        # Pre-compute the silence threshold in chunks.
        # chunks_per_second = 16000 / 512 = 31.25 chunks per second at 16 kHz.
        chunks_per_second: float = self._config.sample_rate / self._config.vad_chunk_size
        # int(...) — truncate float to int; 31.25 * 4.0 → int(125.0) = 125 chunks.
        silence_threshold_chunks: int = int(chunks_per_second * self._config.silence_threshold)

        while True:   # while True — run forever; exit only when the process is killed
            # audio_queue.get() — block this thread until an item is available.
            # This is efficient: the thread sleeps and consumes no CPU while waiting.
            # In TS: await queue.dequeue() with an async queue library.
            data = self._audio_queue.get()

            # Concatenate the leftover from the previous iteration with the new block.
            # np.concatenate([a, b]) — join two 1-D arrays end-to-end.
            # data.flatten() — convert shape (frames, channels) → 1-D array.
            samples: np.ndarray = np.concatenate([leftover, data.flatten()])

            # ── Manual mode: bypass VAD ──────────────────────────────────────
            if self._mode == "manual":
                # In manual mode, skip VAD entirely. Accumulate all audio in
                # _manual_buffer; the user will press Enter to trigger transcription.
                # The lock prevents a race between this write and the main thread's read.
                with self._manual_lock:  # with lock: — acquire the lock; release on exit
                    # list.extend(iterable) — append all elements of iterable to the list.
                    # .tolist() — convert numpy array to a plain Python list of floats.
                    self._manual_buffer.extend(samples.tolist())

                leftover = np.array([], dtype=np.float32)  # no leftover in manual mode
                continue  # continue — skip to the next loop iteration

            # ── Auto mode: VAD pipeline ──────────────────────────────────────
            idx: int = 0  # idx — current position within `samples`

            # Slide a window of VAD_CHUNK_SIZE through `samples`, processing
            # each fixed-size chunk with the VAD model.
            while idx + self._config.vad_chunk_size <= len(samples):
                # Slice one VAD_CHUNK_SIZE chunk from the samples array.
                # samples[idx : idx + size] — numpy slice; returns a view (no copy).
                chunk: np.ndarray = samples[idx: idx + self._config.vad_chunk_size]
                idx += self._config.vad_chunk_size  # advance the read pointer

                # Ask the VoiceActivityDetector whether this chunk contains speech.
                # is_speech() runs a Silero forward pass and compares to the threshold.
                if self._vad.is_speech(chunk):
                    # Speech detected in this chunk.
                    if not is_speaking:
                        print("[ speaking... ]")  # print — display status to the terminal
                        is_speaking = True          # mark the start of a speech segment
                    silence_chunks = 0              # reset the silence counter
                    # list.extend(iterable) — append chunk samples to the speech buffer.
                    # chunk.tolist() — convert numpy array to plain Python list of floats.
                    speech_buffer.extend(chunk.tolist())

                else:
                    # Silence detected. If we were speaking, count this silent chunk.
                    if is_speaking:
                        silence_chunks += 1               # increment silence counter
                        speech_buffer.extend(chunk.tolist())  # include trailing silence in buffer

                        # Check if we've had enough consecutive silent chunks to declare
                        # the utterance complete.
                        if silence_chunks >= silence_threshold_chunks:
                            self._commit_utterance(speech_buffer)  # transcribe and buffer
                            speech_buffer = []        # reset speech buffer for next utterance
                            silence_chunks = 0        # reset silence counter
                            is_speaking = False       # mark end of speaking segment

            # Samples that didn't complete a full VAD chunk — carry them over to
            # the next audio block so no audio is silently dropped.
            leftover = samples[idx:]  # samples[idx:] — slice from idx to the end of the array

    def _commit_utterance(self, speech_buffer: list) -> None:
        """
        Transcribe a completed speech buffer and feed the text to the flush buffer.

        Called when the VAD detects enough consecutive silence to consider the
        current utterance finished.

        Parameters
        ----------
        speech_buffer : list[float]
            Accumulated float32 audio samples for one utterance.
        """
        # Convert the Python list of floats back to a numpy float32 array.
        # WhisperModel.transcribe() requires a numpy array, not a plain list.
        utterance: np.ndarray = np.array(speech_buffer, dtype=np.float32)

        # Discard utterances that are too short (clicks, breaths, single phonemes).
        # len(utterance) / sample_rate — duration in seconds.
        duration_seconds: float = len(utterance) / self._config.sample_rate
        if duration_seconds < self._config.min_utterance_seconds:
            return  # return early — utterance too short to transcribe

        print("[ processing... ]")
        # Ask the SpeechTranscriber for a list of text segments.
        texts: list = self._transcriber.transcribe(utterance)

        print(f"[ transcribed ] {texts}\n")  # print the raw transcribed segments for debugging

        for text in texts:
            # Feed each non-empty segment to the flush buffer.
            # The buffer resets its debounce timer on each add() call.
            self._flush_buffer.add(text)

    # ── Flush callback (called by FlushBuffer on its timer thread) ────────────

    def _on_flush(self, full_text: str) -> None:
        """
        Called by FlushBuffer when the debounce timer fires.

        Spawns a daemon thread to call suggestion_generator.generate() so
        the long-running OpenAI API call does not block the VAD loop.

        Parameters
        ----------
        full_text : str
            The joined text of all buffered segments for one speaker turn.
        """
        # threading.Thread(target=fn, args=(...), daemon=True).start() —
        # spawn a new background thread that calls fn with the given args.
        # daemon=True — the thread exits automatically when the main thread dies.
        # This pattern is the Python equivalent of Promise.resolve().then(...) in JS:
        # fire-and-forget async execution.
        threading.Thread(
            target=self._suggestion_gen.generate,  # function to run on the new thread
            args=(full_text, self._mode == "manual"),     # positional argument: the transcribed text
            daemon=True,                           # die with the main thread
        ).start()

    # ── Manual mode transcription ─────────────────────────────────────────────

    def _transcribe_manual(self) -> None:
        """
        Triggered when the user presses Enter in manual mode.

        Snapshots and clears the manual audio buffer, then transcribes it on a
        daemon thread so the command loop is not blocked during Whisper inference.
        """
        # Acquire the lock to safely read and clear the manual buffer.
        with self._manual_lock:
            # list(...) — create a copy of the buffer so we can release the lock
            # before doing the slow transcription work.
            audio_snapshot: list = list(self._manual_buffer)
            # .clear() — empty the buffer so new audio starts accumulating fresh.
            self._manual_buffer.clear()

        if not audio_snapshot:
            print("[ nothing recorded yet ]")
            return  # nothing to transcribe; return early

        def _do_transcribe() -> None:
            # This inner function runs on a daemon thread so Whisper's inference
            # doesn't block the main thread's command loop.
            print("[ processing... ]")

            # Convert the flat list of floats back to a numpy float32 array.
            utterance: np.ndarray = np.array(audio_snapshot, dtype=np.float32)

            # Check minimum duration before passing to Whisper.
            if len(utterance) / self._config.sample_rate < self._config.min_utterance_seconds:
                print("[ recording too short to transcribe ]")
                return

            # Transcribe and send directly, bypassing the flush buffer (manual
            # mode fires one shot per Enter press, no debouncing needed).
            texts: list = self._transcriber.transcribe(utterance)
            if texts:
                # manual=True — bypasses the greeting filter in SuggestionGenerator.
                self._suggestion_gen.generate(" ".join(texts), manual=True)

        # threading.Thread(target=_do_transcribe, daemon=True).start() —
        # launch the transcription on a background thread.
        threading.Thread(target=_do_transcribe, daemon=True).start()

    # ── Session end ───────────────────────────────────────────────────────────

    def _end_session(self) -> None:
        """
        Display the full session chat history and offer to export it as a TXT file.

        Called when the user types "q" in the command loop. After this returns,
        the command loop breaks and the audio stream is closed.
        """
        if self._chat_history is None:
            print("[ no chat history service configured ]")
            return

        self._chat_history.display()

        print("\nExport chat history as TXT? Press e + Enter to export, or Enter to skip: ", end="", flush=True)
        answer: str = sys.stdin.readline().strip()
        if answer == "e":
            path: str = self._chat_history.export_txt()
            print(f"[ exported → {path} ]")

        print("[ session ended ]\n")

    # ── Main-thread command loop ───────────────────────────────────────────────

    def _command_loop(self) -> None:
        """
        Read keyboard commands from stdin and dispatch them.

        Runs on the main thread for the lifetime of the session.
        Commands:
          "m" + Enter — toggle between auto and manual mode.
          Enter (in manual mode) — transcribe the accumulated audio buffer.
          Ctrl+C — raise KeyboardInterrupt; the with-block in run() closes the stream.
          EOF — break the loop and exit cleanly.
        """
        while True:
            # sys.stdin.readline() — read one line of text from standard input.
            # Blocks until the user presses Enter. Returns "" on EOF (Ctrl+D on Unix).
            line: str = sys.stdin.readline()

            if not line:
                # EOF — stdin was closed (e.g. the terminal was closed, or piped input ended).
                break  # break — exit the while loop; run() returns and the stream closes

            # .strip() — remove the trailing newline (\n) and any surrounding whitespace.
            cmd: str = line.strip()

            if cmd == "q":
                # End session: display history and offer TXT export, then exit.
                self._end_session()
                break

            elif cmd == "m":
                # Toggle between auto and manual mode.
                if self._mode == "auto":
                    self._mode = "manual"
                    # Cancel any pending flush timer — we don't want a stale AI call
                    # firing after switching to manual mode.
                    self._flush_buffer.cancel()
                    print("[ MODE: MANUAL — press Enter to transcribe ]")
                else:
                    self._mode = "auto"
                    # Discard any audio that accumulated while the user was in manual mode.
                    with self._manual_lock:
                        self._manual_buffer.clear()
                    print("[ MODE: AUTO — VAD will detect end of speech ]")

            elif self._mode == "manual":
                # Any key press (other than "m" or "q") in manual mode triggers transcription.
                self._transcribe_manual()
