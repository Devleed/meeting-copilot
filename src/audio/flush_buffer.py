# =============================================================================
# COMPONENT: flush_buffer.py
#
# Debounce buffer that collects transcribed text segments and fires a callback
# after a configurable quiet period.
#
# Problem it solves:
#   Whisper often emits multiple segments for one speaker turn — e.g. "Can you"
#   and then "walk me through that?" two seconds apart. Without debouncing,
#   the AI would be called twice with partial sentences, producing low-quality
#   suggestions. FlushBuffer collects all segments and waits until no new
#   segments arrive for FLUSH_WAIT_SECONDS before firing — merging the fragments
#   into one coherent utterance.
#
# Single Responsibility: accumulate text strings and fire an on_flush callback
# after a quiet period (debounce). No I/O, no AI calls, no audio — just a
# timer-gated string accumulator.
#
# SOLID notes:
#   S — sole job: debounce a stream of strings and fire one callback per burst.
#   O — wait duration and callback are injected; no hardcoded constants.
#   D — depends only on threading.Timer from the standard library; the callback
#       is injected so this class never knows what happens with the flushed text.
# =============================================================================

import threading  # threading — standard library; threading.Timer fires a callback after a delay


class FlushBuffer:
    """
    Debounce buffer: accumulates text segments and fires a callback after silence.

    Responsibility: each time add(text) is called, append the text to an
    internal list and (re)start a countdown timer. If no new text arrives
    before the timer expires, join all collected text into one string and
    call on_flush with it. If new text arrives before the timer expires,
    cancel the old timer and start a fresh one — effectively resetting the
    countdown each time.

    This ensures that rapid consecutive segments from the same speaker turn
    are joined into one coherent string before the AI is called.

    Thread safety note:
      add() and _fire() may be called from different threads (the VAD
      transcription thread calls add(); threading.Timer calls _fire() on its
      own thread). Python's GIL protects simple list.append() and list.clear()
      operations in CPython, so no explicit Lock is needed for _pending.
      The timer reference (_timer) is also managed under the GIL.

    Usage:
        def handle_flush(text):
            print(f"Ready to send: {text}")

        buf = FlushBuffer(wait_seconds=4.0, on_flush=handle_flush)
        buf.add("Can you")       # starts a 4s timer
        buf.add("walk me through that?")  # resets the timer
        # 4 seconds of silence → handle_flush("Can you walk me through that?")
    """

    def __init__(
        self,
        wait_seconds: float,   # wait_seconds — seconds of quiet before the callback fires
        on_flush,              # on_flush     — callable(text: str); called with the joined text
    ) -> None:
        self._wait: float = wait_seconds   # store the debounce interval
        self._on_flush = on_flush          # store the callback to call when the timer fires

        # _pending — list of text strings accumulated since the last flush.
        # Reset to [] after each flush.
        self._pending: list = []

        # _timer — reference to the active threading.Timer, or None if no timer is running.
        # Keeping this reference allows us to cancel (reset) the timer when new text arrives.
        self._timer = None

    def add(self, text: str) -> None:
        """
        Add a text segment to the buffer and reset the debounce timer.

        Each call cancels the current timer (if any) and starts a fresh one.
        The callback fires only if no further add() calls arrive within
        wait_seconds.

        Parameters
        ----------
        text : str
            A transcribed text segment to buffer.
        """
        # list.append(text) — add the new segment to the accumulator.
        # Thread-safe in CPython due to the GIL protecting list mutations.
        self._pending.append(text)

        # Cancel any currently running timer so it doesn't fire prematurely.
        # If _timer is None (no timer running), this branch is a no-op.
        if self._timer is not None:
            # threading.Timer.cancel() — mark the timer as cancelled.
            # If the timer has already fired, cancel() is a safe no-op.
            self._timer.cancel()

        # Start a new countdown timer.
        # threading.Timer(interval, function) — creates a timer that calls function
        # after `interval` seconds on a new daemon thread.
        # interval=self._wait  — the configured quiet period in seconds.
        # function=self._fire  — the method to call when the timer expires.
        self._timer = threading.Timer(self._wait, self._fire)

        # .start() — begin the countdown. The timer runs on its own thread.
        # Python's timer threads are always daemon threads when created this way,
        # meaning they do not prevent the process from exiting.
        self._timer.start()

    def cancel(self) -> None:
        """
        Cancel the pending timer without flushing.

        Used when the pipeline switches from auto to manual mode — any buffered
        segments are discarded along with the timer.
        """
        if self._timer is not None:
            self._timer.cancel()   # stop the countdown
            self._timer = None     # clear the reference

    def clear(self) -> None:
        """
        Cancel the timer and discard all buffered text.

        Used when switching modes or restarting a session cleanly.
        """
        self.cancel()             # cancel any running timer first
        self._pending.clear()     # list.clear() — remove all elements from the list

    def _fire(self) -> None:
        """
        Internal: called by the timer thread when the quiet period elapses.

        Joins all pending text into one string, resets the buffer, and calls
        on_flush with the result. Skips the callback if the buffer is empty.
        """
        # Only flush if there is something to flush.
        # `if self._pending` — a list is falsy when empty, truthy when non-empty.
        if self._pending:
            # " ".join(self._pending) — concatenate all buffered segments with spaces.
            # e.g. ["Can you", "walk me through that?"] → "Can you walk me through that?"
            full_text: str = " ".join(self._pending)

            # Reset the buffer and timer reference before calling the callback.
            # This ensures that any add() calls made inside on_flush() start a
            # fresh timer rather than overlapping with the old state.
            self._pending = []   # replace with a new empty list (not .clear()) — thread-safe
            self._timer = None   # clear the timer reference; the timer has already fired

            # Call the injected callback with the assembled text.
            # The callback is responsible for deciding what to do (e.g. call OpenAI).
            self._on_flush(full_text)
