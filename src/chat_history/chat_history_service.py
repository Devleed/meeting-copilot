import uuid
from datetime import datetime


class _ChatEntry:
    def __init__(self, they_said: str, response: str, timestamp: datetime) -> None:
        self.they_said = they_said
        self.response = response
        self.timestamp = timestamp


class ChatHistoryService:
    """
    Records every exchange (they_said + LLM response) for one session.

    A session begins when this object is constructed and ends when end_session()
    or export_txt() is called. The session_id is an 8-char hex string derived
    from a UUID, short enough to display but unique enough to avoid collisions.

    Usage:
        svc = ChatHistoryService()
        svc.add_entry("What's the timeline?", "ANSWER: ...\nFOLLOW-UP: ...")
        svc.display()
        svc.export_txt()          # writes chat_history_<id>_<ts>.txt
    """

    def __init__(self) -> None:
        self.session_id: str = uuid.uuid4().hex[:8]
        self.started_at: datetime = datetime.now()
        self._entries: list = []

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_entry(self, they_said: str, response: str) -> None:
        self._entries.append(_ChatEntry(they_said, response, datetime.now()))

    # ── Read ───────────────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    # ── Output ─────────────────────────────────────────────────────────────────

    def display(self) -> None:
        ended_at = datetime.now()
        print("\n" + "═" * 62)
        print(f"  CHAT HISTORY  —  Session: {self.session_id}")
        print(f"  Started  : {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Ended    : {ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Exchanges: {len(self._entries)}")
        print("═" * 62)

        if self.is_empty():
            print("\n  (no exchanges recorded in this session)\n")
            return

        for entry in self._entries:
            print(f"\n[{entry.timestamp.strftime('%H:%M:%S')}] THEY SAID: {entry.they_said}")
            print()
            print(entry.response)
            print("\n" + "─" * 62)

    def export_txt(self, filepath: str = None) -> str:
        ended_at = datetime.now()
        if filepath is None:
            filepath = (
                f"chat_history_{self.session_id}"
                f"_{ended_at.strftime('%Y%m%d_%H%M%S')}.txt"
            )

        lines = [
            "=" * 62,
            f"  CHAT HISTORY  —  Session: {self.session_id}",
            f"  Started  : {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Ended    : {ended_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Exchanges: {len(self._entries)}",
            "=" * 62,
        ]

        if self.is_empty():
            lines.append("\n(no exchanges recorded in this session)\n")
        else:
            for entry in self._entries:
                lines.append("")
                lines.append(
                    f"[{entry.timestamp.strftime('%H:%M:%S')}] THEY SAID: {entry.they_said}"
                )
                lines.append("")
                lines.append(entry.response)
                lines.append("")
                lines.append("-" * 62)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return filepath
