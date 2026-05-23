import json
import urllib.request
import urllib.error


class SpeechForwarder:
    """
    Drop-in replacement for SuggestionGenerator that forwards transcribed speech
    to a remote URL via HTTP POST instead of calling an LLM.

    POST body: {"speech": "<transcribed text>", "manual": boolean}
    """

    def __init__(self, url: str) -> None:
        self._url = url

    def generate(self, text: str, manual: bool = False) -> None:
        print(f"[ forwarding speech to {self._url} ] text: {text}")

        payload = json.dumps({"speech": text, "manual": manual}).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"[ forwarded to {self._url} — {resp.status} ]")
        except urllib.error.URLError as exc:
            print(f"[ forward failed: {exc.reason} ]")
