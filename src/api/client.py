import json

import httpx


class RemoteSuggestionClient:
    """
    Calls the suggestion REST API and prints streamed responses to the terminal.

    Implements the same .generate(text, manual) interface as SuggestionGenerator
    so it can be passed to AudioPipeline in place of the local generator.

    Usage:
        client = RemoteSuggestionClient("http://localhost:8000")
        client.generate("Can you walk me through the architecture?")
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def generate(self, text: str, manual: bool = False) -> None:
        print("\n" + "─" * 50)
        print(f"THEY SAID: {text}")
        print()
        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    f"{self._base_url}/suggest/stream",
                    json={"text": text, "manual": manual},
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        print(json.loads(data), end="", flush=True)
            print("\n" + "─" * 50 + "\n")
        except Exception as e:
            print(f"[Remote suggestion error]: {e}")
