import base64
import os
import requests

import config


class FishVerifier:
    def __init__(self):
        """Resolve the llama-server URL from the environment (falls back to config.LLM_URL).

        The LLM_URL env var is set by docker-compose so the detector container can
        reach the llm service by its Docker service name (http://llm:8080).
        """
        self._url = os.getenv("LLM_URL", config.LLM_URL)

    def verify(self, snapshot_path: str) -> bool:
        """Ask Granite Vision whether the snapshot contains a fish.
        Returns True if confirmed, or True if LLM unavailable (fail-open)."""
        try:
            with open(snapshot_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            resp = requests.post(
                f"{self._url}/v1/chat/completions",
                json={
                    "model": "granite-vision",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        "This is a frame from an underwater riverbed camera. "
                                        "Is there a fish visible in this image? "
                                        "Answer only 'yes' or 'no'."
                                    ),
                                },
                            ],
                        }
                    ],
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
                timeout=config.LLM_TIMEOUT,
            )
            if resp.ok:
                answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
                confirmed = "yes" in answer
                label = "fish" if confirmed else "rejected"
                print(f"[LLM] '{answer}' -> {label}")
                return confirmed
        except Exception as e:
            print(f"[LLM] Error: {e} — allowing notification (fail-open).")
        return True  # fail-open: never block notifications due to LLM issues
