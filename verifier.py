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
                                        "You are reviewing a frame from a low-visibility underwater monitoring camera to decide whether a fish is visible.\n\n"
                                        "Context:\n"
                                        "- The water is murky, noisy, low-contrast, and may contain haze, sediment, blur, shadows, and reflections.\n"
                                        "- Fish may appear from the side, head-on, tail-on, partially cropped, very blurry, or very close to the lens.\n"
                                        "- Ignore any green boxes, timestamps, labels, or overlays.\n\n"
                                        "Answer YES only if there is a plausible fish form present.\n"
                                        "A plausible fish form means one or more of these:\n"
                                        "- an elongated or tapered body\n"
                                        "- a coherent curved body mass\n"
                                        "- a head/body/tail relationship\n"
                                        "- a fin, tail, or fish-like silhouette\n"
                                        "- a partial but still believable fish-shaped body\n\n"
                                        "Answer NO if the image shows only:\n"
                                        "- uniform murk or haze\n"
                                        "- sediment clouds or floating particles\n"
                                        "- vague shadow patches\n"
                                        "- reflections or light artifacts\n"
                                        "- shapeless dark blobs without a coherent fish form\n\n"
                                        "Important rule:\n"
                                        "Do NOT require a perfect, sharp fish.\n"
                                        "Do NOT answer YES for a vague blob alone.\n"
                                        "Answer YES when there is a believable fish-like structure, even if partial or blurry.\n"
                                        "Answer NO when the shape is only ambiguous murk or debris.\n\n"
                                        "Reply with exactly one word: YES or NO"
                                    ),
                                },
                            ],
                        }
                    ],
                    "max_tokens": 10,
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
