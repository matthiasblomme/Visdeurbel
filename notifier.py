import os
import requests
from dotenv import load_dotenv

load_dotenv()


class TelegramNotifier:
    def __init__(self):
        self._token = os.environ["TELEGRAM_BOT_TOKEN"]
        self._chat_id = os.environ["TELEGRAM_CHAT_ID"]
        self._base = f"https://api.telegram.org/bot{self._token}"

    def send(self, image_path: str, caption: str = "🐟 Fish spotted at visdeurbel!") -> bool:
        """Send a photo with caption to Telegram. Returns True on success."""
        try:
            with open(image_path, "rb") as photo:
                resp = requests.post(
                    f"{self._base}/sendPhoto",
                    data={"chat_id": self._chat_id, "caption": caption},
                    files={"photo": photo},
                    timeout=15,
                )
            if resp.ok:
                return True
            print(f"[Telegram] Error {resp.status_code}: {resp.text}")
            return False
        except Exception as e:
            print(f"[Telegram] Exception: {e}")
            return False

    def send_text(self, text: str) -> bool:
        """Send a plain text message."""
        try:
            resp = requests.post(
                f"{self._base}/sendMessage",
                data={"chat_id": self._chat_id, "text": text},
                timeout=15,
            )
            return resp.ok
        except Exception as e:
            print(f"[Telegram] Exception: {e}")
            return False
