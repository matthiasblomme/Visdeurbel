import os
import time
import datetime
import cv2

import config
from detector import FishDetector
from notifier import TelegramNotifier


def save_snapshot(frame) -> str:
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.SNAPSHOT_DIR, f"fish_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def open_stream():
    cap = cv2.VideoCapture(config.HLS_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    detector = FishDetector()
    notifier = TelegramNotifier()

    print(f"[Visdeurbel] Starting — stream: {config.HLS_URL}")
    notifier.send_text("Visdeurbel fish detector started.")

    last_notification = 0.0
    last_processed = 0.0

    cap = open_stream()
    if not cap.isOpened():
        print("[ERROR] Could not open stream. Check the URL or your internet connection.")
        return

    print("[Visdeurbel] Stream opened. Watching for fish...")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[Visdeurbel] Stream lost, reconnecting in 10s...")
                cap.release()
                time.sleep(10)
                cap = open_stream()
                continue

            now = time.time()
            if now - last_processed < config.FRAME_INTERVAL_SEC:
                continue
            last_processed = now

            detected, annotated = detector.process_frame(frame)

            if config.SHOW_PREVIEW:
                cv2.imshow("Visdeurbel", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if detected:
                cooldown_remaining = config.COOLDOWN_SEC - (now - last_notification)
                if cooldown_remaining <= 0:
                    path = save_snapshot(annotated)
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    caption = f"Fish spotted at visdeurbel! ({ts})"
                    ok = notifier.send(path, caption)
                    if ok:
                        print(f"[{ts}] Fish detected — notification sent. Snapshot: {path}")
                        last_notification = now
                    else:
                        print(f"[{ts}] Fish detected — notification FAILED.")
                else:
                    print(
                        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Fish detected "
                        f"(cooldown {int(cooldown_remaining)}s remaining)"
                    )
            else:
                print(
                    f"[{datetime.datetime.now().strftime('%H:%M:%S')}] No fish.",
                    end="\r",
                )

    except KeyboardInterrupt:
        print("\n[Visdeurbel] Stopped by user.")
    finally:
        cap.release()
        if config.SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
