import os
import time
import datetime
import cv2

import config
from detector import FishDetector
from notifier import TelegramNotifier
from verifier import FishVerifier


def save_snapshot(frame) -> str:
    """Save an annotated frame as a timestamped JPEG in the snapshots directory.

    Returns the path to the saved file so it can be forwarded to Telegram.
    """
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.SNAPSHOT_DIR, f"fish_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def open_stream():
    """Open the HLS livestream via OpenCV's FFmpeg backend.

    Buffer size is set to 1 to minimise latency — we want the most recent frame,
    not a queue of older ones.
    """
    cap = cv2.VideoCapture(config.HLS_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    """Main application loop.

    Initialises the detector, notifier, and verifier, then continuously reads
    frames from the HLS stream. On a positive motion detection:
      1. Save an annotated snapshot.
      2. Optionally confirm with the vision LLM (LLM_VERIFY).
      3. Send a Telegram notification if confirmed and the cooldown has elapsed.

    Reconnects automatically if the stream drops.
    """
    detector = FishDetector()
    notifier = TelegramNotifier()
    verifier = FishVerifier()

    print(f"[Visdeurbel] Starting — stream: {config.HLS_URL}")
    notifier.send_text("Visdeurbel fish detector started.")

    last_notification = 0.0  # timestamp of the last successful Telegram notification
    last_processed = 0.0     # timestamp of the last processed frame

    cap = open_stream()
    if not cap.isOpened():
        print("[ERROR] Could not open stream. Check the URL or your internet connection.")
        return

    print("[Visdeurbel] Stream opened. Watching for fish...")

    try:
        while True:
            ret, frame = cap.read()

            # Stream dropped — wait and reconnect
            if not ret:
                print("[Visdeurbel] Stream lost, reconnecting in 10s...")
                cap.release()
                time.sleep(10)
                cap = open_stream()
                continue

            # Throttle: only process one frame every FRAME_INTERVAL_SEC seconds
            now = time.time()
            if now - last_processed < config.FRAME_INTERVAL_SEC:
                continue
            last_processed = now

            detected, annotated = detector.process_frame(frame)

            # Optional debug window — disabled in production (SHOW_PREVIEW = False)
            if config.SHOW_PREVIEW:
                cv2.imshow("Visdeurbel", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if detected:
                cooldown_remaining = config.COOLDOWN_SEC - (now - last_notification)
                if cooldown_remaining <= 0:
                    path = save_snapshot(annotated)
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Second-pass: ask the vision LLM to confirm before notifying
                    if config.LLM_VERIFY and not verifier.verify(path):
                        print(f"[{ts}] OpenCV hit — LLM rejected. Skipping notification.")
                    else:
                        caption = f"Fish spotted at visdeurbel! ({ts})"
                        ok = notifier.send(path, caption)
                        if ok:
                            print(f"[{ts}] Fish detected — notification sent. Snapshot: {path}")
                            last_notification = now
                        else:
                            print(f"[{ts}] Fish detected — notification FAILED.")
                else:
                    # Still in cooldown — log but don't notify
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
