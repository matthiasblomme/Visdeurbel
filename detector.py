import cv2
import numpy as np
import config


class FishDetector:
    def __init__(self):
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_THRESHOLD,
            detectShadows=False,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process_frame(self, frame):
        """
        Analyse a single frame for fish.

        Returns:
            detected (bool): True if at least one fish-sized blob was found.
            annotated (np.ndarray): Frame with bounding boxes drawn (copy of input).
        """
        small = cv2.resize(frame, (640, 360))
        frame_area = small.shape[0] * small.shape[1]  # 640*360 = 230400

        # Strong blur kills sediment particles and water shimmer before MOG2 sees them
        blurred = cv2.GaussianBlur(small, (config.BLUR_KERNEL, config.BLUR_KERNEL), 0)
        mask = self._subtractor.apply(blurred)

        # Remove noise: erode then dilate
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = small.copy()
        detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_CONTOUR_AREA:
                continue  # too small — sediment / particle

            x, y, w, h = cv2.boundingRect(cnt)

            if max(w, h) < config.MIN_FISH_LENGTH:
                continue  # bounding box too compact — not a fish

            if (w * h) > config.MAX_FRAME_COVERAGE * frame_area:
                continue  # covers too much of the frame — overall light/shimmer change

            detected = True
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"Fish! ({int(area)}px)",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return detected, annotated
