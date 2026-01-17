import cv2
import numpy as np


def build_default_template():
    canvas = np.zeros((220, 220), dtype=np.uint8)
    cv2.putText(
        canvas,
        "6",
        (35, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        5.5,
        255,
        15,
        cv2.LINE_AA,
    )
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    return thresh


def contour_has_hole(hierarchy, idx):
    if hierarchy is None:
        return False
    # hierarchy: [next, prev, first_child, parent]
    return hierarchy[idx][2] != -1


def match_six(contour, template_contour, score_threshold):
    if template_contour is None:
        return False, None
    score = cv2.matchShapes(contour, template_contour, cv2.CONTOURS_MATCH_I1, 0.0)
    return score < score_threshold, score


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    template_contour = build_default_template()
    score_threshold = 0.35
    min_area = 400
    max_area = 200000

    print("Controls: [t] capture template from largest contour, [r] reset template, [q] quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        thresh = preprocess(frame)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        hierarchy = hierarchy[0] if hierarchy is not None and len(hierarchy) > 0 else None

        candidates = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / float(h)
            if aspect < 0.35 or aspect > 1.3:
                continue

            has_hole = contour_has_hole(hierarchy, idx)
            candidates.append((idx, cnt, x, y, w, h, has_hole))

        for idx, cnt, x, y, w, h, has_hole in candidates:
            is_six, score = match_six(cnt, template_contour, score_threshold)
            if not is_six and has_hole:
                is_six, score = match_six(cnt, template_contour, score_threshold + 0.1)

            if is_six:
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = max(h / 80.0, 0.5)
                thickness = max(int(scale * 3), 2)
                (text_w, text_h), _ = cv2.getTextSize("7", font, scale, thickness)
                text_x = x + w + 8
                text_y = y + h - max(int(h * 0.1), 5)
                cv2.putText(
                    frame,
                    "7",
                    (text_x, text_y),
                    font,
                    scale,
                    (0, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        cv2.imshow("6 to 67", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t") and candidates:
            # Capture the largest contour as the new template.
            largest = max(candidates, key=lambda item: cv2.contourArea(item[1]))
            template_contour = largest[1]
            print("Template captured.")
        if key == ord("r"):
            template_contour = build_default_template()
            print("Template reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
