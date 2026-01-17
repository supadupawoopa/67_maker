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


def build_default_template_images(size):
    templates = []
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_PLAIN,
    ]
    for font in fonts:
        for thickness in (10, 14, 18):
            canvas = np.zeros((220, 220), dtype=np.uint8)
            cv2.putText(
                canvas,
                "6",
                (35, 185),
                font,
                5.5,
                255,
                thickness,
                cv2.LINE_AA,
            )
            templates.append(normalize_roi(canvas, size))
    return templates


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


def contour_hole_ratio(contours, hierarchy, idx):
    if hierarchy is None:
        return 0.0
    child = hierarchy[idx][2]
    if child == -1:
        return 0.0
    parent_area = cv2.contourArea(contours[idx])
    if parent_area <= 0:
        return 0.0
    # Use the largest child contour as the "hole" area.
    hole_areas = []
    while child != -1:
        hole_areas.append(cv2.contourArea(contours[child]))
        child = hierarchy[child][0]
    if not hole_areas:
        return 0.0
    return max(hole_areas) / parent_area


def contour_solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return area / hull_area


def normalize_roi(roi, size):
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.uint8)
    scale = size / float(max(h, w))
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def match_six(
    contour,
    template_contour,
    score_threshold,
    roi,
    template_imgs,
    template_threshold_low,
    template_threshold_high,
):
    if template_contour is None or not template_imgs:
        return False, None, None
    shape_score = cv2.matchShapes(contour, template_contour, cv2.CONTOURS_MATCH_I1, 0.0)
    norm_roi = normalize_roi(roi, template_imgs[0].shape[0])
    template_score = max(
        cv2.matchTemplate(norm_roi, template, cv2.TM_CCOEFF_NORMED)[0][0]
        for template in template_imgs
    )
    is_match = (
        (shape_score < score_threshold and template_score > template_threshold_low)
        or template_score > template_threshold_high
    )
    return is_match, shape_score, template_score


def text_scale_for_height(font, target_height):
    base_size, _ = cv2.getTextSize("7", font, 1.0, 1)
    base_height = base_size[1]
    if base_height <= 0:
        return 1.0
    return max((target_height / float(base_height)) * 0.92, 0.4)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    template_contour = build_default_template()
    template_size = 64
    template_imgs = build_default_template_images(template_size)
    score_threshold = 0.45
    template_threshold_low = 0.2
    template_threshold_high = 0.45
    min_area = 400
    max_area = 200000
    min_solidity = 0.5
    min_hole_ratio = 0.04
    max_hole_ratio = 0.6
    hold_frames = 15

    print("Controls: [t] capture template from largest contour, [r] reset template, [q] quit.")

    frame_idx = 0
    last_overlays = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

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
            if not has_hole:
                continue

            hole_ratio = contour_hole_ratio(contours, hierarchy, idx)
            if hole_ratio < min_hole_ratio or hole_ratio > max_hole_ratio:
                continue

            solidity = contour_solidity(cnt)
            if solidity < min_solidity:
                continue

            roi = thresh[y : y + h, x : x + w]
            candidates.append((idx, cnt, x, y, w, h, roi))

        current_overlays = []
        for idx, cnt, x, y, w, h, roi in candidates:
            is_six, shape_score, template_score = match_six(
                cnt,
                template_contour,
                score_threshold,
                roi,
                template_imgs,
                template_threshold_low,
                template_threshold_high,
            )

            if is_six:
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = text_scale_for_height(font, h)
                current_overlays.append((frame_idx, x, y, w, h, scale))

        if current_overlays:
            last_overlays = current_overlays

        if last_overlays:
            kept = []
            for last_frame, x, y, w, h, scale in last_overlays:
                if frame_idx - last_frame > hold_frames:
                    continue
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = max(int(scale * 2.5), 2)
                (text_w, text_h), baseline = cv2.getTextSize("7", font, scale, thickness)
                gap = max(int(w * 0.05), 6)
                text_x = x + w + gap
                text_y = y + h - max(int(h * 0.02), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
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
                kept.append((last_frame, x, y, w, h, scale))
            last_overlays = kept

        cv2.imshow("6 to 67", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t") and candidates:
            # Capture the largest contour as the new template.
            largest = max(candidates, key=lambda item: cv2.contourArea(item[1]))
            template_contour = largest[1]
            template_imgs = [normalize_roi(largest[6], template_size)]
            print("Template captured.")
        if key == ord("r"):
            template_contour = build_default_template()
            template_imgs = build_default_template_images(template_size)
            print("Template reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
