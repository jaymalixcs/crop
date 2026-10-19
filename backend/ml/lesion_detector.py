"""
lesion_detector.py
~~~~~~~~~~~~~~~~~~
Detects individual disease/bacterial lesion spots on a leaf image using OpenCV
color-space analysis and morphological processing.

Works well on both clean (white-background) PlantVillage-style images AND
real-life field photos with complex backgrounds because it:
  - Accepts an optional YOLO bounding-box to restrict the search area.
  - Uses ADAPTIVE local-contrast thresholding (not just a global fixed value),
    which avoids flagging leaf veins and natural shadows as lesions.
  - Combines multiple colour cues (HSV brown/orange + LAB adaptive dark).
  - Allows up to 70% of the leaf area to be diseased (for severe blight).
  - Draws a YOLO green bounding-box + red lesion circles in one annotated image.
"""

import cv2
import numpy as np
import io
import base64
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple, Optional


# ---------------------------------------------------------------------------
# Spot detection
# ---------------------------------------------------------------------------

def detect_disease_spots(
    image_bytes: bytes,
    yolo_bbox: Optional[List[float]] = None,
) -> Tuple[List[Dict[str, Any]], float, Image.Image]:
    """Detect diseased spots/lesions on a leaf.

    Parameters
    ----------
    image_bytes : bytes
        Raw image data.
    yolo_bbox : [x1, y1, x2, y2] in pixel coords, optional
        If provided the spot search is restricted to this region, which greatly
        improves precision on real-life photos with complex backgrounds.

    Returns
    -------
    spots : list[dict]
        Each dict has: center (x,y), radius (px), bbox [x1,y1,x2,y2], area.
        Coordinates are in the *original full-image* coordinate space.
    severity_ratio : float  (0–1)
    pil_image : PIL.Image   (original, un-annotated)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return [], 0.0, pil_img

    h, w = img_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # ------------------------------------------------------------------ #
    # 1.  Define Region-Of-Interest                                        #
    # ------------------------------------------------------------------ #
    if yolo_bbox and len(yolo_bbox) == 4:
        rx1, ry1, rx2, ry2 = [int(v) for v in yolo_bbox]
        pad_w = max(5, int((rx2 - rx1) * 0.05))
        pad_h = max(5, int((ry2 - ry1) * 0.05))
        rx1 = max(0, rx1 - pad_w)
        ry1 = max(0, ry1 - pad_h)
        rx2 = min(w, rx2 + pad_w)
        ry2 = min(h, ry2 + pad_h)
        roi_bgr = img_bgr[ry1:ry2, rx1:rx2]
    else:
        rx1, ry1 = 0, 0
        rx2, ry2 = w, h
        roi_bgr = img_bgr

    rh, rw = roi_bgr.shape[:2]
    if rh < 10 or rw < 10:
        return [], 0.0, pil_img

    # ------------------------------------------------------------------ #
    # 2.  Compute colour spaces                                            #
    # ------------------------------------------------------------------ #
    hsv_roi  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lab_roi  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    L_chan   = lab_roi[:, :, 0]

    # ------------------------------------------------------------------ #
    # 3.  Leaf mask (contour-filled to prevent expanding onto background) #
    # ------------------------------------------------------------------ #
    # Green plant tissue (H=20-100, S>=25)
    green_tissue = cv2.inRange(hsv_roi, np.array([20, 25, 20]), np.array([100, 255, 255]))

    # Find leaf contours and fill internal holes WITHOUT spilling onto background
    cnts, _ = cv2.findContours(green_tissue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros((rh, rw), dtype=np.uint8)
    if cnts:
        # Keep contours that represent leaf portions
        main_cnts = [c for c in cnts if cv2.contourArea(c) > max(100, int(rh * rw * 0.01))]
        if main_cnts:
            cv2.drawContours(leaf_mask, main_cnts, -1, 255, -1)
        else:
            c_leaf = max(cnts, key=cv2.contourArea)
            cv2.drawContours(leaf_mask, [c_leaf], -1, 255, -1)

    leaf_area = int(cv2.countNonZero(leaf_mask))

    # Fallback when green mask is very sparse (e.g. completely yellowed/dessicated leaf)
    if leaf_area < int(rh * rw * 0.05):
        _, bg_inv = cv2.threshold(gray_roi, 220, 255, cv2.THRESH_BINARY_INV)
        leaf_mask = bg_inv
        leaf_area = int(cv2.countNonZero(leaf_mask)) or (rh * rw)

    # Safe interior (erode slightly so boundary drop-shadows on background aren't mistaken for lesions)
    k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    leaf_safe = cv2.erode(leaf_mask, k_er) if leaf_area > 500 else leaf_mask

    # ------------------------------------------------------------------ #
    # 4.  Disease spot masks                                               #
    # ------------------------------------------------------------------ #
    a_chan = lab_roi[:, :, 1]
    b_chan = lab_roi[:, :, 2]
    h_chan = hsv_roi[:, :, 0]
    s_chan = hsv_roi[:, :, 1]

    # --- cue A: Warm-hue necrotic blight/bacterial patches (brown/tan/rust) ---
    # Non-green hue: H in [5, 33] (orange/brown) or [165, 180] (red wrap-around)
    is_warm_hue = ((h_chan >= 5) & (h_chan <= 33)) | (h_chan >= 165)
    # Must have color saturation (S >= 30) and shift away from healthy deep green (a* >= 120)
    blight_brown = is_warm_hue & (s_chan >= 30) & (a_chan >= 120) & (leaf_safe > 0)

    # --- cue B: Local contrast dark spots (small fungal/bacterial leaf specks) ---
    blur_L = cv2.GaussianBlur(L_chan, (5, 5), 0)
    local_mean = cv2.GaussianBlur(blur_L.astype(np.float32), (25, 25), 0)
    diff = local_mean - blur_L.astype(np.float32)
    dark_spots = (diff > 18) & (a_chan >= 116) & (leaf_safe > 0)

    # --- cue C: Yellowing chlorosis/halo (early blight ring, b* > 152) ------
    yellow_halo = (b_chan > 152) & (a_chan >= 115) & (leaf_safe > 0)

    # Combine all disease cues
    spot_mask = (blight_brown | dark_spots | yellow_halo).astype(np.uint8) * 255

    # Morphological clean-up: remove tiny specks, connect nearby lesion regions
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, kernel_open)
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_CLOSE, kernel_close)

    # ------------------------------------------------------------------ #
    # 5.  Find contours → build spot list                                  #
    # ------------------------------------------------------------------ #
    contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Allow spots down to 20 px (catches small satellite lesions) up to 60 % of leaf area
    min_area = max(20, int(leaf_area * 0.0001))
    max_area = int(leaf_area * 0.60)

    spots: List[Dict[str, Any]] = []
    total_spot_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            total_spot_area += area
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            spots.append({
                "center": (int(cx) + rx1, int(cy) + ry1),   # full-image coords
                "radius": max(8, int(radius)),
                "bbox":   [x + rx1, y + ry1, x + bw + rx1, y + bh + ry1],
                "area":   float(area),
            })

    # Sort largest first; keep at most 25 (for readable visualisation)
    spots.sort(key=lambda s: s["area"], reverse=True)
    spots = spots[:25]

    severity_ratio = min(1.0, total_spot_area / float(leaf_area)) if leaf_area > 0 else 0.0

    return spots, severity_ratio, pil_img


# ---------------------------------------------------------------------------
# Annotation renderer
# ---------------------------------------------------------------------------

def draw_spot_circles(
    image_bytes: bytes,
    spots: List[Dict[str, Any]],
    main_label: str,
    confidence: float,
    yolo_detections: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Draw YOLO bounding-boxes (green) and lesion bounding-boxes + circles (red).

    Returns a base64-encoded PNG string.
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        annotated = image.convert("RGB")

    draw = ImageDraw.Draw(annotated)
    img_w, img_h = annotated.size
    line_w = max(2, int(min(img_w, img_h) / 160))

    # ── YOLO bounding box(es) in bright green ─────────────────────────────
    if yolo_detections:
        for det in yolo_detections:
            bbox = det.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                conf_pct = int(det.get("confidence", 0) * 100)
                lbl = f'{det["class_name"].replace("_", " ")} {conf_pct}%'
                draw.rectangle([x1, y1, x2, y2], outline="#22c55e", width=max(line_w, 3))
                lbl_bb = draw.textbbox((x1, max(0, y1 - 20)), lbl)
                draw.rectangle(lbl_bb, fill="#22c55e")
                draw.text((x1, max(0, y1 - 20)), lbl, fill="white")

    # ── Individual lesion spots: Red Bounding Boxes (matching paper) ─────
    if spots:
        for i, spot in enumerate(spots):
            bx1, by1, bx2, by2 = spot["bbox"]
            cx, cy = spot["center"]
            r = spot["radius"] + 2

            # Red bounding box around each disease spot (paper style)
            draw.rectangle([bx1 - 2, by1 - 2, bx2 + 2, by2 + 2],
                           outline="#ef4444", width=max(2, line_w - 1))

            # Outer amber corner / circle accent for visual prominence
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         outline="#fbbf24", width=1)

            # Tag label for top-5 largest spots
            if i < 5:
                tag = f"#{i+1}"
                tx = max(0, min(bx1, img_w - 32))
                ty = max(0, by1 - 15)
                tb = draw.textbbox((tx, ty), tag)
                draw.rectangle(tb, fill="#ef4444")
                draw.text((tx, ty), tag, fill="white")

    # ── Header banner ────────────────────────────────────────────────────
    n = len(spots)
    banner = (
        f"  {main_label}  {confidence:.0%}  │  "
        f"{n} lesion{'s' if n != 1 else ''} detected  "
    )
    lb = draw.textbbox((0, 0), banner)
    bh_px = lb[3] - lb[1]
    draw.rectangle([0, 0, img_w, bh_px + 12], fill="#0f172a")
    draw.text((10, 6), banner, fill="#f1f5f9")

    # ── Encode ───────────────────────────────────────────────────────────
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
