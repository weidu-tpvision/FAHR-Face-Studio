import os
import math
import threading
import urllib.request
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")

_detector_lock = threading.Lock()
_cached_detector: Optional[ Any ] = None


def ensure_yunet_model() -> str:
    """Download the lightweight (230 KB) OpenCV YuNet 5-landmark face detector if not cached."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(YUNET_PATH):
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    return YUNET_PATH


def _get_detector(input_size: Tuple[int, int], score_threshold: float) -> Any:
    global _cached_detector
    model_path = ensure_yunet_model()
    if _cached_detector is None:
        _cached_detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=50,
        )
    else:
        _cached_detector.setInputSize(input_size)
        _cached_detector.setScoreThreshold(score_threshold)
    return _cached_detector


def align_and_crop_face(
    img_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    right_eye: Tuple[float, float],
    left_eye: Tuple[float, float],
) -> Image.Image:
    """
    Replicates RetinaFace.extract_faces(align=True):
    1. Computes the angle between the two eye landmarks so the inter-ocular line is horizontal.
    2. Rotates the image around the midpoint between the eyes.
    3. Crops the tightly bounded face region and returns a PIL RGB Image.
    """
    h, w = img_rgb.shape[:2]
    x, y, bw, bh = bbox

    if right_eye[0] > left_eye[0]:
        right_eye, left_eye = left_eye, right_eye

    dx = float(left_eye[0] - right_eye[0])
    dy = float(left_eye[1] - right_eye[1])
    angle = math.degrees(math.atan2(dy, dx))

    eyes_center = (
        float(right_eye[0] + left_eye[0]) / 2.0,
        float(right_eye[1] + left_eye[1]) / 2.0,
    )

    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_rgb = cv2.warpAffine(
        img_rgb,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    box_cx = x + bw / 2.0
    box_cy = y + bh / 2.0
    rel_x = box_cx - eyes_center[0]
    rel_y = box_cy - eyes_center[1]
    rad = math.radians(-angle)
    rot_cx = eyes_center[0] + rel_x * math.cos(rad) - rel_y * math.sin(rad)
    rot_cy = eyes_center[1] + rel_x * math.sin(rad) + rel_y * math.cos(rad)

    x1 = max(0, int(round(rot_cx - bw / 2.0)))
    y1 = max(0, int(round(rot_cy - bh / 2.0)))
    x2 = min(w, int(round(rot_cx + bw / 2.0)))
    y2 = min(h, int(round(rot_cy + bh / 2.0)))

    if x2 <= x1 or y2 <= y1:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)

    cropped = aligned_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        return Image.fromarray(img_rgb)
    return Image.fromarray(cropped)


def detect_and_align_faces(
    pil_image: Image.Image,
    score_threshold: float = 0.6,
    already_cropped: bool = False,
) -> List[Dict[str, Any]]:
    """
    Detects all faces in a PIL image, aligns each one horizontally by eye landmarks,
    and returns metadata + aligned PIL images ready for 112x112 FAHR-Face inference.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    orig_h, orig_w = img_rgb.shape[:2]

    if already_cropped:
        return [
            {
                "face_id": 0,
                "bbox": [0, 0, orig_w, orig_h],
                "confidence": 1.0,
                "alignment_angle_deg": 0.0,
                "interocular_px": None,
                "landmarks": None,
                "aligned_pil": pil_image.convert("RGB"),
                "is_fallback": False,
            }
        ]

    max_dim = 1024
    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / float(max(orig_h, orig_w))
        det_w = int(round(orig_w * scale))
        det_h = int(round(orig_h * scale))
        det_bgr = cv2.cvtColor(
            cv2.resize(img_rgb, (det_w, det_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2BGR,
        )
    else:
        det_w, det_h = orig_w, orig_h
        det_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    with _detector_lock:
        detector = _get_detector((det_w, det_h), score_threshold)
        _, faces = detector.detect(det_bgr)

    results: List[Dict[str, Any]] = []
    if faces is not None and len(faces) > 0:
        faces_list = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        for idx, face in enumerate(faces_list):
            x, y, bw, bh = [int(round(v / scale)) for v in face[:4]]
            x = max(0, x)
            y = max(0, y)
            bw = min(orig_w - x, bw)
            bh = min(orig_h - y, bh)
            if bw < 16 or bh < 16:
                continue

            right_eye = (float(face[4] / scale), float(face[5] / scale))
            left_eye = (float(face[6] / scale), float(face[7] / scale))
            nose = (float(face[8] / scale), float(face[9] / scale))
            right_mouth = (float(face[10] / scale), float(face[11] / scale))
            left_mouth = (float(face[12] / scale), float(face[13] / scale))
            conf = float(face[14])

            aligned_pil = align_and_crop_face(
                img_rgb, (x, y, bw, bh), right_eye, left_eye
            )

            re_pt, le_pt = right_eye, left_eye
            if re_pt[0] > le_pt[0]:
                re_pt, le_pt = le_pt, re_pt
            dx = le_pt[0] - re_pt[0]
            dy = le_pt[1] - re_pt[1]
            align_angle = round(math.degrees(math.atan2(dy, dx)), 2)
            interocular_dist = round(math.hypot(dx, dy), 1)

            results.append(
                {
                    "face_id": idx,
                    "bbox": [x, y, bw, bh],
                    "confidence": round(conf, 4),
                    "alignment_angle_deg": align_angle,
                    "interocular_px": interocular_dist,
                    "landmarks": {
                        "right_eye": [round(right_eye[0], 1), round(right_eye[1], 1)],
                        "left_eye": [round(left_eye[0], 1), round(left_eye[1], 1)],
                        "nose": [round(nose[0], 1), round(nose[1], 1)],
                        "right_mouth": [
                            round(right_mouth[0], 1),
                            round(right_mouth[1], 1),
                        ],
                        "left_mouth": [
                            round(left_mouth[0], 1),
                            round(left_mouth[1], 1),
                        ],
                    },
                    "aligned_pil": aligned_pil,
                    "is_fallback": False,
                }
            )

    if not results:
        results.append(
            {
                "face_id": 0,
                "bbox": [0, 0, orig_w, orig_h],
                "confidence": 1.0,
                "alignment_angle_deg": 0.0,
                "interocular_px": None,
                "landmarks": None,
                "aligned_pil": pil_image.convert("RGB"),
                "is_fallback": True,
            }
        )

    return results
