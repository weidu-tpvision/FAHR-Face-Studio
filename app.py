import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image, ImageOps

from face_pipeline import detect_and_align_faces
from model_service import FAHRFaceService, pil_to_base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(
    title="FAHR-Face Studio API",
    description="Interactive Biological FaceAge, ViT Attention Rollout & MAE Foundation Model Explorer",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = FAHRFaceService()


class MAERequest(BaseModel):
    aligned_face_b64: str
    mask_ratio: float = 0.75
    seed: Optional[int] = None


def decode_base64_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


@app.get("/api/status")
def api_status(warmup: bool = False):
    if warmup:
        service.ensure_age_model()
        service.ensure_mae_model()
    return service.get_status()


@app.post("/api/analyze")
async def api_analyze(
    file: UploadFile = File(...),
    chronological_age: Optional[float] = Form(None),
    already_cropped: bool = Form(False),
    score_threshold: float = Form(0.55),
):
    try:
        contents = await file.read()
        raw_img = Image.open(io.BytesIO(contents))
        pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    faces_meta = detect_and_align_faces(
        pil_img,
        score_threshold=score_threshold,
        already_cropped=already_cropped,
    )

    analyzed_faces = []
    for f in faces_meta:
        age_res = service.predict_age_and_explain(
            f["aligned_pil"],
            chronological_age=chronological_age,
        )
        analyzed_faces.append(
            {
                "face_id": f["face_id"],
                "bbox": f["bbox"],
                "confidence": f["confidence"],
                "alignment_angle_deg": f.get("alignment_angle_deg", 0.0),
                "interocular_px": f.get("interocular_px"),
                "landmarks": f["landmarks"],
                "is_fallback": f["is_fallback"],
                **age_res,
            }
        )

    return {
        "image_width": pil_img.width,
        "image_height": pil_img.height,
        "original_image_b64": pil_to_base64(pil_img, fmt="JPEG"),
        "num_faces_detected": len(analyzed_faces),
        "faces": analyzed_faces,
        "status": service.get_status(),
    }


@app.post("/api/mae")
async def api_mae(req: MAERequest):
    try:
        aligned_pil = decode_base64_image(req.aligned_face_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")

    res = service.run_mae_reconstruction(
        aligned_pil=aligned_pil,
        mask_ratio=req.mask_ratio,
        seed=req.seed,
    )
    return {
        **res,
        "status": service.get_status(),
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/mobile")
def mobile_ui():
    return FileResponse(os.path.join(STATIC_DIR, "mobile.html"))

