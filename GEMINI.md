# GEMINI.md — Developer & AI Agent Context Guide

This file documents the architecture, technical conventions, and operational details of **FAHR-Face Studio** for developers and AI coding agents.

---

## 1. Project Purpose

**FAHR-Face Studio** is an interactive web application and FastAPI server built around the **FAHR-Face** (*Foundation Artificial Intelligence for Health Recognition using Face Photographs*) models from AIM-Harvard (*The Lancet Digital Health*, 2026; Hugging Face: `fridolin22/FAHR-Face`).

It provides:
1. **Multi-face detection and 5-point eye-horizontal alignment** (`face_pipeline.py`).
2. **Biological age estimation (`FAHR-FaceAge` / `fahr_faceage.pth`)**, **Age Gap ($\Delta\text{Age}$)** calculation, and **12-layer $\times$ 12-head ViT Self-Attention Rollout** explainability (`model_service.py`).
3. **Foundation Model Masked Autoencoder (`FAHR-Face` / `fahr_face.pth`)** patch masking and pixel reconstruction (`model_service.py`).
4. **Interactive Single-Page Desktop Web UI** (`static/index.html`) and **Pixel 9 3-Tab Mobile Web UI** (`static/mobile.html`) served by FastAPI (`app.py`).
5. **Native Android App for Google Pixel 9** (`mobile/` — Kotlin + Jetpack Compose Material 3) with a concise 3-Tab Bottom Navigation (`Scan & Age`, `AI Heatmap`, `MAE Rebuild`).

---

## 2. Architecture & Key Technical Decisions

### A. Zero-TensorFlow Face Detection & Alignment (`face_pipeline.py`)
- The upstream `AIM-Harvard/FAHR-Face` repository used `retina-face==0.0.17` on Python 3.8, which pulls in TensorFlow/Keras alongside PyTorch.
- To keep this project 100% PyTorch-native and compatible with Python 3.10–3.14, `face_pipeline.py` uses OpenCV's lightweight **YuNet** 5-point landmark detector (`cv2.FaceDetectorYN` with `face_detection_yunet_2023mar.onnx`, 230 KB, auto-downloaded to `./checkpoints/`).
- **Eye-Horizontal Alignment (`align_and_crop_face`)**:
  - Computes the angle $\theta = \text{atan2}(\Delta y, \Delta x)$ between the subject's right and left eye landmarks.
  - Applies an affine rotation (`cv2.getRotationMatrix2D` + `cv2.warpAffine`) around the inter-ocular midpoint so the eyes are horizontal, then crops the transformed face bounding box.
  - Exposes `alignment_angle_deg`, `interocular_px`, `bbox`, and 5-point `landmarks` in the API response.

### B. `transformers 4.x` vs `5.x` Checkpoint Compatibility (`model_service.py`)
- The upstream checkpoints (`fahr_faceage.pth` and `fahr_face.pth`) were saved using `transformers 4.38.2`, where `ViTModel` layer keys are named:
  - `vit.encoder.layer.{i}.attention.attention.{query,key,value}.{weight,bias}`
  - `vit.encoder.layer.{i}.attention.output.dense.{weight,bias}`
  - `vit.encoder.layer.{i}.intermediate.dense.{weight,bias}`
  - `vit.encoder.layer.{i}.output.dense.{weight,bias}`
- In `transformers 5.x`, `ViTModel` renamed these submodules to `.layers.{i}.attention.{q,k,v,o}_proj` and `.layers.{i}.mlp.{fc1,fc2}`.
- `adapt_vit_state_dict()` in `model_service.py` automatically detects which naming convention the installed `transformers` version expects and remaps the keys 1-to-1 before calling `model.load_state_dict(state_dict, strict=True)`.

### C. Eager Attention for ViT Attention Rollout (`model_service.py`)
- `ViTConfig` is initialized with `attn_implementation="eager"` (`image_size=112, num_hidden_layers=12, num_attention_heads=12`).
- Without `attn_implementation="eager"`, newer versions of `transformers` default to `sdpa` (Scaled Dot-Product Attention), which returns `None` when `output_attentions=True` is requested.
- `compute_attention_rollout()` averages attention across the 12 heads at each layer, adds the identity matrix $I_{50}$ for residual connections ($0.5 A_l + 0.5 I$), row-normalizes, multiplies across all 12 layers, and extracts `rollout[0, 1:]` (the `CLS` token's attention to the $7\times 7 = 49$ image patches).

### D. Input Preprocessing Contract
- Both `FAHR-FaceAge` and `FAHR-Face MAE` strictly require:
  - Eye-aligned, tightly cropped RGB face image resized to **`112 × 112`** (`Image.Resampling.BILINEAR`).
  - Pixel values scaled to `[0, 1]` and normalized with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### E. Standalone On-Device Android AI Engine (`export_onnx.py` & `OnDeviceAiEngine.kt`)
- `export_onnx.py` wraps `ViTForAgeRegression` (including the 12-layer Attention Rollout matrix multiplication) and `ViTMAEForPreTraining` (including patch unpatchifying and MSE loss) into self-contained ONNX graphs (`opset=17`) and applies `UINT8` dynamic quantization (`fahr_faceage_int8.onnx` = `82.4 MB`, `fahr_face_mae_int8.onnx` = `107.7 MB`).
- On Android (`mobile/app/src/main/java/com/fahrface/mobile/OnDeviceAiEngine.kt`), the `.onnx` assets are stream-copied once to `context.filesDir` so `onnxruntime-android` can memory-map (`mmap`) them directly from flash storage without consuming JVM heap memory.
- `OnDeviceAiEngine.kt` executes `face_detection_yunet_2023mar.onnx` (`640×640` multi-stride anchor decoding + NMS), performs eye-horizontal affine alignment via `android.graphics.Matrix`, runs `fahr_faceage_int8.onnx` and `fahr_face_mae_int8.onnx`, and renders the `224×224` Turbo colormap attention heatmap 100% offline on the Pixel 9.

---

## 3. Development & Common Commands

Always use the local `.venv` environment (never global `pip`):

```powershell
# Run the FastAPI server
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000

# Pre-warm models into RAM and check status
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/status?warmup=true"

# Quick CLI smoke test on the sample portrait
.\.venv\Scripts\python.exe -c "from PIL import Image; from face_pipeline import detect_and_align_faces; from model_service import FAHRFaceService; img = Image.open('static/sample_face.jpg'); faces = detect_and_align_faces(img); svc = FAHRFaceService(); print(svc.predict_age_and_explain(faces[0]['aligned_pil'], 42)['predicted_age'])"
```

### Updating the UI Screenshot (`docs/ui-screenshot.png`)
The frontend supports `?autorun=1`, which automatically loads `/static/sample_face.jpg`, sets chronological age to `42`, runs `/api/analyze`, and triggers `/api/mae` with `seed=42`. To regenerate `docs/ui-screenshot.png` after UI changes:
1. Ensure the server is running and warmed up (`GET /api/status?warmup=true`).
2. Run headless Chrome with `--window-size=1480,1280` and `--virtual-time-budget=15000` pointing to `http://127.0.0.1:8000/?autorun=1`.
