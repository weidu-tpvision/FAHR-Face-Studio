import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from model_service import FAHRFaceService, CHECKPOINTS_DIR, BASE_DIR
from face_pipeline import detect_and_align_faces, ensure_yunet_model


class FAHRFaceAgeOnnxWrapper(nn.Module):
    """
    Self-contained ONNX wrapper for FAHR-FaceAge:
    Input:
      rgb_01: FloatTensor [1, 3, 112, 112] scaled in [0.0, 1.0]
    Outputs:
      predicted_age: FloatTensor [1]
      attention_grid_7x7: FloatTensor [7, 7] normalized in [0.0, 1.0]
    """

    def __init__(self, age_model: nn.Module) -> None:
        super().__init__()
        self.model = age_model
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer("eye50", torch.eye(50, dtype=torch.float32))

    def forward(self, rgb_01: torch.Tensor):
        pixel_values = (rgb_01 - self.mean) / self.std
        outputs = self.model(pixel_values, output_attentions=True, return_dict=True)
        predicted_age = outputs.age_output.view(1)

        rollout = self.eye50
        for layer_attn in outputs.attentions:
            attn_heads_mean = layer_attn[0].mean(dim=0)
            aug_attn = 0.5 * attn_heads_mean + 0.5 * self.eye50
            aug_attn = aug_attn / aug_attn.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(aug_attn, rollout)

        cls_patch = rollout[0, 1:].reshape(7, 7)
        min_v = cls_patch.min()
        max_v = cls_patch.max()
        grid_7x7 = (cls_patch - min_v) / (max_v - min_v + 1e-6)
        return predicted_age, grid_7x7


class FAHRFaceMAEOnnxWrapper(nn.Module):
    """
    Self-contained ONNX wrapper for FAHR-Face Foundation MAE (75% masking):
    Inputs:
      rgb_01: FloatTensor [1, 3, 112, 112] scaled in [0.0, 1.0]
      noise: FloatTensor [1, 49] random uniform noise in [0.0, 1.0]
    Outputs:
      recon_rgb_01: FloatTensor [1, 3, 112, 112] reconstructed image in [0.0, 1.0]
      mask_pixel_01: FloatTensor [1, 3, 112, 112] binary patch mask in {0.0, 1.0}
      loss: FloatTensor [1] MSE reconstruction loss on masked patches
    """

    def __init__(self, mae_model: nn.Module) -> None:
        super().__init__()
        self.model = mae_model
        self.model.config.mask_ratio = 0.75
        self.model.vit.embeddings.config.mask_ratio = 0.75
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, rgb_01: torch.Tensor, noise: torch.Tensor):
        pixel_values = (rgb_01 - self.mean) / self.std
        outputs = self.model(pixel_values=pixel_values, noise=noise)
        recon_norm = self.model.unpatchify(outputs.logits)
        recon_rgb_01 = torch.clamp(recon_norm * self.std + self.mean, 0.0, 1.0)

        mask_patches = outputs.mask.unsqueeze(-1).repeat(1, 1, 16 * 16 * 3)
        mask_pixel_01 = self.model.unpatchify(mask_patches)
        loss = outputs.loss.view(1)
        return recon_rgb_01, mask_pixel_01, loss


def main() -> None:
    svc = FAHRFaceService()
    age_model = svc.ensure_age_model().cpu().eval()
    mae_model = svc.ensure_mae_model().cpu().eval()

    onnx_dir = os.path.join(CHECKPOINTS_DIR, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    dummy_rgb = torch.rand(1, 3, 112, 112, dtype=torch.float32)
    dummy_noise = torch.rand(1, 49, dtype=torch.float32)

    # 1. Export & Quantize FAHR-FaceAge
    age_wrapper = FAHRFaceAgeOnnxWrapper(age_model).eval()
    age_fp32_path = os.path.join(onnx_dir, "fahr_faceage_fp32.onnx")
    age_quant_path = os.path.join(onnx_dir, "fahr_faceage_int8.onnx")

    print("Exporting FAHR-FaceAge to ONNX...")
    torch.onnx.export(
        age_wrapper,
        (dummy_rgb,),
        age_fp32_path,
        input_names=["rgb_01"],
        output_names=["predicted_age", "attention_grid_7x7"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("Quantizing FAHR-FaceAge (INT8/UINT8 dynamic quantization)...")
    quantize_dynamic(
        model_input=age_fp32_path,
        model_output=age_quant_path,
        weight_type=QuantType.QUInt8,
    )

    # 2. Export & Quantize FAHR-Face MAE
    mae_wrapper = FAHRFaceMAEOnnxWrapper(mae_model).eval()
    mae_fp32_path = os.path.join(onnx_dir, "fahr_face_mae_fp32.onnx")
    mae_quant_path = os.path.join(onnx_dir, "fahr_face_mae_int8.onnx")

    print("Exporting FAHR-Face MAE to ONNX...")
    torch.onnx.export(
        mae_wrapper,
        (dummy_rgb, dummy_noise),
        mae_fp32_path,
        input_names=["rgb_01", "noise"],
        output_names=["recon_rgb_01", "mask_pixel_01", "loss"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("Quantizing FAHR-Face MAE (INT8/UINT8 dynamic quantization)...")
    quantize_dynamic(
        model_input=mae_fp32_path,
        model_output=mae_quant_path,
        weight_type=QuantType.QUInt8,
    )

    # 3. Verify Numerical Accuracy on static/sample_face.jpg
    sample_img = Image.open("static/sample_face.jpg")
    faces = detect_and_align_faces(sample_img)
    aligned_pil = faces[0]["aligned_pil"].convert("RGB").resize((112, 112), Image.Resampling.BILINEAR)
    rgb_np = (np.array(aligned_pil, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

    with torch.inference_mode():
        pt_age, pt_grid = age_wrapper(torch.from_numpy(rgb_np))

    sess_age = ort.InferenceSession(age_quant_path, providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    ort_age, ort_grid = sess_age.run(None, {"rgb_01": rgb_np})
    age_ms = (time.perf_counter() - t0) * 1000

    sess_mae = ort.InferenceSession(mae_quant_path, providers=["CPUExecutionProvider"])
    noise_np = np.random.RandomState(42).rand(1, 49).astype(np.float32)
    t0 = time.perf_counter()
    ort_recon, ort_mask, ort_loss = sess_mae.run(None, {"rgb_01": rgb_np, "noise": noise_np})
    mae_ms = (time.perf_counter() - t0) * 1000

    age_mb = os.path.getsize(age_quant_path) / (1024 * 1024)
    mae_mb = os.path.getsize(mae_quant_path) / (1024 * 1024)

    # 4. Copy quantized ONNX models + YuNet + sample portrait into Android assets
    android_assets = os.path.join(BASE_DIR, "mobile", "app", "src", "main", "assets")
    os.makedirs(android_assets, exist_ok=True)
    yunet_path = ensure_yunet_model()
    shutil.copy2(yunet_path, os.path.join(android_assets, "face_detection_yunet_2023mar.onnx"))
    shutil.copy2(age_quant_path, os.path.join(android_assets, "fahr_faceage_int8.onnx"))
    shutil.copy2(mae_quant_path, os.path.join(android_assets, "fahr_face_mae_int8.onnx"))
    shutil.copy2(
        os.path.join(BASE_DIR, "static", "sample_face.jpg"),
        os.path.join(android_assets, "sample_face.jpg"),
    )

    # Remove intermediate unquantized FP32 ONNX files to save disk space
    for tmp_f in (age_fp32_path, mae_fp32_path):
        if os.path.exists(tmp_f):
            os.remove(tmp_f)

    print(f"=== ONNX Quantized Export Summary ===")
    print(f"fahr_faceage_int8.onnx : {age_mb:.1f} MB | PyTorch Age={pt_age.item():.2f}, ONNX INT8 Age={ort_age[0]:.2f} ({age_ms:.1f} ms)")
    print(f"fahr_face_mae_int8.onnx: {mae_mb:.1f} MB | MAE Loss={ort_loss[0]:.4f}, Masked Patches={int(ort_mask.sum() / (16*16*3))} ({mae_ms:.1f} ms)")
    print(f"Bundled assets copied to: {android_assets}")


if __name__ == "__main__":
    main()
