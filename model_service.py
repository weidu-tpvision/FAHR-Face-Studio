import os
import io
import base64
import time
import threading
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import ViTConfig, ViTMAEConfig, ViTMAEForPreTraining

from ViTForAgeEstimation import ViTForAgeRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
HF_REPO_ID = "fridolin22/FAHR-Face"
AGE_CHECKPOINT_FILE = "fahr_faceage.pth"
MAE_CHECKPOINT_FILE = "fahr_face.pth"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def adapt_vit_state_dict(
    state_dict: Dict[str, torch.Tensor], target_model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Adapts checkpoints saved in transformers 4.x (vit.encoder.layer.X...) to
    transformers 5.x (vit.layers.X...) if needed, preserving strict=True validation.
    """
    target_keys = set(target_model.state_dict().keys())
    if set(state_dict.keys()) == target_keys:
        return state_dict

    remapped: Dict[str, torch.Tensor] = {}
    replacements = [
        (".encoder.layer.", ".layers."),
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".intermediate.dense.", ".mlp.fc1."),
        (".output.dense.", ".mlp.fc2."),
    ]
    for k, v in state_dict.items():
        if k in target_keys:
            remapped[k] = v
            continue
        new_k = k
        for old_sub, new_sub in replacements:
            new_k = new_k.replace(old_sub, new_sub)
        remapped[new_k] = v
    return remapped


class FAHRFaceService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.age_model: Optional[ViTForAgeRegression] = None
        self.mae_model: Optional[ViTMAEForPreTraining] = None
        self._lock = threading.Lock()
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    def get_status(self) -> Dict[str, Any]:
        age_local = os.path.join(CHECKPOINTS_DIR, AGE_CHECKPOINT_FILE)
        mae_local = os.path.join(CHECKPOINTS_DIR, MAE_CHECKPOINT_FILE)
        return {
            "device": str(self.device),
            "age_model_loaded": self.age_model is not None,
            "mae_model_loaded": self.mae_model is not None,
            "age_checkpoint_cached": os.path.exists(age_local),
            "mae_checkpoint_cached": os.path.exists(mae_local),
        }

    def ensure_age_model(self) -> ViTForAgeRegression:
        if self.age_model is not None:
            return self.age_model

        with self._lock:
            if self.age_model is not None:
                return self.age_model

            ckpt_path = os.path.join(CHECKPOINTS_DIR, AGE_CHECKPOINT_FILE)
            if not os.path.exists(ckpt_path):
                ckpt_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=AGE_CHECKPOINT_FILE,
                    local_dir=CHECKPOINTS_DIR,
                )

            config = ViTConfig(
                image_size=112,
                num_hidden_layers=12,
                num_attention_heads=12,
                attn_implementation="eager",
            )
            model = ViTForAgeRegression(config)
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            state_dict = adapt_vit_state_dict(
                checkpoint.get("model_state_dict", checkpoint), model
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()
            self.age_model = model
            return self.age_model

    def ensure_mae_model(self) -> ViTMAEForPreTraining:
        if self.mae_model is not None:
            return self.mae_model

        with self._lock:
            if self.mae_model is not None:
                return self.mae_model

            ckpt_path = os.path.join(CHECKPOINTS_DIR, MAE_CHECKPOINT_FILE)
            if not os.path.exists(ckpt_path):
                ckpt_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=MAE_CHECKPOINT_FILE,
                    local_dir=CHECKPOINTS_DIR,
                )

            config = ViTMAEConfig(image_size=112)
            model = ViTMAEForPreTraining(config)
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            state_dict = adapt_vit_state_dict(
                checkpoint.get("model_state_dict", checkpoint), model
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()
            self.mae_model = model
            return self.mae_model

    def preprocess_aligned_face(
        self, aligned_pil: Image.Image
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Resizes aligned face to 112x112 RGB and applies ImageNet normalization."""
        face_112 = aligned_pil.convert("RGB").resize(
            (112, 112), Image.Resampling.BILINEAR
        )
        arr = np.array(face_112, dtype=np.float32) / 255.0
        norm_arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = (
            torch.from_numpy(norm_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        return face_112, tensor

    @staticmethod
    def compute_attention_rollout(attentions: Tuple[torch.Tensor, ...]) -> np.ndarray:
        """
        Computes Vision Transformer Attention Rollout across all 12 layers and 12 heads
        from the CLS token to the 7x7 (49) image patches.
        """
        seq_len = attentions[0].shape[-1]
        eye = torch.eye(seq_len, device=attentions[0].device)
        rollout = eye.clone()

        for layer_attn in attentions:
            attn_heads_mean = layer_attn[0].mean(dim=0)
            aug_attn = 0.5 * attn_heads_mean + 0.5 * eye
            aug_attn = aug_attn / aug_attn.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(aug_attn, rollout)

        cls_patch_attn = rollout[0, 1:].detach().cpu().numpy()
        grid_7x7 = cls_patch_attn.reshape(7, 7)
        min_v, max_v = float(grid_7x7.min()), float(grid_7x7.max())
        if max_v > min_v:
            grid_7x7 = (grid_7x7 - min_v) / (max_v - min_v)
        return grid_7x7

    @staticmethod
    def compute_regional_attribution(grid_7x7: np.ndarray) -> Dict[str, float]:
        """
        Summarizes the 7x7 ViT patch attention grid into intuitive anatomical facial zones.
        """
        forehead = float(grid_7x7[0:2, 1:6].mean())
        eyes = float(grid_7x7[2:4, 1:6].mean())
        midface = float(grid_7x7[3:5, 1:6].mean())
        lowerface = float(grid_7x7[5:7, 1:6].mean())
        total = forehead + eyes + midface + lowerface + 1e-8
        return {
            "Forehead & Glabella": round((forehead / total) * 100, 1),
            "Periorbital & Eyes": round((eyes / total) * 100, 1),
            "Cheeks & Nasolabial": round((midface / total) * 100, 1),
            "Mouth, Chin & Jaw": round((lowerface / total) * 100, 1),
        }

    @staticmethod
    def render_attention_overlay(
        face_112: Image.Image, grid_7x7: np.ndarray, size: int = 224
    ) -> Tuple[Image.Image, Image.Image]:
        face_up = np.array(face_112.resize((size, size), Image.Resampling.BICUBIC))
        heatmap_up = cv2.resize(
            grid_7x7.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC
        )
        heatmap_up = np.clip(heatmap_up, 0.0, 1.0)
        heatmap_u8 = np.uint8(255 * heatmap_up)

        colored_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_TURBO)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        alpha = (0.25 + 0.45 * heatmap_up)[..., None]
        blended = np.clip(
            face_up * (1.0 - alpha) + colored_rgb * alpha, 0, 255
        ).astype(np.uint8)
        return Image.fromarray(colored_rgb), Image.fromarray(blended)

    def predict_age_and_explain(
        self,
        aligned_pil: Image.Image,
        chronological_age: Optional[float] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        model = self.ensure_age_model()
        face_112, pixel_tensor = self.preprocess_aligned_face(aligned_pil)

        with torch.inference_mode():
            outputs = model(pixel_tensor, output_attentions=True, return_dict=True)
            predicted_age = float(outputs.age_output.item())
            grid_7x7 = self.compute_attention_rollout(outputs.attentions)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        heatmap_pil, overlay_pil = self.render_attention_overlay(
            face_112, grid_7x7, size=224
        )
        face_display = face_112.resize((224, 224), Image.Resampling.BICUBIC)
        regions = self.compute_regional_attribution(grid_7x7)

        age_gap = None
        age_gap_status = None
        if chronological_age is not None and chronological_age > 0:
            age_gap = round(predicted_age - float(chronological_age), 2)
            if age_gap <= -2.5:
                age_gap_status = (
                    "Decelerated Facial Aging (Younger Biological Appearance)"
                )
            elif age_gap >= 2.5:
                age_gap_status = (
                    "Accelerated Facial Aging (Older Biological Appearance)"
                )
            else:
                age_gap_status = "Concordant Biological & Chronological Age"

        return {
            "predicted_age": round(predicted_age, 2),
            "chronological_age": chronological_age,
            "age_gap": age_gap,
            "age_gap_status": age_gap_status,
            "inference_ms": elapsed_ms,
            "device": str(self.device),
            "regional_attribution": regions,
            "attention_grid_7x7": np.round(grid_7x7, 3).tolist(),
            "aligned_face_b64": pil_to_base64(face_display),
            "attention_heatmap_b64": pil_to_base64(heatmap_pil),
            "attention_overlay_b64": pil_to_base64(overlay_pil),
        }

    def run_mae_reconstruction(
        self,
        aligned_pil: Image.Image,
        mask_ratio: float = 0.75,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs the FAHR-Face Foundation Model (ViTMAEForPreTraining) on a 112x112 aligned face:
        Masks `mask_ratio` (default 75% = 36 of 49 patches) and reconstructs raw pixels.
        """
        t0 = time.perf_counter()
        model = self.ensure_mae_model()
        clamped_ratio = float(np.clip(mask_ratio, 0.1, 0.9))
        model.config.mask_ratio = clamped_ratio
        model.vit.embeddings.config.mask_ratio = clamped_ratio

        face_112, pixel_tensor = self.preprocess_aligned_face(aligned_pil)

        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(int(seed))
            with torch.inference_mode():
                outputs = model(pixel_values=pixel_tensor)
                recon_norm = model.unpatchify(outputs.logits)
                mask_patches = outputs.mask.unsqueeze(-1).repeat(1, 1, (16**2) * 3)
                mask_pixel = model.unpatchify(mask_patches)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        orig_np = np.array(face_112, dtype=np.float32) / 255.0
        recon_np = recon_norm[0].permute(1, 2, 0).detach().cpu().numpy()
        recon_np = np.clip(recon_np * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)

        mask_np = mask_pixel[0].permute(1, 2, 0).detach().cpu().numpy()

        masked_vis = orig_np * (1.0 - mask_np) + 0.14 * mask_np
        masked_u8 = np.clip(masked_vis * 255.0, 0, 255).astype(np.uint8)
        for g in range(16, 112, 16):
            masked_u8[g : g + 1, :, :] = np.clip(
                masked_u8[g : g + 1, :, :].astype(np.int32) + 35, 0, 255
            ).astype(np.uint8)
            masked_u8[:, g : g + 1, :] = np.clip(
                masked_u8[:, g : g + 1, :].astype(np.int32) + 35, 0, 255
            ).astype(np.uint8)

        composite_np = orig_np * (1.0 - mask_np) + recon_np * mask_np

        num_masked = int(outputs.mask[0].sum().item())
        total_patches = int(outputs.mask.shape[1])

        def to_upscaled_b64(arr_01: np.ndarray) -> str:
            u8 = np.clip(arr_01 * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(u8).resize((224, 224), Image.Resampling.BICUBIC)
            return pil_to_base64(pil_img)

        masked_pil = Image.fromarray(masked_u8).resize(
            (224, 224), Image.Resampling.NEAREST
        )

        return {
            "mask_ratio": round(num_masked / total_patches, 2),
            "num_masked_patches": num_masked,
            "num_visible_patches": total_patches - num_masked,
            "total_patches": total_patches,
            "loss": round(float(outputs.loss.item()), 4),
            "inference_ms": elapsed_ms,
            "original_b64": pil_to_base64(
                face_112.resize((224, 224), Image.Resampling.BICUBIC)
            ),
            "masked_input_b64": pil_to_base64(masked_pil),
            "reconstructed_composite_b64": to_upscaled_b64(composite_np),
            "reconstructed_full_b64": to_upscaled_b64(recon_np),
        }
