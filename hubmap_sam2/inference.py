from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from hubmap_sam2.prompts import PromptRecord, instance_prompts_from_instance_map, prompts_from_binary_mask
from training.utils.checkpoint_utils import (
    load_checkpoint_and_apply_kernels,
    load_state_dict_into_model,
)
from training.utils.train_utils import register_omegaconf_resolvers


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_finetuned_model(
    config_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
) -> torch.nn.Module:
    try:
        register_omegaconf_resolvers()
    except Exception:
        # Resolvers may already be registered in the current process.
        pass
    torch_device = resolve_device(device)

    cfg = compose(config_name=config_path)
    OmegaConf.resolve(cfg)

    if "trainer" in cfg and "model" in cfg.trainer:
        model = instantiate(cfg.trainer.model, _recursive_=True)
        state_dict = load_checkpoint_and_apply_kernels(
            checkpoint_path=checkpoint_path,
            ckpt_state_dict_keys=("model",),
            map_location="cpu",
        )
        load_state_dict_into_model(
            state_dict=state_dict,
            model=model,
            strict=False,
        )
        model = model.to(torch_device)
        model.eval()
        return model

    return build_sam2(config_path, checkpoint_path, device=torch_device, mode="eval")


def predict_instance_masks(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    prompts: Sequence[PromptRecord],
    prompt_mode: str = "point_box",
    multimask_output: bool = False,
) -> List[Tuple[np.ndarray, float, PromptRecord]]:
    predictor.set_image(image)
    predictions: List[Tuple[np.ndarray, float, PromptRecord]] = []
    for prompt in prompts:
        point_coords = None
        point_labels = None
        box = None

        if prompt_mode in {"point", "point_box"} and prompt.point is not None:
            point_coords = np.asarray([prompt.point], dtype=np.float32)
            point_labels = np.asarray([1], dtype=np.int32)
        if prompt_mode in {"box", "point_box"} and prompt.box is not None:
            box = np.asarray(prompt.box, dtype=np.float32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
            return_logits=False,
            normalize_coords=True,
        )
        best_index = 0 if masks.ndim == 2 else int(np.argmax(scores))
        best_mask = masks if masks.ndim == 2 else masks[best_index]
        best_score = float(scores[0] if np.ndim(scores) == 0 else scores[best_index])
        predictions.append((best_mask.astype(bool), best_score, prompt))
    predictor.reset_predictor()
    return predictions


def oracle_prompts_from_annotation(instance_map: np.ndarray, prompt_mode: str = "point_box") -> List[PromptRecord]:
    prompts = instance_prompts_from_instance_map(instance_map)
    if prompt_mode == "point":
        return [PromptRecord(p.object_id, p.point, None, p.area, p.mask) for p in prompts]
    if prompt_mode == "box":
        return [PromptRecord(p.object_id, None, p.box, p.area, p.mask) for p in prompts]
    return prompts


def prompts_from_prior(prior_mask: np.ndarray, min_component_area: int = 32) -> List[PromptRecord]:
    return prompts_from_binary_mask(prior_mask, min_component_area=min_component_area)


def predictions_to_instance_map(
    predictions: Sequence[Tuple[np.ndarray, float, PromptRecord]],
    image_shape: Tuple[int, int],
    min_mask_area: int = 16,
) -> np.ndarray:
    instance_map = np.zeros(image_shape, dtype=np.int32)
    occupied = np.zeros(image_shape, dtype=bool)
    next_id = 1
    sorted_predictions = sorted(predictions, key=lambda item: float(item[1]), reverse=True)
    for mask, _, _ in sorted_predictions:
        binary_mask = np.asarray(mask).astype(bool)
        if int(binary_mask.sum()) < min_mask_area:
            continue
        assign_mask = np.logical_and(binary_mask, np.logical_not(occupied))
        if int(assign_mask.sum()) < min_mask_area:
            continue
        instance_map[assign_mask] = next_id
        occupied[assign_mask] = True
        next_id += 1
    return instance_map


def automatic_mask_generation(
    model: torch.nn.Module,
    image: np.ndarray,
    points_per_side: int = 24,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.75,
    stability_score_thresh: float = 0.9,
    min_mask_region_area: int = 32,
    crop_n_layers: int = 0,
) -> List[Dict[str, object]]:
    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        output_mode="binary_mask",
        crop_n_layers=crop_n_layers,
        multimask_output=False,
    )
    return generator.generate(image)


def anns_to_instance_map(
    anns: Sequence[Dict[str, object]],
    image_shape: Tuple[int, int],
    min_mask_area: int = 16,
) -> np.ndarray:
    instance_map = np.zeros(image_shape, dtype=np.int32)
    occupied = np.zeros(image_shape, dtype=bool)
    next_id = 1
    sorted_anns = sorted(
        anns,
        key=lambda ann: (
            float(ann.get("predicted_iou", 0.0)),
            float(ann.get("stability_score", 0.0)),
            -float(ann.get("area", 0.0)),
        ),
        reverse=True,
    )
    for ann in sorted_anns:
        segmentation = np.asarray(ann["segmentation"]).astype(bool)
        if int(segmentation.sum()) < min_mask_area:
            continue
        assign_mask = np.logical_and(segmentation, np.logical_not(occupied))
        if int(assign_mask.sum()) < min_mask_area:
            continue
        instance_map[assign_mask] = next_id
        occupied[assign_mask] = True
        next_id += 1
    return instance_map
