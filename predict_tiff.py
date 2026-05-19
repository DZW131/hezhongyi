import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import tifffile
from tqdm import tqdm

from unet import UNet
from utils.checkpoint_io import load_torch_state
from utils.data_loading import BasicDataset
from utils.inference_postprocessing import (
    apply_binary_postprocessing,
    estimate_tissue_mask,
    summarize_binary_mask,
)


def preprocess_image_tensor(pil_img, scale_factor: float, device: torch.device) -> torch.Tensor:
    image_array = BasicDataset.preprocess(None, pil_img, scale_factor, is_mask=False)
    image_array = np.ascontiguousarray(image_array.copy())
    image_tensor = torch.from_numpy(image_array)
    return image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def predict_tiff(
    net,
    tiff_path,
    device,
    tile_size=1024,
    out_threshold=0.5,
    scale_factor=1.0,
    postprocess=False,
    apply_tissue_mask=False,
    white_threshold=230.0,
    black_threshold=0.0,
    fill_holes=False,
    closing_radius=0,
    opening_radius=0,
    min_component_area=0,
    max_component_area=0,
    max_component_extent=0,
    split_touching=False,
    split_min_component_area=0,
    split_min_peak_distance=32,
    split_max_markers=4,
):
    """
    Predicts a mask for a large TIFF image using a sliding window approach.
    """
    net.eval()
    
    with tifffile.TiffFile(tiff_path) as tif:
        # Load image data as a memory-mapped array or numpy array
        image_data = tif.asarray()
        if image_data.ndim == 3:
            h, w, c = image_data.shape
        else:
            h, w = image_data.shape
            c = 1
        
        logging.info(f"Image dimensions: {w}x{h}")
        
        # Initialize an empty mask for the full image
        full_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Sliding window coordinates
        # It is recommended that tile_size matches the training size (e.g., 1024)
        for y in tqdm(range(0, h, tile_size), desc="Predicting rows"):
            for x in range(0, w, tile_size):
                # Define tile boundaries
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Extract the current tile
                tile = image_data[y:y_end, x:x_end]
                
                # Handle padding for tiles smaller than tile_size at the edges
                actual_h, actual_w = tile.shape[0], tile.shape[1]
                pad_h = tile_size - actual_h
                pad_w = tile_size - actual_w
                
                if pad_h > 0 or pad_w > 0:
                    # Constant padding with zeros
                    if tile.ndim == 3:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                    else:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='constant')

                # Convert to PIL Image to reuse the existing preprocessing logic
                tile_pil = Image.fromarray(tile)
                
                # Preprocessing and inference
                img_tensor = preprocess_image_tensor(tile_pil, scale_factor, device)

                with torch.no_grad():
                    output = net(img_tensor)

                    if net.n_classes > 1:
                        probs = F.softmax(output, dim=1)[0]
                    else:
                        probs = torch.sigmoid(output)[0]

                    # Resize back to tile_size (to undo internal scaling if any)
                    full_probs = F.interpolate(probs.unsqueeze(0), size=(tile_size, tile_size), mode='bilinear')[0]

                    if net.n_classes == 2:
                        mask_tile = (full_probs[1] > out_threshold).cpu().numpy().astype(np.uint8)
                    elif net.n_classes > 2:
                        mask_tile = full_probs.argmax(dim=0).cpu().numpy()
                    else:
                        mask_tile = (full_probs[0] > out_threshold).cpu().numpy().astype(np.uint8)

                # Map the predicted tile back to the full mask (excluding padded areas)
                full_mask[y:y_end, x:x_end] = mask_tile[0:actual_h, 0:actual_w]

        if postprocess and net.n_classes <= 2:
            before_stats = summarize_binary_mask(full_mask)
            logging.info("Mask before postprocessing: %s", before_stats)
            tissue_mask = (
                estimate_tissue_mask(
                    image_data,
                    white_threshold=white_threshold,
                    black_threshold=black_threshold,
                )
                if apply_tissue_mask
                else None
            )
            full_mask = apply_binary_postprocessing(
                full_mask,
                tissue_mask=tissue_mask,
                fill_holes=fill_holes,
                closing_radius=closing_radius,
                opening_radius=opening_radius,
                min_component_area=min_component_area,
                max_component_area=max_component_area,
                max_component_extent=max_component_extent,
                split_touching=split_touching,
                split_min_component_area=split_min_component_area,
                split_min_peak_distance=split_min_peak_distance,
                split_max_markers=split_max_markers,
            )
            after_stats = summarize_binary_mask(full_mask)
            logging.info("Mask after postprocessing: %s", after_stats)

    return full_mask

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from large TIFF images')
    parser.add_argument('--model', '-m', default='checkpoints/best.pth', help='Model path')
    parser.add_argument('--input', '-i', nargs='+', required=True, help='Paths to input .tiff files')
    parser.add_argument('--output', '-o', nargs='+', help='Output filenames')
    parser.add_argument('--tile-size', '-t', type=int, default=1024, help='Size of tiles for processing')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for each tile before inference')
    parser.add_argument('--threshold', type=float, default=0.5, help='Mask threshold')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--postprocess', action='store_true', default=False,
                        help='Apply HZY glomerulus mask postprocessing after whole-image prediction')
    parser.add_argument('--apply-tissue-mask', action='store_true', default=False,
                        help='Keep predictions only inside non-white tissue regions estimated from the source TIFF')
    parser.add_argument('--white-threshold', type=float, default=230.0,
                        help='Intensity threshold used to estimate non-white tissue for postprocessing')
    parser.add_argument('--black-threshold', type=float, default=0.0,
                        help='Optional lower intensity threshold to suppress near-black mosaic background')
    parser.add_argument('--fill-holes', action='store_true', default=False,
                        help='Fill holes inside predicted glomerulus components')
    parser.add_argument('--closing-radius', type=int, default=0,
                        help='Morphological closing radius used to fill small gaps and cracks')
    parser.add_argument('--opening-radius', type=int, default=0,
                        help='Morphological opening radius used to remove tiny protrusions/noise')
    parser.add_argument('--min-component-area', type=int, default=0,
                        help='Discard connected components smaller than this many pixels')
    parser.add_argument('--max-component-area', type=int, default=0,
                        help='Discard connected components larger than this many pixels')
    parser.add_argument('--max-component-extent', type=int, default=0,
                        help='Discard connected components whose width or height exceeds this limit')
    parser.add_argument('--split-touching', action='store_true', default=False,
                        help='Split likely touching glomeruli using distance-transform marker separation')
    parser.add_argument('--split-min-component-area', type=int, default=0,
                        help='Only attempt touching split on components at least this large')
    parser.add_argument('--split-min-peak-distance', type=int, default=32,
                        help='Minimum distance-transform peak distance for touching split markers')
    parser.add_argument('--split-max-markers', type=int, default=4,
                        help='Maximum number of split markers kept per connected component')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=args.classes)
    
    logging.info(f'Loading model: {args.model}')
    logging.info(f'Using device: {device}')
    
    state_dict = load_torch_state(args.model, map_location=device)
    # Remove metadata if present in state_dict
    if 'mask_values' in state_dict:
        state_dict.pop('mask_values')
        
    net.load_state_dict(state_dict)
    net.to(device=device)
    logging.info('Model loaded successfully!')

    # Map output filenames if not provided
    in_files = args.input
    out_files = args.output or [f"{os.path.splitext(f)[0]}_MASK.png" for f in in_files]

    for i, filename in enumerate(in_files):
        logging.info(f'Processing file: {filename}...')
        
        # Perform tile-based prediction
        mask = predict_tiff(
            net,
            filename,
            device,
            tile_size=args.tile_size,
            out_threshold=args.threshold,
            scale_factor=args.scale,
            postprocess=args.postprocess,
            apply_tissue_mask=args.apply_tissue_mask,
            white_threshold=args.white_threshold,
            black_threshold=args.black_threshold,
            fill_holes=args.fill_holes,
            closing_radius=args.closing_radius,
            opening_radius=args.opening_radius,
            min_component_area=args.min_component_area,
            max_component_area=args.max_component_area,
            max_component_extent=args.max_component_extent,
            split_touching=args.split_touching,
            split_min_component_area=args.split_min_component_area,
            split_min_peak_distance=args.split_min_peak_distance,
            split_max_markers=args.split_max_markers,
        )
        
        # Save the resulting mask
        out_name = out_files[i]
        
        # Map binary mask to 0-255 for visibility if classes <= 2
        if args.classes <= 2:
            mask_img = Image.fromarray(mask * 255)
        else:
            mask_img = Image.fromarray(mask.astype(np.uint8))

        Path(out_name).parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(out_name)
        logging.info(f'Successfully saved mask to: {out_name}')
