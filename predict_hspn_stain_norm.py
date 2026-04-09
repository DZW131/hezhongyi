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
from utils.inference_postprocessing import apply_binary_postprocessing, estimate_tissue_mask


def preprocess_image_tensor(pil_img, scale_factor: float, device: torch.device) -> torch.Tensor:
    image_array = BasicDataset.preprocess(None, pil_img, scale_factor, is_mask=False)
    image_array = np.ascontiguousarray(image_array.copy())
    image_tensor = torch.from_numpy(image_array)
    return image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def color_transfer_reinhard(source_tile):
    """
    Color Transfer based on Reinhard's Method:
    Transfers the color characteristics from HuBMAP (target) to Hospital data (source).
    This helps the model 'see' the pale slides in the dark purple style it was trained on.
    """
    # Reference statistics from typical HuBMAP training images
    # These represent the 'Deep Purple/Pink' distribution
    target_mean = np.array([160.2, 125.1, 175.8]) # Target Lab-space like mean
    target_std = np.array([45.3, 55.4, 40.2])    # Target Lab-space like std

    # Current tile statistics
    source_tile = source_tile.astype(np.float32)
    s_mean = np.mean(source_tile, axis=(0, 1))
    s_std = np.std(source_tile, axis=(0, 1))

    # Perform color normalization
    # (Source - Mean) * (Target_Std / Source_Std) + Target_Mean
    norm_tile = (source_tile - s_mean) * (target_std / (s_std + 1e-6)) + target_mean
    
    # Clip values to valid RGB range
    return np.clip(norm_tile, 0, 255).astype(np.uint8)

def predict_hspn_with_stain_norm(
    net,
    tiff_path,
    device,
    tile_size=1024,
    out_threshold=0.5,
    scale_factor=1.0,
    apply_tissue_mask=False,
    white_threshold=230.0,
    min_component_area=0,
    max_component_area=0,
    max_component_extent=0,
):
    """
    Predicts masks for large TIFF using sliding window and color normalization.
    """
    net.eval()
    
    with tifffile.TiffFile(tiff_path) as tif:
        # Load WSI data
        image_data = tif.asarray()
        if image_data.ndim == 3:
            h, w, c = image_data.shape
        else:
            h, w = image_data.shape
            c = 1
        
        logging.info(f"Image Dimensions: {w}x{h}")
        full_mask = np.zeros((h, w), dtype=np.uint8)
        
        for y in tqdm(range(0, h, tile_size), desc="Processing rows"):
            for x in range(0, w, tile_size):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Extract tile
                raw_tile = image_data[y:y_end, x:x_end]
                tile = raw_tile
                
                # --- [Color Normalization Step] ---
                # This makes the pale HSPN slide look like a dark HuBMAP slide
                if tile.ndim == 3 and tile.max() > 5: # Skip near-empty/black tiles
                    tile = color_transfer_reinhard(tile)
                # ----------------------------------

                # Padding for edges
                actual_h, actual_w = tile.shape[0], tile.shape[1]
                pad_h, pad_w = tile_size - actual_h, tile_size - actual_w
                
                if pad_h > 0 or pad_w > 0:
                    if tile.ndim == 3:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                    else:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='constant')

                # Standard Preprocessing
                tile_pil = Image.fromarray(tile)
                img_tensor = preprocess_image_tensor(tile_pil, scale_factor, device)

                with torch.no_grad():
                    output = net(img_tensor)
                    
                    # Handle multi-class vs binary output
                    if net.n_classes > 1:
                        probs = F.softmax(output, dim=1)[0]
                        # Use index 1 as the glomeruli class in a 2-class setup
                        target_prob = probs[1] 
                    else:
                        probs = torch.sigmoid(output)[0]
                        target_prob = probs[0]
                    
                    # Ensure mask size matches tile size
                    full_probs = F.interpolate(target_prob.unsqueeze(0).unsqueeze(0), 
                                               size=(tile_size, tile_size), 
                                               mode='bilinear')[0][0]
                    
                    mask_tile = (full_probs > out_threshold).cpu().numpy().astype(np.uint8)

                if net.n_classes <= 2:
                    tissue_mask = None
                    if apply_tissue_mask:
                        tissue_mask = estimate_tissue_mask(raw_tile, white_threshold=white_threshold)
                    mask_tile = apply_binary_postprocessing(
                        mask_tile,
                        tissue_mask=tissue_mask,
                        min_component_area=min_component_area,
                        max_component_area=max_component_area,
                        max_component_extent=max_component_extent,
                    )

                # Stitch back to full mask
                full_mask[y:y_end, x:x_end] = mask_tile[0:actual_h, 0:actual_w]

    return full_mask

def get_args():
    parser = argparse.ArgumentParser(description='Inference with Reinhard Stain Normalization for HSPN data')
    parser.add_argument('--model', '-m', default='checkpoints/best.pth', help='Path to weights')
    parser.add_argument('--input', '-i', nargs='+', required=True, help='Input TIFF file path')
    parser.add_argument('--output', '-o', nargs='+', help='Output filename')
    parser.add_argument('--tile-size', '-t', type=int, default=1024, help='Tile size')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for each tile before inference')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes in the checkpoint')
    parser.add_argument('--apply-tissue-mask', action='store_true', default=False,
                        help='Keep predictions only inside non-white tissue regions estimated from the raw tile')
    parser.add_argument('--white-threshold', type=float, default=230.0,
                        help='Intensity threshold used to estimate non-white tissue for postprocessing')
    parser.add_argument('--min-component-area', type=int, default=0,
                        help='Discard connected components smaller than this many pixels')
    parser.add_argument('--max-component-area', type=int, default=0,
                        help='Discard connected components larger than this many pixels')
    parser.add_argument('--max-component-extent', type=int, default=0,
                        help='Discard connected components whose width or height exceeds this limit')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=args.classes)
    
    logging.info(f'Loading checkpoint: {args.model}')
    state_dict = load_torch_state(args.model, map_location=device)
    if 'mask_values' in state_dict:
        state_dict.pop('mask_values')
    net.load_state_dict(state_dict)
    net.to(device=device)
    logging.info('Model loaded successfully!')

    in_files = args.input
    out_files = args.output or [f"{os.path.splitext(f)[0]}_STAIN_NORM.png" for f in in_files]

    for i, filename in enumerate(in_files):
        logging.info(f'Normalizing and Predicting: {filename}...')
        mask = predict_hspn_with_stain_norm(net, filename, device, 
                                            tile_size=args.tile_size, 
                                            out_threshold=args.threshold,
                                            scale_factor=args.scale,
                                            apply_tissue_mask=args.apply_tissue_mask,
                                            white_threshold=args.white_threshold,
                                            min_component_area=args.min_component_area,
                                            max_component_area=args.max_component_area,
                                            max_component_extent=args.max_component_extent)
        
        # Save as PNG
        mask_img = Image.fromarray(mask * 255)
        Path(out_files[i]).parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(out_files[i])
        logging.info(f'Inference finished. Saved to: {out_files[i]}')
