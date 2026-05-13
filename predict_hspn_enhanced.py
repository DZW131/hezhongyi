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


def predict_hspn_tiles(
    net,
    tiff_path,
    device,
    tile_size=1024,
    out_threshold=0.5,
    scale_factor=1.0,
    apply_tissue_mask=False,
    white_threshold=230.0,
    fill_holes=False,
    closing_radius=0,
    min_component_area=0,
    max_component_area=0,
    max_component_extent=0,
):
    """
    Predicts a mask for a large TIFF image using a sliding window approach 
    with Contrast Stretching preprocessing to handle domain shift (pale staining).
    """
    net.eval()
    
    with tifffile.TiffFile(tiff_path) as tif:
        # Load the TIFF image data
        image_data = tif.asarray()
        if image_data.ndim == 3:
            h, w, c = image_data.shape
        else:
            h, w = image_data.shape
            c = 1
        
        logging.info(f"Full image dimensions: {w}x{h}")
        
        # Initialize an empty mask for the entire WSI (Whole Slide Image)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Iterate over tiles using a sliding window
        for y in tqdm(range(0, h, tile_size), desc="Processing rows"):
            for x in range(0, w, tile_size):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Extract the current tile from the large image
                raw_tile = image_data[y:y_end, x:x_end]
                tile = raw_tile
                
                # --- [Preprocessing: Linear Contrast Stretching] ---
                # Purpose: Normalize pale hospital slides to match HuBMAP distribution.
                # It maps the 2nd and 98th percentiles to 0 and 255.
                if tile.ndim == 3 and tile.max() > 0:
                    p_low, p_high = np.percentile(tile, (2, 98))
                    tile = np.clip(tile, p_low, p_high)
                    # Avoid division by zero and rescale to 0-255
                    tile = ((tile - p_low) / (max(p_high - p_low, 1)) * 255).astype(np.uint8)
                # ----------------------------------------------------

                # Handle boundary tiles that are smaller than tile_size
                actual_h, actual_w = tile.shape[0], tile.shape[1]
                pad_h = tile_size - actual_h
                pad_w = tile_size - actual_w
                
                if pad_h > 0 or pad_w > 0:
                    if tile.ndim == 3:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                    else:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='constant')

                # Convert to PIL Image to utilize standard preprocessing pipeline
                tile_pil = Image.fromarray(tile)
                img_tensor = preprocess_image_tensor(tile_pil, scale_factor, device)

                with torch.no_grad():
                    output = net(img_tensor)
                    
                    # Compute probabilities based on number of classes
                    if net.n_classes > 1:
                        probs = F.softmax(output, dim=1)[0]
                        if net.n_classes == 2:
                            target_prob = probs[1]
                            full_probs = F.interpolate(
                                target_prob.unsqueeze(0).unsqueeze(0),
                                size=(tile_size, tile_size),
                                mode='bilinear',
                            )[0][0]
                            mask_tile = (full_probs > out_threshold).cpu().numpy().astype(np.uint8)
                        else:
                            full_probs = F.interpolate(probs.unsqueeze(0), size=(tile_size, tile_size), mode='bilinear')[0]
                            mask_tile = full_probs.argmax(dim=0).cpu().numpy()
                    else:
                        probs = torch.sigmoid(output)[0]
                        full_probs = F.interpolate(probs.unsqueeze(0), size=(tile_size, tile_size), mode='bilinear')[0]
                        mask_tile = (full_probs[0] > out_threshold).cpu().numpy().astype(np.uint8)

                if net.n_classes <= 2:
                    tissue_mask = None
                    if apply_tissue_mask:
                        tissue_mask = estimate_tissue_mask(raw_tile, white_threshold=white_threshold)
                    mask_tile = apply_binary_postprocessing(
                        mask_tile,
                        tissue_mask=tissue_mask,
                        fill_holes=fill_holes,
                        closing_radius=closing_radius,
                        min_component_area=min_component_area,
                        max_component_area=max_component_area,
                        max_component_extent=max_component_extent,
                    )

                # Map the predicted tile mask back into the global mask array
                full_mask[y:y_end, x:x_end] = mask_tile[0:actual_h, 0:actual_w]

    return full_mask

def get_args():
    parser = argparse.ArgumentParser(description='Predict HSPN masks from large TIFFs with Contrast Enhancement')
    parser.add_argument('--model', '-m', default='checkpoints/best.pth', help='Path to model checkpoint')
    parser.add_argument('--input', '-i', nargs='+', required=True, help='Paths to input .tiff files')
    parser.add_argument('--output', '-o', nargs='+', help='Custom output filenames')
    parser.add_argument('--tile-size', '-t', type=int, default=1024, help='Sliding window tile size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for mask generation')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for each tile before inference')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of target classes')
    parser.add_argument('--apply-tissue-mask', action='store_true', default=False,
                        help='Keep predictions only inside non-white tissue regions estimated from the raw tile')
    parser.add_argument('--white-threshold', type=float, default=230.0,
                        help='Intensity threshold used to estimate non-white tissue for postprocessing')
    parser.add_argument('--fill-holes', action='store_true', default=False,
                        help='Fill holes inside each predicted binary mask tile')
    parser.add_argument('--closing-radius', type=int, default=0,
                        help='Morphological closing radius used to fill small gaps in predicted masks')
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
    
    # Initialize U-Net architecture
    net = UNet(n_channels=3, n_classes=args.classes)
    
    logging.info(f'Loading weights from: {args.model}')
    state_dict = load_torch_state(args.model, map_location=device)
    if 'mask_values' in state_dict:
        state_dict.pop('mask_values')
    net.load_state_dict(state_dict)
    net.to(device=device)
    logging.info('Model successfully loaded on device: {}'.format(device))

    input_files = args.input
    output_files = args.output or [f"{os.path.splitext(f)[0]}_PRED_HSPN.png" for f in input_files]

    for i, file_path in enumerate(input_files):
        logging.info(f'Processing image: {file_path}...')
        # Execute prediction with built-in contrast stretching
        result_mask = predict_hspn_tiles(
            net,
            file_path,
            device,
            tile_size=args.tile_size,
            out_threshold=args.threshold,
            scale_factor=args.scale,
            apply_tissue_mask=args.apply_tissue_mask,
            white_threshold=args.white_threshold,
            fill_holes=args.fill_holes,
            closing_radius=args.closing_radius,
            min_component_area=args.min_component_area,
            max_component_area=args.max_component_area,
            max_component_extent=args.max_component_extent,
        )
        
        save_path = output_files[i]
        # For binary-style segmentation outputs, rescale 0/1 to 0/255 for visibility.
        if args.classes <= 2:
            mask_to_save = Image.fromarray(result_mask * 255)
        else:
            mask_to_save = Image.fromarray(result_mask.astype(np.uint8))

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mask_to_save.save(save_path)
        logging.info(f'Saved prediction to: {save_path}')
