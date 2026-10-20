# Third-party code

- Root U-Net implementation retains the repository's existing GPL-3.0 license and attribution.
- `vendor/ihc-main/` is the user-supplied `ihc-main.zip` implementation of DPA-P2PNet. Original archive SHA-256: `30d2221b15d224c7419bbab01a523877711bd852b9166b3e4a87d5eb75531789`.
- Related upstream: https://github.com/windygoo/DPA-P2PNet . Existing source-file copyright/license notices are retained. The local archive did not contain a top-level license grant; confirm upstream terms before redistributing it outside the project.
- Local adaptations include dataset class-count forwarding, training augmentation selection, matching-index-zero correction, evaluation/save intervals and a differentiable PyTorch deformable-attention fallback.
- Unused hospital database extraction scripts and bundled example images are excluded from this delivery.
- The HFRM-CAM implementation in `scripts/hzy_weakseg_hfrm.py` is SSHR-inspired project code, not a claim of exact reproduction of an upstream SSHR release.
