# Changelog

- 2025-09-06: Shape-aware checkpoint weight merge for pi0 fine-tuning with smaller action_dim (6). Updated `src/openpi/training/weight_loaders.py` (`_merge_params`) to load only matching-shape weights and warn on mismatches (e.g., `action_in_proj`, `state_proj`, `action_out_proj`). This enables fine-tuning `pi0_clear_tray_fine_tune` from the `pi0_base` checkpoint while preserving pretrained backbone weights.
