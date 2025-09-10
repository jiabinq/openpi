# Changelog

- 2025-09-10: Updated SO-100 policy for PI0.5 compatibility. Modified `src/openpi/policies/so100_policy.py` to add model type switching logic supporting both PI0 and PI0.5 models using the same image layout. This enables the `pi05_clear_tray_fine_tune` config to work correctly with the SO-100 robot's 3-camera setup. Successfully tested data loading and normalization statistics computation.

- 2025-09-09: Added pi0.5 fine-tuning config for the clear_tray dataset. Created `src/openpi/training/config05.py` based on the latest config structure, which includes the `pi05_clear_tray_fine_tune` config for fine-tuning from the `pi05_base` checkpoint. Removed obsolete `config_old.py` and `config_old_2.py` to simplify project structure.

- 2025-09-06: Shape-aware checkpoint weight merge for pi0 fine-tuning with smaller action_dim (6). Updated `src/openpi/training/weight_loaders.py` (`_merge_params`) to load only matching-shape weights and warn on mismatches (e.g., `action_in_proj`, `state_proj`, `action_out_proj`). This enables fine-tuning `pi0_clear_tray_fine_tune` from the `pi0_base` checkpoint while preserving pretrained backbone weights.
