# SO-100 Policy PI0.5 Compatibility Update Plan (Revised)

Note Saved: 10 Sep, 08:41am

## Current Issue
The `src/openpi/policies/so100_policy.py` has hardcoded PI0 model type and lacks proper model type switching logic, preventing compatibility with PI0.5 models.

## Focused Implementation Plan

### 1. **Update S0100Inputs Class Logic**
**File**: `src/openpi/policies/so100_policy.py`

**Changes needed**:
- Replace the current image mapping logic (lines 48-58) with model type switching
- Add match statement similar to `droid_policy.py` pattern:

```python
# Current fixed logic will be replaced with:
match self.model_type:
    case _model.ModelType.PI0 | _model.ModelType.PI05:
        # PI0 and PI0.5 use same image layout (current SO-100 logic)
        images = {
            "base_0_rgb": top_image,
            "left_wrist_0_rgb": wrist_image, 
            "right_wrist_0_rgb": side_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
    case _model.ModelType.PI0_FAST:
        # PI0_FAST uses different image naming/layout if needed later
        images = {
            "base_0_rgb": top_image,
            "base_1_rgb": side_image,
            "wrist_0_rgb": wrist_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "base_1_rgb": np.True_,
            "wrist_0_rgb": np.True_,
        }
    case _:
        raise ValueError(f"Unsupported model type: {self.model_type}")
```

### 2. **Testing Strategy - PI0.5 Focus Only**

**Target configuration**: `pi05_clear_tray_fine_tune` only

**Validation steps**:
1. Verify PI0.5 config loads successfully with updated policy
2. Check image mappings work correctly for PI0.5
3. Test data pipeline with compute_norm_stats
4. Verify training initialization works

### 3. **Verification Commands**

```bash
# Test PI0.5 config data loading only
uv run scripts/compute_norm_stats.py --config-name pi05_clear_tray_fine_tune --max-frames 100

# Verify PI0.5 training init works
XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 uv run scripts/train.py pi05_clear_tray_fine_tune --exp-name=test_pi05 --num-train-steps=1
```

## Key Scope Changes
- ❌ **No changes to `config_pi0.py`** - kept for reference only
- ✅ **Focus only on PI0.5 compatibility** via `pi05_clear_tray_fine_tune`
- ✅ **Single policy file update** in `so100_policy.py`
- ✅ **Targeted testing** on current priority configuration

## Expected Outcomes
- ✅ SO-100 policy works with PI0.5 models
- ✅ `pi05_clear_tray_fine_tune` config functions properly
- ✅ Clear_tray_3cam dataset loads correctly with PI0.5
- ✅ Proper error handling for unsupported model types

## Risk Assessment
- **Low risk**: Minimal changes, focused scope
- **Well-tested pattern**: Same logic used in other policies
- **Current priority**: Directly addresses PI0.5 training needs