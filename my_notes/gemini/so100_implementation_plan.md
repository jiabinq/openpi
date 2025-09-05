# SO-100 Policy Implementation Plan

This document outlines the concrete steps to write the necessary code for integrating a new SO-100 robot arm (with 5 joints, 3 cameras, and 1 gripper) into the `openpi` framework for fine-tuning.

### Objective
To create a new policy and training configuration for the specified SO-100 robot, enabling fine-tuning of the base `pi0` or `pi0_fast` models on a custom dataset.

### Files to Be Created / Modified
1.  **New File:** `src/openpi/policies/so100_policy.py`
2.  **Modified File:** `src/openpi/training/config_phospho.py`

---

## Step 1: Create the Policy "Toolkit" (`so100_policy.py`)

This is the most important step. We will create a new file that defines the specific data conversion logic for the SO-100 arm. We will start with the simpler, "Droid-style" approach that assumes no complex mathematical transformations are needed.

**Action:** Create the file `src/openpi/policies/so100_policy.py` with the following content.

```python
# src/openpi/policies/so100_policy.py

import dataclasses
import numpy as np
from openpi import transforms

# Helper function to make sure images are in the right format
def _parse_image(image: np.ndarray) -> np.ndarray:
    """Ensures image is a uint8 numpy array in (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # Add other checks here if needed, e.g., rearranging channels if your
    # dataset uses a different convention.
    return image


@dataclasses.dataclass(frozen=True)
class SO100Inputs(transforms.DataTransformFn):
    """
    This class converts data from the SO-100 dataset into the format
    the model expects.
    """
    # This will be 6 for your robot (5 joints + 1 gripper).
    # It will be passed in from the TrainConfig.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        # 1. --- STATE ---
        # Combine joint and gripper positions into a single '''state''' array.
        #
        # IMPORTANT: Replace the keys below with the actual keys from your dataset!
        state = np.concatenate([
            data["observation/joint_position"], # Should be an array of 5 numbers
            data["observation/gripper_position"] # Should be an array of 1 number
        ])

        # 2. --- IMAGES ---
        # IMPORTANT: Replace the camera names below with the actual keys from your dataset!
        main_image = _parse_image(data["observation/images/main_cam"])
        wrist_1_image = _parse_image(data["observation/images/wrist_cam_1"])
        wrist_2_image = _parse_image(data["observation/images/wrist_cam_2"])

        # The model expects specific names. We map our 3 cameras to these names.
        image_dict = {
            "base_0_rgb": main_image,
            "left_wrist_0_rgb": wrist_1_image,
            "right_wrist_0_rgb": wrist_2_image,
        }
        # The image_mask tells the model that all 3 images are real data.
        image_mask_dict = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        # 3. --- ASSEMBLE FINAL INPUTS ---
        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask_dict,
        }

        # This part is for training, where actions are available in the dataset.
        if "actions" in data:
            actions = np.array(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO100Outputs(transforms.DataTransformFn):
    """
    This class converts the model's predicted actions back into a format
    the robot environment can understand.
    """
    def __call__(self, data: dict) -> dict:
        # The model might output a longer action vector, so we trim it
        # to the 6 dimensions our robot uses.
        actions = np.asarray(data["actions"][:, :6])
        return {"actions": actions}
```

---

## Step 2: Modify the Configuration File (`config_phospho.py`)

Now we will create the "recipes" that use the toolkit we just defined.

**Action:** Open `src/openpi/training/config_phospho.py` and make the following three changes.

#### A. Import the New Policy

At the top of the file, with the other policy imports, add:

```python
import openpi.policies.so100_policy as so100_policy
```

#### B. Create the "Data Recipe"

Add this new `DataConfigFactory` class alongside the other `LeRobot...` classes.

```python
@dataclasses.dataclass(frozen=True)
class LeRobotSO100DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # This transform renames/reorganizes keys from the dataset.
        # For this simple case, we assume the keys are already what the policy expects,
        # but you could add a RepackTransform here if needed.
        repack_transform = _transforms.Group()

        # Use the SO100 toolkit we defined in so100_policy.py
        data_transforms = _transforms.Group(
            inputs=[so100_policy.SO100Inputs(action_dim=model_config.action_dim)],
            outputs=[so100_policy.SO100Outputs()],
        )

        # Apply delta actions to the 5 joints, but not the gripper.
        delta_action_mask = _transforms.make_bool_mask(5, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms are standard (e.g., tokenizing prompts).
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

#### C. Create the "Experiment Blueprint"

At the bottom of the file, inside the `_CONFIGS` list, add a new `TrainConfig` for your experiment.

```python
# Add this inside the _CONFIGS = [...] list

TrainConfig(
    # A unique name for your experiment blueprint
    name="pi0_so100_custom_fine_tune",

    # Define the model and its action dimension (5 joints + 1 gripper = 6)
    model=pi0.Pi0Config(action_dim=6),

    # Use the data recipe you just created and point it to your dataset
    data=LeRobotSO100DataConfig(
        # IMPORTANT: Replace with your Hugging Face dataset repository ID
        repo_id="your-hf-username/your-so100-dataset",
        base_config=DataConfig(
            prompt_from_task=True,
            local_files_only=False, # Set to True if dataset is only local
        ),
    ),

    # Load weights from the pre-trained base model to start fine-tuning
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "s3://openpi-assets/checkpoints/pi0_base/params"
    ),

    # Set your desired number of training steps
    num_train_steps=30_000,
),
```

---

## Step 3: Training and Verification

After implementing the code changes above, you can start a training run using your new configuration name:

```bash
python -m openpi.train --config-name=pi0_so100_custom_fine_tune --exp-name=first_run
```

**How to Verify:**

1.  **Monitor Training Loss:** Watch the training logs or `wandb`. If the loss decreases steadily, it's a great sign that the simple, "Droid-style" policy is working correctly.
2.  **If Loss is Flat/Erratic:** If the model fails to learn, it suggests a data mismatch. You would then need to investigate the differences between your SO-100's data (joint ranges, directions) and the "standard" data the `pi0` model expects. If you find a mismatch, you would then implement the more complex, "Aloha-style" mathematical transformations in your `so100_policy.py` file.
