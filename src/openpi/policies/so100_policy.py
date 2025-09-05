import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so100_example() -> dict:
    """Creates a random input example for the SO-100 single-arm policy."""
    return {
        "observation.state": np.random.rand(6),
        "observation.images.top": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.side": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class S0100Inputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Your dataset provides the state as a single, combined vector.
        state = data["observation.state"]
        state = transforms.pad_to_dim(state, self.action_dim)

        # Parse images using the keys from your dataset_info.json
        top_image = _parse_image(data["observation.images.top"])
        wrist_image = _parse_image(data["observation.images.wrist"])
        side_image = _parse_image(data["observation.images.side"])

        # Map the dataset cameras to the logical cameras the model expects.
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

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        # Actions are only available during training.
        if "action" in data:
            actions = transforms.pad_to_dim(data["action"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class S0100Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Trim actions to 6 for your single-arm robot.
        return {"actions": np.asarray(data["actions"][:, :6])}
