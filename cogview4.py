# Save this as `cogview4_node.py` in your ComfyUI `custom_nodes` folder

import torch
import numpy as np
import comfy.model_management as mm
from PIL import Image
from diffusers import CogView4Pipeline


# Utility functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def convert_preview_image(images):
    images_tensors = []
    for img in images:
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float() / 255.
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        images_tensors.append(img_tensor)

    if len(images_tensors) > 1:
        output_image = torch.cat(images_tensors, dim=0)
    else:
        output_image = images_tensors[0]
    return output_image


# Model Loader Node
class CogView4ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "THUDM/CogView4-6B"}),
                "dtype": (["bfloat16", "float32"], {"default": "bfloat16"}),
                "load_local_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "local_model_path": ("STRING", {"default": "THUDM/CogView4-6B"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "CogView4"

    def load_model(self, model, dtype,load_local_model, *args, **kwargs):
        _DTYPE = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        device = mm.get_torch_device()

        if load_local_model:
            model_path = kwargs.get("local_model_path", model)
        else:
            model_path = model

        # Load the pre-trained CogView4 pipeline
        pipe = CogView4Pipeline.from_pretrained(model_path, torch_dtype=_DTYPE)

        # Optimize for performance
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        # Move to device
        pipe.to(device)

        return (pipe,)


# Image Generator Node
class CogView4ImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": "A serene landscape with mountains and a river", "multiline": True}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 500}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 720, "min": 256, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "cogview4_output.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "CogView4"

    def generate_image(
            self,
            model,
            prompt,
            guidance_scale,
            num_images_per_prompt,
            num_inference_steps,
            width,
            height,
            seed,
            output_path=None
    ):
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Generate the image using the loaded pipeline
        images = model(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
        ).images

        # Save the first image if output_path is provided
        if output_path:
            from pathlib import Path
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            images[0].save(output_path)
            print(f"Image saved to {output_path}")

        # Convert images to ComfyUI-compatible tensor format
        output_image = convert_preview_image(images)
        return (output_image,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CogView4ModelLoader": CogView4ModelLoader,
    "CogView4ImageGenerator": CogView4ImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CogView4ModelLoader": "CogView4 Model Loader",
    "CogView4ImageGenerator": "CogView4 Image Generator"
}
