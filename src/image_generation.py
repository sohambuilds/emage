"""
Image Generation Module with Flux Models.
"""

import torch
from PIL import Image
import numpy as np
from diffusers import (
    FluxPipeline,
    FluxControlPipeline,
    FluxFillPipeline,
    DiffusionPipeline
)
from transformers import pipeline

from .config import (
    FLUX_BASE_MODEL_ID,
    FLUX_CANNY_MODEL_ID,
    FLUX_FILL_MODEL_ID,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    HF_TOKEN
)


def perform_base_generation(
    prompt,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    seed=None,
    width=1024,
    height=1024
):
    """Generate a base image using the Flux model."""
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        
    # Load the Flux pipeline
    flux_pipeline = FluxPipeline.from_pretrained(
        FLUX_BASE_MODEL_ID,
        use_auth_token=HF_TOKEN
    )
    
    # Enable CPU offloading to reduce VRAM usage
    flux_pipeline.enable_model_cpu_offload()
    
    # Generate the image
    image = flux_pipeline(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height
    ).images[0]
    
    return image


def perform_control_generation(
    prompt,
    control_image,
    control_mode="canny",
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    seed=None,
    strength=0.5
):
    """Generate an image using the Flux Control model."""
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Load the Flux Control pipeline
    control_pipeline = FluxControlPipeline.from_pretrained(
        FLUX_CANNY_MODEL_ID,
        use_auth_token=HF_TOKEN
    )
    
    # Enable CPU offloading to reduce VRAM usage
    control_pipeline.enable_model_cpu_offload()
    
    # Generate the image
    image = control_pipeline(
        prompt=prompt,
        image=control_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    
    return image


class DepthPreprocessor:
    """Preprocessor for generating depth maps from images."""
    
    def __init__(self, model):
        self.model = model
    
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        """Load a depth estimation model from HuggingFace."""
        depth_estimator = pipeline(
            "depth-estimation", 
            model=model_id,
            **kwargs
        )
        return cls(depth_estimator)
    
    def __call__(self, image):
        """Process an image to get its depth map."""
        depth = self.model(image)["depth"]
        depth_image = depth.convert("RGB")
        return depth_image


def perform_inpainting(
    prompt,
    image,
    mask_image,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    seed=None
):
    """Perform inpainting on parts of an image."""
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Load the Flux Fill pipeline
    inpaint_pipeline = FluxFillPipeline.from_pretrained(
        FLUX_FILL_MODEL_ID,
        use_auth_token=HF_TOKEN
    )
    
    # Enable CPU offloading to reduce VRAM usage
    inpaint_pipeline.enable_model_cpu_offload()
    
    # Perform inpainting
    result = inpaint_pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    
    return result 