"""
Flux VLM Pipeline integrating image generation and VLM evaluation components.
"""

from PIL import Image
import os
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union

from .image_generation import (
    perform_base_generation,
    perform_control_generation,
    perform_inpainting,
    DepthPreprocessor
)
from .vlm_evaluation import (
    evaluate_image,
    analyze_prompt,
    refine_prompt,
    determine_refinement_strategy
)
from .utils import (
    image_to_base64,
    count_prompt_related_issues,
    count_image_generation_issues
)
from .config import HF_TOKEN


class FluxVLMPipeline:
    """Intelligent VLM feedback pipeline combining Flux image models with Gemini evaluations."""
    
    def __init__(self):
        """Initialize the Flux VLM Pipeline."""
        # Initialize preprocessors
        self.depth_preprocessor = DepthPreprocessor.from_pretrained(
            "stabilityai/control-lora-depth-rank-anything",
            use_auth_token=HF_TOKEN
        )
        
        # Track pipeline history
        self.history = []
    
    def generate_and_evaluate(
        self,
        prompt: str,
        num_iterations: int = 3,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024
    ) -> Dict:
        """Generate an image and evaluate it with VLM feedback, iteratively refining as needed."""
        current_prompt = prompt
        image = None
        current_iteration = 0
        
        while current_iteration < num_iterations:
            print(f"Iteration {current_iteration + 1}/{num_iterations}")
            print(f"Current prompt: {current_prompt}")
            
            # Generate initial image if first iteration or based on strategy
            if current_iteration == 0:
                # First iteration: Generate base image
                print("Generating base image...")
                image = perform_base_generation(
                    prompt=current_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    width=width,
                    height=height
                )
            else:
                # Base refinement on previous evaluation
                last_entry = self.history[-1]
                issues = last_entry["evaluation"]["issues"]
                feedback = last_entry["evaluation"]["feedback"]
                
                # Determine refinement strategy
                strategy = determine_refinement_strategy(issues, feedback)
                print(f"Using refinement strategy: {strategy}")
                
                if strategy == "prompt_only":
                    # Generate new image with refined prompt
                    print("Regenerating with refined prompt...")
                    image = perform_base_generation(
                        prompt=current_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        seed=seed,
                        width=width,
                        height=height
                    )
                
                elif strategy == "image_only":
                    # Refine the existing image with controls
                    print("Refining image with control models...")
                    
                    # Try with depth control for structural issues
                    depth_map = self.depth_preprocessor(image)
                    image = perform_control_generation(
                        prompt=current_prompt,
                        control_image=depth_map,
                        control_mode="depth",
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        seed=seed,
                        strength=0.7  # Higher strength for more adherence to structure
                    )
                
                else:  # prompt_and_image
                    # Determine if we should use control or inpainting
                    if any(issue.get("type") == "incorrect_element" for issue in issues):
                        print("Using inpainting for targeted fixes...")
                        # For issues with specific areas, use inpainting
                        # In a production system, would use segmentation to create a mask
                        center_mask = np.zeros((height, width), dtype=np.uint8)
                        center_mask[height//4:3*height//4, width//4:3*width//4] = 255
                        mask_image = Image.fromarray(center_mask)
                        
                        image = perform_inpainting(
                            prompt=current_prompt,
                            image=image,
                            mask_image=mask_image,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            seed=seed
                        )
                    else:
                        # For more general issues, use control models
                        print("Refining with control model...")
                        image = perform_control_generation(
                            prompt=current_prompt,
                            control_image=image,
                            control_mode="canny",
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            seed=seed,
                            strength=0.5
                        )
            
            # Evaluate the generated image
            print("Evaluating image...")
            issues, feedback, satisfaction_score = evaluate_image(image, current_prompt)
            
            # Record in history
            self.history.append({
                "iteration": current_iteration,
                "prompt": current_prompt,
                "image": image.copy(),
                "evaluation": {
                    "issues": issues,
                    "feedback": feedback,
                    "satisfaction_score": satisfaction_score
                }
            })
            
            print(f"Satisfaction score: {satisfaction_score:.2f}")
            
            # Check if quality is satisfactory
            if satisfaction_score >= 0.8:
                print("Image quality is satisfactory. Stopping iterations.")
                break
            
            # Analyze prompt and get suggestions for refinement
            print("Analyzing prompt for refinement...")
            suggestions, analysis = analyze_prompt(current_prompt, issues, feedback)
            
            # Refine the prompt
            improved_prompt = refine_prompt(current_prompt, suggestions)
            
            # Update prompt for next iteration
            current_prompt = improved_prompt
            print(f"Refined prompt: {current_prompt}")
            
            current_iteration += 1
        
        # Return final results and history
        return {
            "final_image": image,
            "final_prompt": current_prompt,
            "original_prompt": prompt,
            "history": self.history,
            "satisfaction_score": self.history[-1]["evaluation"]["satisfaction_score"]
        }
    
    def get_history(self) -> List[Dict]:
        """Return the pipeline execution history."""
        return self.history 