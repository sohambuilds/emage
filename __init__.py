"""
Flux VLM Pipeline: Dual-Stream Vision-Language Feedback Loop for Flux Diffusion Models
"""

# Import main components
from flux_vlm_pipeline.src.pipeline import dual_stream_feedback_loop, visualize_feedback_loop, save_results
from flux_vlm_pipeline.src.image_generation import generate_initial_image, perform_inpainting
from flux_vlm_pipeline.src.vlm_evaluation import evaluate_image, analyze_prompt, refine_prompt

__version__ = "0.1.0" 