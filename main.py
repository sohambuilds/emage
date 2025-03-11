"""
Main script for the Flux VLM Pipeline.
Demonstrates how to use the dual-stream feedback loop.
"""

import os
import argparse
from dotenv import load_dotenv

from src.pipeline import dual_stream_feedback_loop, save_results
from src.config import DEFAULT_MAX_ITERATIONS, DEFAULT_SATISFACTION_THRESHOLD

# Load environment variables
load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Flux VLM Dual-Stream Feedback Loop")
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Text prompt for image generation"
    )
    
    parser.add_argument(
        "--negative_prompt", 
        type=str, 
        default="",
        help="Negative text prompt for image generation"
    )
    
    parser.add_argument(
        "--max_iterations", 
        type=int, 
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum number of refinement iterations (default: {DEFAULT_MAX_ITERATIONS})"
    )
    
    parser.add_argument(
        "--satisfaction_threshold", 
        type=float, 
        default=DEFAULT_SATISFACTION_THRESHOLD,
        help=f"Score threshold to stop refinement (default: {DEFAULT_SATISFACTION_THRESHOLD})"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for model inference (default: cuda)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs",
        help="Directory to save output files (default: outputs)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check environment
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Flux VLM Dual-Stream Feedback Loop ===")
    print(f"Prompt: {args.prompt}")
    print(f"Negative prompt: {args.negative_prompt}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Satisfaction threshold: {args.satisfaction_threshold}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("=========================================")
    
    # Run the dual-stream feedback loop
    final_image, final_prompt, final_score, prompt_history, image_history = dual_stream_feedback_loop(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        max_iterations=args.max_iterations,
        satisfaction_threshold=args.satisfaction_threshold,
        device=args.device
    )
    
    # Save the results
    result_dir = save_results(
        image=final_image,
        prompt=final_prompt,
        score=final_score,
        prompt_history=prompt_history,
        image_history=image_history,
        output_dir=args.output_dir
    )
    
    print(f"\nResults saved to: {result_dir}")
    print(f"Final prompt: '{final_prompt}'")
    print(f"Final satisfaction score: {final_score:.2f}")


if __name__ == "__main__":
    main() 