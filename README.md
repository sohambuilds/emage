# Dual-Stream Vision-Language Feedback Loop for Flux Diffusion Models

This project implements a feedback-based image generation system that combines Flux diffusion models with Vision-Language Model (VLM) evaluation to create a closed-loop improvement pipeline for generated images.

## Features

- Initial image generation with Flux.1-dev
- Intelligent model selection based on detected issues
- Control conditioning using Flux Canny and Depth models
- Targeted inpainting with Flux Fill model
- Vision-Language Model evaluation using Google's Gemini API
- Automated mask generation for precise refinements
- Prompt refinement based on VLM feedback
- Dual-stream feedback loop optimization

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. For enhanced functionality, install optional dependencies:
   ```
   pip install controlnet-aux
   pip install git+https://github.com/huggingface/image_gen_aux
   ```
4. Set up environment variables:
   - Create a `.env` file with your API keys
   - Add `GEMINI_API_KEY=your_gemini_api_key_here`
   - Add `HF_TOKEN=your_huggingface_token_here` (required for accessing Flux models)

## Pipeline Architecture

Our system implements an intelligent feedback loop that selects the most appropriate model based on the type of issues detected:

1. **Initial Generation**: Create an image using Flux.1-dev
2. **VLM Evaluation**: Analyze the image with Gemini to detect issues and calculate satisfaction score
3. **Strategy Determination**: Decide whether to refine the prompt, image, or both
4. **Model Selection Logic**:
   - **Spatial/positioning issues**: Use Depth control model for better spatial awareness
   - **Style/textural issues**: Use Canny control model for edge preservation
   - **General issues**: Use Fill model directly for targeted refinements
5. **Refinement Application**: Apply the selected model(s) to refine the image
6. **Re-evaluation**: Assess the improved image and continue iterating until satisfaction

## Usage

```python
from flux_vlm_pipeline.pipeline import dual_stream_feedback_loop

# Generate image with feedback loop
final_image, refined_prompt, score, prompt_history, image_history = dual_stream_feedback_loop(
    prompt="A serene landscape with mountains and a lake at sunset",
    negative_prompt="ugly, blurry", # Used only for initial generation
    max_iterations=5,
    satisfaction_threshold=0.9
)
```

## Command Line Usage

```bash
python main.py --prompt "A magical forest with glowing mushrooms" --max_iterations 3
```

## Model Information

This implementation uses the following models from Hugging Face:

- **Base Model**: `black-forest-labs/FLUX.1-dev` with `FluxPipeline`
  - Used for: Initial image generation
- **Canny Model**: `black-forest-labs/FLUX.1-Canny-dev` with `FluxControlPipeline`
  - Used for: Style and edge-preserving refinements
- **Depth Model**: `black-forest-labs/FLUX.1-Depth-dev` with `FluxControlPipeline`
  - Used for: Spatial and depth-aware refinements
- **Fill Model**: `black-forest-labs/FLUX.1-Fill-dev` with `FluxFillPipeline`
  - Used for: Targeted inpainting of specific regions

Note: Flux control models use channel-wise concatenation rather than traditional ControlNet architecture.

## Implementation Notes

- Negative prompts are only used for the initial image generation, as Flux control and fill models don't support them
- The system intelligently selects which model to use based on issue types
- Masks are generated automatically based on VLM feedback
- Memory usage is optimized through 8-bit quantization support

## References

Based on the research proposal "Dual-Stream Vision-Language Feedback Loop for Flux Diffusion Models"
