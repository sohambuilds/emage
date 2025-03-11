# EMage: A Dual-Stream Vision-Language Feedback Pipeline for Flux Diffusion Models

This project implements an intelligent image editing and refinement system called **EMage** (Editing Mage/Image) that combines advanced Flux diffusion models with Vision-Language Model (VLM) evaluation to create a closed-loop improvement pipeline for generated images.

## Technical Overview

The EMage pipeline implements a sophisticated dual-stream optimization approach that alternates between prompt refinement and image manipulation based on semantic evaluation of generation quality. The system leverages three specialized Flux diffusion models and integrates them with Google's Gemini Pro Vision for automated assessment and refinement.

### Core Components

- **Base Image Generation Module**: Implements Flux base model for high-fidelity initial image synthesis
- **Control Conditional Generation**: Utilizes latent space manipulation techniques through Flux Canny control models
- **Depth-Guided Refinement**: Employs monocular depth estimation for structural awareness in image refinement
- **Region-Specific Inpainting**: Applies targeted modifications via Flux Fill models for localized corrections
- **VLM-Powered Evaluation**: Performs multi-faceted image analysis with Gemini API
- **Automated Mask Generation**: Implements contour detection and semantic segmentation for precise editing regions
- **Prompt Engineering System**: Applies NLP techniques to systematically improve text descriptions
- **Intelligent Strategy Selection**: Employs a weighted decision algorithm to choose optimal refinement paths

## Technical Implementation

### Model Architecture

EMage employs a dynamic model selection approach with the following diffusion models:

1. **Flux Base Model**

   - Model ID: `black-forest-labs/FLUX.1-dev`
   - Architecture: Advanced latent diffusion with transformer backbone
   - Parameters: ~12B parameters
   - Primary Application: Initial high-quality image synthesis
   - Features: State-of-the-art text-to-image generation capabilities

2. **Flux Canny Control Model**

   - Model ID: `black-forest-labs/FLUX.1-Canny-dev`
   - Architecture: Specialized control model with edge conditioning
   - Parameters: ~12B parameters
   - Integration Method: Channel-wise concatenation (not ControlNet)
   - Conditioning Inputs: Canny edge maps and raw images
   - Strength Parameter: Controls influence of condition (0.0-1.0)
   - VRAM Optimization: CPU offloading for efficient operation

3. **Flux Fill Model**
   - Model ID: `black-forest-labs/FLUX.1-Fill-dev`
   - Architecture: Inpainting-optimized latent diffusion
   - Parameters: ~12B parameters
   - Mask Processing: Binary attention mechanism
   - Inpainting Method: Masked latent diffusion
   - Coherence Control: Guided latent interpolation at boundaries

All Flux models are built on the same core architecture with specialized training for their respective tasks. Each model contains approximately 12 billion parameters and implements advanced attention mechanisms for high-quality image generation and manipulation.

### VLM Integration Architecture

The system integrates Google's Gemini API for multi-modal analysis:

- **Model**: Gemini Pro Vision
- **Input**: Image-text pairs (generated image + original prompt)
- **Analysis Tasks**:
  - Semantic correspondence detection
  - Object presence verification
  - Stylistic coherence assessment
  - Spatial relationship validation
- **Output Processing**: Natural language feedback parsing with regex-based entity extraction

### Feedback Loop Pipeline

The EMage feedback loop implements the following technical workflow:

1. **Initial State Representation**: Text prompt → Flux Base Model → Initial image tensor
2. **Evaluation**: Image + Prompt → Gemini VLM → Structured issue dictionary + satisfaction score
3. **Decision Tree Processing**:
   - If satisfaction score >= 0.8: Terminate loop (convergence achieved)
   - If prompt issues > image issues: Execute prompt refinement branch
   - If image issues > prompt issues: Execute image modification branch
   - Otherwise: Execute hybrid refinement
4. **Strategy Selection Algorithm**:
   - Issue type classification → Model selection → Parameter optimization
   - Weighting factor for issue severity influences selection probability
5. **Execution and Re-evaluation**: Apply selected strategy → Generate new image → Re-evaluate

## System Requirements

- **Compute**: CUDA-compatible GPU with 24GB+ VRAM
- **API Requirements**:
  - Gemini API access with sufficient quota for multi-modal requests
  - Hugging Face authentication token with model access permissions
- **Memory Requirements**: 16GB+ RAM
- **Dependencies**: PyTorch 2.0+, diffusers 0.19+, transformers 4.30+

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

## EMage Architecture

Our system implements an intelligent feedback loop that utilizes an algorithmic approach to model selection based on issue classification:

1. **Initial Generation Phase**:

   - Utilize Flux Base model with classifier-free guidance
   - Apply scaled latent diffusion with configurable inference steps
   - Extract high-quality latent representations to RGB space

2. **VLM Evaluation Phase**:

   - Multi-modal prompt construction for effective VLM analysis
   - Structured data extraction from natural language responses
   - Quantification of satisfaction via sentiment analysis techniques
   - Issue categorization into object-level, semantic, and structural categories

3. **Strategy Determination Phase**:

   - Apply weighted issue counting to determine refinement vector
   - Calculate ratio of issue types to select optimal modification strategy
   - Dynamic threshold adjustment based on satisfaction trajectory

4. **Model Selection Logic**:

   - **Spatial/positioning issues**: Apply depth-conditioned diffusion for structural refinement
   - **Style/textural issues**: Utilize edge-preservation via Canny conditioning
   - **Object-specific issues**: Implement targeted inpainting with boundary preservation
   - **Prompt ambiguity issues**: Employ semantic parsing and prompt enhancement

5. **Execution Phase**:

   - Apply conditionally-selected model with optimized parameters
   - Process condition maps when applicable (depth, edges, etc.)
   - Maintain coherence via strength parameter optimization
   - Preserve semantic content via prompt guidance

6. **Iterative Refinement**:
   - Track history with state preservation for rollback capability
   - Calculate convergence metrics for termination condition
   - Apply diminishing guidance scale for fine refinement in later iterations
   - Perform cross-iteration analysis for global optimization

## Usage

```python
from flux_vlm_pipeline.pipeline import FluxVLMPipeline

# Initialize the EMage pipeline
emage = FluxVLMPipeline()

# Generate and refine an image with EMage
results = emage.generate_and_evaluate(
    prompt="A photorealistic mountain landscape with pine trees and a lake reflecting the sunset",
    num_iterations=3,
    guidance_scale=7.5,
    num_inference_steps=30,
    seed=42,  # For reproducibility
    width=1024,
    height=1024
)

# Access the final image and refined prompt
final_image = results["final_image"]
final_prompt = results["final_prompt"]
satisfaction_score = results["satisfaction_score"]

# Display the refinement history
history = results["history"]
```

## Technical Details of Model Integration

### Flux Base Model Integration

The EMage system loads the `fluxbeaver/flux-dev-base` model through the diffusers FluxPipeline interface:

```python
flux_pipeline = FluxPipeline.from_pretrained(
    FLUX_BASE_MODEL_ID,
    use_auth_token=HF_TOKEN
)
```

Key parameters for optimal generation:

- **guidance_scale**: Controls adherence to prompt (7.5 default)
- **num_inference_steps**: Controls quality vs. speed tradeoff (30 default)
- **seed**: Integer value for reproducible outputs
- **width/height**: Output image dimensions (defaults to 1024x1024)

### Control Model Implementation

The EMage system utilizes conditional image generation through the Flux Control Pipeline:

```python
control_pipeline = FluxControlPipeline.from_pretrained(
    FLUX_CANNY_MODEL_ID,
    use_auth_token=HF_TOKEN
)
```

Key conditioning parameters:

- **control_image**: The conditioning image (e.g., canny edge map)
- **control_mode**: Type of conditioning ("canny" or "depth")
- **strength**: Degree of conditioning influence (0.5-0.7 recommended)

### Inpainting Integration

For targeted refinements, EMage implements mask-based inpainting:

```python
inpaint_pipeline = FluxFillPipeline.from_pretrained(
    FLUX_FILL_MODEL_ID,
    use_auth_token=HF_TOKEN
)
```

Inpainting parameters:

- **mask_image**: Binary mask indicating regions to modify
- **guidance_scale**: Controls adherence to prompt in modified regions
- **num_inference_steps**: Quality vs. speed tradeoff for inpainting

## Memory Optimization Techniques

The EMage pipeline implements several memory optimization strategies:

1. **CPU Offloading**: Models use `enable_model_cpu_offload()` to reduce VRAM footprint
2. **Single Model Loading**: Only one model is loaded at a time
3. **Strategic Initialization**: The Depth Preprocessor is initialized once at pipeline creation

## References and Technical Papers

EMage builds on several key technical innovations:

- Latent Diffusion Models (Rombach et al., 2022)
- Classifier-Free Guidance (Ho and Salimans, 2021)
- Multimodal Large Language Models (Gemini Technical Report, 2023)
- Feedback Loops for Generative AI (Iterative Refinement via Feedback, Saunders et al., 2022)

## License

This project is provided as an open-source implementation under [MIT License](LICENSE).
