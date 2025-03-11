"""
Configuration settings for the Flux VLM Pipeline.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables. VLM evaluation will not work.")

# Hugging Face token for accessing models
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in environment variables. Model loading may fail.")

# Model IDs
FLUX_BASE_MODEL_ID = "fluxbeaver/flux-dev-base"
FLUX_CANNY_MODEL_ID = "fluxbeaver/flux-dev-canny"
FLUX_FILL_MODEL_ID = "fluxbeaver/flux-dev-fill"

# Pipeline parameters
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_SATISFACTION_THRESHOLD = 0.8
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# Image generation parameters
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_STRENGTH = 0.75 