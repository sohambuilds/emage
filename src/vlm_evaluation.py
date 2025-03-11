"""
VLM Evaluation Module using Google's Gemini API.
"""

import os
import requests
import google.generativeai as genai
from PIL import Image

from .config import GEMINI_API_KEY
from .utils import image_to_base64, parse_api_feedback, calculate_satisfaction


# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)


def evaluate_image(image, prompt):
    """Evaluate an image against a prompt using Google's Gemini API."""
    try:
        # Configure the Gemini model
        model = genai.GenerativeModel('gemini-pro-vision')
        
        # Create a prompt for the Gemini model
        gemini_prompt = f"""
        Carefully analyze this image in relation to the following prompt:
        
        "{prompt}"
        
        First, describe the image in detail.
        
        Then, identify any issues or discrepancies between the prompt and the image, focusing on:
        1. Missing objects or elements mentioned in the prompt
        2. Incorrect or inaccurate depictions of objects
        3. Style inconsistencies
        4. Other problems with prompt adherence
        
        For each issue, be specific about what's wrong and where in the image the problem is located.
        
        If there are no significant issues, acknowledge that the image matches the prompt well.
        """
        
        # Send the request to Gemini
        response = model.generate_content(
            [gemini_prompt, image],
            stream=False
        )
        
        # Extract feedback from response
        feedback = response.text
        
        # Parse feedback to identify issues
        issues = parse_api_feedback(feedback)
        
        # Calculate satisfaction score
        satisfaction_score = calculate_satisfaction(feedback)
        
        return issues, feedback, satisfaction_score
    
    except Exception as e:
        print(f"Error evaluating image with Gemini API: {e}")
        # Return default values in case of error
        return [{"type": "api_error", "description": f"API evaluation failed: {str(e)}"}], f"Error evaluating image: {str(e)}", 0.0


def analyze_prompt(prompt, issues, feedback):
    """Analyze the original prompt to identify potential improvements."""
    try:
        # Configure the Gemini model
        model = genai.GenerativeModel('gemini-pro')
        
        # Create a prompt for the Gemini model
        gemini_prompt = f"""
        I need your help improving a text prompt for an AI image generator.
        
        Original prompt: "{prompt}"
        
        Based on an AI evaluation of the generated image, the following feedback was provided:
        "{feedback}"
        
        The specific issues identified were:
        {format_issues(issues)}
        
        Analyze whether the original prompt could be improved to address these issues. Consider:
        
        1. Ambiguities in the prompt that might lead to misinterpretations
        2. Missing details that would help guide the image generation
        3. Contradictions or unclear instructions
        4. Style specifications that could be clarified
        
        Provide specific suggestions for improving the prompt. Be detailed and actionable in your recommendations.
        Focus on making the prompt clearer and more precise, not on complete rewrites.
        """
        
        # Send the request to Gemini
        response = model.generate_content(gemini_prompt)
        
        # Extract analysis from response
        analysis = response.text
        
        # Extract refinement suggestions from analysis
        from .utils import extract_suggestions
        suggestions = extract_suggestions(analysis)
        
        return suggestions, analysis
    
    except Exception as e:
        print(f"Error analyzing prompt with Gemini API: {e}")
        # Return default values in case of error
        return ["Make the prompt more specific and detailed."], f"Error analyzing prompt: {str(e)}"


def refine_prompt(prompt, suggestions):
    """Generate a refined prompt based on suggestions."""
    try:
        # Configure the Gemini model
        model = genai.GenerativeModel('gemini-pro')
        
        # Create a prompt for the Gemini model
        from .utils import format_suggestions
        gemini_prompt = f"""
        I need your help refining a text prompt for an AI image generator.
        
        Original prompt: "{prompt}"
        
        Based on analysis of generated images, the following suggestions were made:
        {format_suggestions(suggestions)}
        
        Please create an improved version of the prompt that:
        1. Maintains the core intent and subject matter of the original prompt
        2. Addresses the suggestions for improvement
        3. Is clearer, more specific, and more detailed
        4. Doesn't exceed 100 words
        
        Provide ONLY the refined prompt, nothing else.
        """
        
        # Send the request to Gemini
        response = model.generate_content(gemini_prompt)
        
        # Extract refined prompt from response
        refined_prompt = response.text.strip()
        
        # Ensure we didn't get an empty response
        if not refined_prompt:
            return prompt
        
        return refined_prompt
    
    except Exception as e:
        print(f"Error refining prompt with Gemini API: {e}")
        # Return the original prompt in case of error
        return prompt


def format_issues(issues):
    """Format issues for Gemini API prompt."""
    formatted_issues = []
    for i, issue in enumerate(issues):
        issue_desc = f"Issue {i+1}: Type: {issue['type']}"
        if "description" in issue:
            issue_desc += f", Description: {issue['description']}"
        if "object" in issue:
            issue_desc += f", Object: {issue['object']}"
        if "region" in issue:
            issue_desc += f", Region: {issue['region']}"
        formatted_issues.append(issue_desc)
    
    return "\n".join(formatted_issues)


def determine_refinement_strategy(issues, feedback):
    """Determine whether to refine the prompt, the image, or both."""
    from .utils import count_prompt_related_issues, count_image_generation_issues
    
    # Count issues by type
    prompt_issues = count_prompt_related_issues(issues)
    image_issues = count_image_generation_issues(issues)
    
    # Determine strategy based on issue counts
    if prompt_issues > image_issues * 2:
        return "prompt_only"  # Prompt issues dominate, focus on refining prompt
    elif image_issues > prompt_issues * 2:
        return "image_only"   # Image issues dominate, focus on inpainting
    else:
        return "prompt_and_image"  # Mixed issues, refine both