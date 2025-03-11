"""
Utility functions for the Flux VLM Pipeline.
"""

import base64
import io
import re
from PIL import Image


def image_to_base64(image):
    """Convert PIL Image to base64 encoded string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


def parse_api_feedback(feedback):
    """
    Parse feedback from the Gemini API to extract issues.
    
    Returns:
        list: List of issue dictionaries with type, description, and possibly object and region.
    """
    issues = []
    
    # Check for missing elements
    missing_pattern = r"missing (\w+|[\w\s]+)"
    missing_matches = re.finditer(missing_pattern, feedback.lower())
    for match in missing_matches:
        issues.append({
            "type": "missing_element",
            "object": match.group(1),
            "description": f"The {match.group(1)} is missing from the image"
        })
    
    # Check for incorrect elements
    incorrect_pattern = r"incorrect (\w+|[\w\s]+)"
    incorrect_matches = re.finditer(incorrect_pattern, feedback.lower())
    for match in incorrect_matches:
        issues.append({
            "type": "incorrect_element",
            "object": match.group(1),
            "description": f"The {match.group(1)} is depicted incorrectly"
        })
    
    # Check for style issues
    style_pattern = r"style (issue|problem|inconsistency)"
    style_matches = re.finditer(style_pattern, feedback.lower())
    if any(style_matches):
        issues.append({
            "type": "style_issue",
            "description": "The style of the image doesn't match the prompt"
        })
    
    # Add a general issue if none were found but the feedback is negative
    negative_indicators = ["issue", "problem", "incorrect", "missing", "doesn't match", "discrepancy"]
    if not issues and any(indicator in feedback.lower() for indicator in negative_indicators):
        issues.append({
            "type": "general_issue",
            "description": "The image has issues matching the prompt"
        })
    
    return issues


def calculate_satisfaction(feedback):
    """
    Calculate a satisfaction score based on feedback.
    
    Returns:
        float: Score between 0.0 and 1.0, where 1.0 indicates perfect satisfaction.
    """
    # List of positive phrases indicating satisfaction
    positive_phrases = [
        "matches the prompt well",
        "accurately depicts",
        "successfully captures",
        "aligns with the prompt",
        "no significant issues",
        "excellent representation",
        "adheres to the prompt",
        "fulfills the prompt"
    ]
    
    # List of negative phrases indicating problems
    negative_phrases = [
        "missing",
        "incorrect",
        "issue",
        "problem",
        "discrepancy",
        "doesn't match",
        "inaccurate",
        "fails to"
    ]
    
    # Count positive and negative indicators
    positive_count = sum(phrase in feedback.lower() for phrase in positive_phrases)
    negative_count = sum(phrase in feedback.lower() for phrase in negative_phrases)
    
    # Calculate base score
    total_indicators = positive_count + negative_count
    if total_indicators == 0:
        return 0.5  # Neutral if no indicators
    
    raw_score = positive_count / total_indicators
    
    # Adjust score based on the presence of specific phrases
    if "no issues" in feedback.lower() or "matches perfectly" in feedback.lower():
        raw_score = min(1.0, raw_score + 0.2)
    
    if "significant issues" in feedback.lower() or "major problems" in feedback.lower():
        raw_score = max(0.0, raw_score - 0.2)
    
    return raw_score


def extract_suggestions(analysis):
    """
    Extract specific suggestions from the analysis text.
    
    Returns:
        list: List of suggestion strings.
    """
    suggestions = []
    
    # Look for lines with suggestions
    lines = analysis.split('\n')
    for line in lines:
        # Check for bullet points, numbers, or suggestion keywords
        if (re.match(r'^[\*\-\•\d]+', line.strip()) or 
            any(keyword in line.lower() for keyword in ["suggest", "should", "could", "add", "specify", "clarify"])):
            
            # Clean the suggestion line
            clean_line = re.sub(r'^[\*\-\•\d\.]+\s*', '', line.strip())
            if clean_line and len(clean_line) > 10:  # Ensure it's a substantial suggestion
                suggestions.append(clean_line)
    
    # If no structured suggestions found, try to extract sentences with recommendation keywords
    if not suggestions:
        sentences = re.split(r'[.!?]+', analysis)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ["suggest", "should", "could", "add", "specify", "clarify"]):
                clean_sentence = sentence.strip()
                if clean_sentence and len(clean_sentence) > 10:
                    suggestions.append(clean_sentence)
    
    # If still no suggestions, add a default one
    if not suggestions:
        suggestions.append("Make the prompt more specific and detailed")
    
    return suggestions


def format_suggestions(suggestions):
    """Format a list of suggestions for use in a prompt."""
    return "\n".join([f"- {suggestion}" for suggestion in suggestions])


def count_prompt_related_issues(issues):
    """Count issues that are likely related to prompt clarity."""
    prompt_issue_types = ["missing_element", "general_issue", "style_issue"]
    return sum(1 for issue in issues if issue.get("type", "") in prompt_issue_types)


def count_image_generation_issues(issues):
    """Count issues that are likely related to image generation quality."""
    image_issue_types = ["incorrect_element", "distortion", "artifact"]
    return sum(1 for issue in issues if issue.get("type", "") in image_issue_types) 