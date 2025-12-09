"""
GAIA Benchmark Scorer
Implements scoring functions for evaluating predictions against ground truth answers.
"""
import re
from typing import Union


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not isinstance(answer, str):
        answer = str(answer)
    
    # Convert to lowercase
    answer = answer.lower().strip()
    
    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer)
    
    # Remove common punctuation
    answer = re.sub(r'[.,;:!?]', '', answer)
    
    return answer


def question_scorer(prediction: str, true_answer: str) -> bool:
    """
    Score a prediction against the true answer.
    
    Args:
        prediction: The model's prediction
        true_answer: The ground truth answer
        
    Returns:
        bool: True if the prediction is correct, False otherwise
    """
    if not prediction or not true_answer:
        return False
    
    pred_norm = normalize_answer(prediction)
    true_norm = normalize_answer(true_answer)
    
    # Exact match
    if pred_norm == true_norm:
        return True
    
    # Check if prediction contains the true answer
    if true_norm in pred_norm:
        return True
    
    # Check if true answer contains the prediction (for partial matches)
    if pred_norm in true_norm and len(pred_norm) > 3:
        return True
    
    # Numeric comparison
    try:
        pred_num = float(re.sub(r'[^\d.]', '', pred_norm))
        true_num = float(re.sub(r'[^\d.]', '', true_norm))
        if abs(pred_num - true_num) < 0.01:
            return True
    except (ValueError, TypeError):
        pass
    
    return False


def check_close_call(prediction: str, true_answer: str, is_correct: bool) -> bool:
    """
    Check if prediction is a close call (nearly correct but not exact).
    
    Args:
        prediction: The model's prediction
        true_answer: The ground truth answer
        is_correct: Whether the prediction was marked as correct
        
    Returns:
        bool: True if it's a close call, False otherwise
    """
    if is_correct:
        return False
    
    if not prediction or not true_answer:
        return False
    
    pred_norm = normalize_answer(prediction)
    true_norm = normalize_answer(true_answer)
    
    # Check for significant overlap
    pred_words = set(pred_norm.split())
    true_words = set(true_norm.split())
    
    if len(pred_words) == 0 or len(true_words) == 0:
        return False
    
    # Calculate word overlap
    overlap = len(pred_words & true_words)
    total_unique = len(pred_words | true_words)
    
    if total_unique > 0:
        overlap_ratio = overlap / total_unique
        if overlap_ratio > 0.5:  # More than 50% word overlap
            return True
    
    # Check for numeric proximity
    try:
        pred_num = float(re.sub(r'[^\d.]', '', pred_norm))
        true_num = float(re.sub(r'[^\d.]', '', true_norm))
        if abs(pred_num - true_num) / max(abs(true_num), 1) < 0.1:  # Within 10%
            return True
    except (ValueError, TypeError):
        pass
    
    return False

