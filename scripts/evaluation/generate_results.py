"""
Helper script to generate result files for GAIA evaluation.
This script runs your model on GAIA questions and saves results in the expected format.
"""
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login

# Import your CoSight system
from CoSight import CoSight
from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision, llm_for_draft, llm_for_verifier

load_dotenv(override=True)

# Login to HuggingFace if needed
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    try:
        login(hf_token)
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace: {e}")


def run_model_on_question(cosight: CoSight, question: str) -> Dict:
    """
    Run your model on a single question and return the result.
    
    Args:
        cosight: Your CoSight instance
        question: The question to answer
        
    Returns:
        Dictionary with question, prediction, and metadata
    """
    try:
        # Run your model
        result = cosight.execute(question, output_format="")
        
        return {
            "question": question,
            "prediction": result if result else "",
            "agent_name": "cosight_agent",
            "intermediate_steps": []  # Add if you want to track steps
        }
    except Exception as e:
        print(f"Error processing question: {e}")
        return {
            "question": question,
            "prediction": f"Error: {str(e)}",
            "agent_name": "cosight_agent",
            "intermediate_steps": []
        }


def generate_results(
    output_dir: str = "output/validation",
    max_examples: Optional[int] = None,
    random_seed: int = 42,
    dataset_name: str = "gaia-benchmark/GAIA",
    split: str = "2023_all"
):
    """
    Generate result files by running your model on GAIA questions.
    
    Args:
        output_dir: Directory to save result files
        max_examples: Maximum number of examples to process (None for all)
        random_seed: Random seed for sampling
        dataset_name: GAIA dataset name
        split: Dataset split/configuration
    """
    print("="*80)
    print("GENERATING RESULTS FOR GAIA EVALUATION")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading GAIA dataset: {dataset_name}/{split}")
    try:
        dataset_dict = datasets.load_dataset(dataset_name, split)
        eval_ds = dataset_dict["validation"]
        eval_ds = eval_ds.rename_columns(
            {"Question": "question", "Final answer": "true_answer", "Level": "task"}
        )
        eval_df = pd.DataFrame(eval_ds)
        
        # Sample if needed
        if max_examples is not None and max_examples > 0:
            if len(eval_df) > max_examples:
                import random
                random.seed(random_seed)
                sampled_indices = random.sample(range(len(eval_df)), max_examples)
                eval_df = eval_df.iloc[sampled_indices].reset_index(drop=True)
                print(f"Sampled {max_examples} random examples")
        
        print(f"Loaded {len(eval_df)} questions")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Initialize your model
    print("\nInitializing CoSight model...")
    try:
        # Create a workspace for this run
        import time
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        work_space_path = os.path.join("work_space", f"gaia_eval_{timestamp}")
        os.makedirs(work_space_path, exist_ok=True)
        
        cosight = CoSight(
            llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision,
            work_space_path=work_space_path,
            draft_llm=llm_for_draft,
            verifier_llm=llm_for_verifier
        )
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise
    
    # Process questions
    print(f"\nProcessing {len(eval_df)} questions...")
    results = []
    
    for idx, row in eval_df.iterrows():
        question = row['question']
        print(f"\n[{idx + 1}/{len(eval_df)}] Processing question...")
        print(f"Question: {question[:100]}...")
        
        result = run_model_on_question(cosight, question)
        results.append(result)
        
        print(f"Prediction: {result['prediction'][:100]}...")
    
    # Save results to JSONL file
    output_file = os.path.join(output_dir, "cosight_results.jsonl")
    print(f"\nSaving results to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(results)} results to {output_file}")
    print(f"\nYou can now run evaluation with:")
    print(f"  python scripts/evaluation/evaluate_gaia.py --max-examples {max_examples or 'all'}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate result files for GAIA evaluation")
    parser.add_argument("--output-dir", type=str, default="output/validation",
                       help="Directory to save result files")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum number of examples to process (default: all). Use 3 for quick testing.")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for sampling examples")
    
    args = parser.parse_args()
    
    generate_results(
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()

