"""
GAIA Benchmark Evaluation Script
Evaluates model accuracy on the GAIA benchmark dataset.
"""
import os
import glob
import re
import random
from collections import Counter
from typing import Dict, List, Optional

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.evaluation.gaia_scorer import question_scorer, check_close_call
from scripts.evaluation.hard_questions import HARD_QUESTIONS

# Load environment variables
load_dotenv(override=True)

# Login to HuggingFace if token is available
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    try:
        login(hf_token)
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace: {e}")

# Set pandas display options
pd.set_option("max_colwidth", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class GAIAEvaluator:
    """Evaluator for GAIA benchmark."""
    
    def __init__(self, output_dir: str = "output", max_examples: Optional[int] = None, random_seed: int = 42):
        self.output_dir = output_dir
        self.max_examples = max_examples
        self.random_seed = random_seed
        self.eval_df: Optional[pd.DataFrame] = None
        self.result_df: Optional[pd.DataFrame] = None
        self.selected_questions: Optional[List[str]] = None

    def load_dataset(self, dataset_name: str = "gaia-benchmark/GAIA", split: str = "2023_all"):
        """Load the GAIA benchmark dataset."""
        print(f"Loading GAIA dataset: {dataset_name}/{split}")
        try:
            # Load dataset - split is a configuration name, not a split name
            dataset_dict = datasets.load_dataset(dataset_name, split)
            # Get the validation split
            eval_ds = dataset_dict["validation"]
            eval_ds = eval_ds.rename_columns(
                {"Question": "question", "Final answer": "true_answer", "Level": "task"}
            )
            self.eval_df = pd.DataFrame(eval_ds)
            
            # Sample random examples if max_examples is set
            if self.max_examples is not None and self.max_examples > 0:
                if len(self.eval_df) > self.max_examples:
                    print(f"\nSampling {self.max_examples} random examples from {len(self.eval_df)} total questions")
                    random.seed(self.random_seed)
                    sampled_indices = random.sample(range(len(self.eval_df)), self.max_examples)
                    self.eval_df = self.eval_df.iloc[sampled_indices].reset_index(drop=True)
                    self.selected_questions = self.eval_df["question"].tolist()
                    print(f"Selected questions: {self.selected_questions}")
                else:
                    print(f"\nDataset has {len(self.eval_df)} questions, using all of them")
                    self.selected_questions = self.eval_df["question"].tolist()
            
            print(f"Loaded {len(self.eval_df)} questions")
            print("\nTask distribution:")
            print(pd.Series(self.eval_df["task"]).value_counts())
            return self.eval_df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Note: Make sure you have set HUGGINGFACEHUB_API_TOKEN in your .env file")
            raise
    
    def load_results(self, result_dir: Optional[str] = None, required: bool = True) -> pd.DataFrame:
        """Load result files from the output directory."""
        if result_dir is None:
            result_dir = f"{self.output_dir}/validation"
        
        print(f"\nLoading results from: {result_dir}")
        
        # Find all JSONL files except answers.jsonl
        result_files = [
            f for f in glob.glob(f"{result_dir}/*.jsonl")
            if "answers.jsonl" not in f
        ]
        
        if not result_files:
            if required:
                raise FileNotFoundError(
                    f"No result files found in {result_dir}\n"
                    f"Please ensure you have result JSONL files in the following format:\n"
                    f"  - Each file should contain JSON objects with 'question' and 'prediction' fields\n"
                    f"  - Place result files in: {result_dir}\n"
                    f"  - Example: {result_dir}/agent_results.jsonl"
                )
            else:
                print(f"⚠️  No result files found in {result_dir}")
                print("   Creating empty results DataFrame for dataset exploration mode")
                return pd.DataFrame()
        
        print(f"Found {len(result_files)} result files")
        
        # Concatenate all result files
        result_dfs = []
        for f in result_files:
            try:
                df = pd.read_json(f, lines=True)
                result_dfs.append(df)
                print(f"  Loaded {len(df)} results from {os.path.basename(f)}")
            except Exception as e:
                print(f"  Warning: Could not load {f}: {e}")
        
        if not result_dfs:
            raise ValueError("No valid result files found")
        
        self.result_df = pd.concat(result_dfs, ignore_index=True)
        
        # Drop timestamp columns if they exist
        cols_to_drop = ["start_time", "end_time"]
        existing_cols_to_drop = [col for col in cols_to_drop if col in self.result_df.columns]
        if existing_cols_to_drop:
            self.result_df = self.result_df.drop(columns=existing_cols_to_drop)
        
        # Filter to selected questions if we're running on a subset
        if self.selected_questions is not None:
            original_count = len(self.result_df)
            self.result_df = self.result_df[self.result_df["question"].isin(self.selected_questions)]
            filtered_count = len(self.result_df)
            print(f"Filtered results to {filtered_count} matching selected questions (from {original_count} total)")
            
            # Check if we have all selected questions
            missing = set(self.selected_questions) - set(self.result_df["question"].unique())
            if missing:
                print(f"Warning: Missing results for {len(missing)} selected questions")
        
        print(f"Total results loaded: {len(self.result_df)}")
        return self.result_df
    
    def merge_with_ground_truth(self):
        """Merge results with ground truth answers."""
        if self.eval_df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        if self.result_df is None:
            raise ValueError("Results not loaded. Call load_results() first.")
        
        # Merge on question
        self.result_df = self.result_df.merge(
            self.eval_df[["question", "true_answer", "task", "file_name"]],
            on="question",
            how="left"
        )
        
        print(f"\nMerged results with ground truth: {len(self.result_df)} rows")
    
    def score_predictions(self):
        """Score all predictions against ground truth."""
        if self.result_df is None:
            raise ValueError("Results not loaded. Call load_results() first.")
        
        print("\nScoring predictions...")
        
        # Score correctness
        self.result_df["is_correct"] = self.result_df.apply(
            lambda x: question_scorer(x["prediction"], x["true_answer"]),
            axis=1
        )
        
        # Check for close calls
        self.result_df["is_near_correct"] = self.result_df.apply(
            lambda x: check_close_call(x["prediction"], x["true_answer"], x["is_correct"]),
            axis=1
        )
        
        correct_count = self.result_df["is_correct"].sum()
        near_correct_count = self.result_df["is_near_correct"].sum()
        
        print(f"Correct: {correct_count}/{len(self.result_df)} ({100*correct_count/len(self.result_df):.2f}%)")
        print(f"Near correct: {near_correct_count}/{len(self.result_df)} ({100*near_correct_count/len(self.result_df):.2f}%)")
    
    def analyze_steps(self):
        """Analyze intermediate steps."""
        if self.result_df is None:
            raise ValueError("Results not loaded.")
        
        if "intermediate_steps" not in self.result_df.columns:
            print("Warning: No intermediate_steps column found")
            return
        
        print("\nAnalyzing intermediate steps...")
        
        # Count steps
        self.result_df["count_steps"] = self.result_df["intermediate_steps"].apply(len)
        avg_steps = self.result_df["count_steps"].mean()
        print(f"Average steps per question: {avg_steps:.2f}")
    
    def find_attachment_type(self):
        """Find attachment type for each question."""
        if self.eval_df is None:
            raise ValueError("Dataset not loaded.")
        if self.result_df is None:
            raise ValueError("Results not loaded.")
        
        print("\nFinding attachment types...")
        
        def find_attachment(question):
            matches = self.eval_df.loc[
                self.eval_df["question"].apply(lambda x: x in question), "file_name"
            ]
            
            if len(matches) == 0:
                return "Not found"
            
            file_path = matches.values[0]
            
            if isinstance(file_path, str) and len(file_path) > 0:
                return file_path.split(".")[-1]
            else:
                return "None"
        
        self.result_df["attachment_type"] = self.result_df["question"].apply(find_attachment)
        
        print("Attachment type distribution:")
        print(self.result_df["attachment_type"].value_counts())
    
    def extract_tool_calls(self, code: str) -> Counter:
        """Extract tool calls from code string."""
        regex = r"\b(\w+)\("
        function_calls = [el for el in re.findall(regex, code) if el.islower()]
        return Counter(function_calls)
    
    def analyze_tool_calls(self):
        """Analyze tool calls from intermediate steps."""
        if self.result_df is None:
            raise ValueError("Results not loaded.")
        
        if "intermediate_steps" not in self.result_df.columns:
            print("Warning: No intermediate_steps column found")
            return
        
        print("\nAnalyzing tool calls...")
        
        def sum_tool_calls(steps):
            total_count = Counter()
            for step in steps:
                if "llm_output" in step:
                    total_count += self.extract_tool_calls(step["llm_output"])
            return total_count
        
        self.result_df["tool_calls"] = self.result_df["intermediate_steps"].apply(sum_tool_calls)
        
        # Aggregate tool call statistics
        all_tools = Counter()
        for tool_calls in self.result_df["tool_calls"]:
            all_tools += tool_calls
        
        print("Most common tool calls:")
        for tool, count in all_tools.most_common(10):
            print(f"  {tool}: {count}")
    
    def save_answers(self, output_file: Optional[str] = None):
        """Save consolidated answers to JSONL file."""
        if self.result_df is None:
            raise ValueError("Results not loaded.")
        
        if output_file is None:
            output_file = f"{self.output_dir}/validation/answers.jsonl"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Drop tool_calls column if it exists (not JSON serializable directly)
        save_df = self.result_df.copy()
        if "tool_calls" in save_df.columns:
            save_df = save_df.drop(columns=["tool_calls"])
        
        save_df.to_json(output_file, lines=True, orient="records")
        print(f"\nSaved answers to: {output_file}")
    
    def generate_summary(self, agent_name_col: str = "agent_name") -> pd.DataFrame:
        """Generate summary statistics by agent and task."""
        if self.result_df is None:
            raise ValueError("Results not loaded.")
        
        if agent_name_col not in self.result_df.columns:
            print(f"Warning: Column '{agent_name_col}' not found. Using all results.")
            agent_name_col = None
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Overall average score
        if agent_name_col:
            print("\nAverage score by agent:")
            avg_scores = self.result_df.groupby(agent_name_col)[["is_correct"]].mean().round(3)
            print(avg_scores)
        
        # Detailed breakdown by agent and task
        if agent_name_col and "task" in self.result_df.columns:
            print("\nDetailed breakdown by agent and task:")
            summary = self.result_df.groupby([agent_name_col, "task"]).agg({
                "is_correct": "mean",
                "is_near_correct": "mean",
                "count_steps": "mean",
                "question": "count",
            }).rename(columns={"question": "count"})
            print(summary)
        
        # Task-level summary
        if "task" in self.result_df.columns:
            print("\nSummary by task:")
            task_summary = self.result_df.groupby("task").agg({
                "is_correct": "mean",
                "is_near_correct": "mean",
                "count_steps": "mean",
                "question": "count",
            }).rename(columns={"question": "count"})
            print(task_summary)
        
        return summary if agent_name_col and "task" in self.result_df.columns else None
    
    def find_near_misses(self, agent_name_col: str = "agent_name"):
        """Find questions that were nearly correct but not exact."""
        if self.result_df is None:
            raise ValueError("Results not loaded.")
        
        near_misses = self.result_df.loc[
            (self.result_df["is_correct"] == False) & 
            (self.result_df["is_near_correct"] == True)
        ]
        
        if len(near_misses) > 0:
            print(f"\nFound {len(near_misses)} near misses:")
            cols = ["question", "prediction", "true_answer"]
            if agent_name_col in near_misses.columns:
                cols.insert(0, agent_name_col)
            print(near_misses[cols].head(10))
        
        return near_misses
    
    def explore_dataset(self):
        """Load and display dataset without requiring results."""
        print("="*80)
        print("GAIA BENCHMARK DATASET EXPLORATION")
        if self.max_examples is not None:
            print(f"⚠️  VIEWING {self.max_examples} RANDOM EXAMPLES")
        print("="*80)
        
        # Load dataset
        self.load_dataset()
        
        print("\n" + "="*80)
        print("DATASET PREVIEW")
        print("="*80)
        print(f"\nTotal questions: {len(self.eval_df)}")
        
        print("\nSample questions:")
        for idx, row in self.eval_df.head(10).iterrows():
            question_preview = row['question'][:150] + "..." if len(row['question']) > 150 else row['question']
            answer_preview = row['true_answer'][:100] + "..." if len(str(row['true_answer'])) > 100 else str(row['true_answer'])
            print(f"\n{idx + 1}. Task: {row['task']}")
            print(f"   Question: {question_preview}")
            print(f"   Answer: {answer_preview}")
        
        print("\n" + "="*80)
        print("Task distribution:")
        print(pd.Series(self.eval_df["task"]).value_counts())
        print("="*80)
        
        return self.eval_df
    
    def run_full_evaluation(self, result_dir: Optional[str] = None, 
                       agent_name_col: str = "agent_name",
                       require_results: bool = True):
        """Run the complete evaluation pipeline."""
        print("="*80)
        print("GAIA BENCHMARK EVALUATION")
        if self.max_examples is not None:
            print(f"⚠️  RUNNING ON {self.max_examples} RANDOM EXAMPLES (TEST MODE)")
        print("="*80)
        
        # Load dataset
        self.load_dataset()
        
        # Load results (optional if just exploring dataset)
        try:
            self.load_results(result_dir, required=require_results)
        except FileNotFoundError as e:
            if require_results:
                raise
            else:
                print(f"\n⚠️  {str(e).split(chr(10))[0]}")
                print("   Running in dataset exploration mode (no results to evaluate)")
                return self.explore_dataset(), None
        
        if self.result_df is None or len(self.result_df) == 0:
            print("\n⚠️  No results loaded. Running in dataset exploration mode.")
            return self.explore_dataset(), None
        
        # Merge with ground truth
        self.merge_with_ground_truth()
        
        # Score predictions
        self.score_predictions()
        
        # Analyze steps
        self.analyze_steps()
        
        # Find attachment types
        self.find_attachment_type()
        
        # Analyze tool calls
        self.analyze_tool_calls()
        
        # Save consolidated answers
        self.save_answers()
        
        # Generate summary
        summary = self.generate_summary(agent_name_col)
        
        # Find near misses
        self.find_near_misses(agent_name_col)
        
        print("\n" + "="*80)
        print("Evaluation complete!")
        print("="*80)
        
        return self.result_df, summary


def main():
    """Main entry point for the evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate models on GAIA benchmark")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory containing results")
    parser.add_argument("--result-dir", type=str, default=None,
                       help="Specific directory containing result files (default: output/validation)")
    parser.add_argument("--agent-name-col", type=str, default="agent_name",
                       help="Column name for agent identifier")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum number of random examples to evaluate (default: all). Use 3 for quick testing.")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for sampling examples (default: 42)")
    parser.add_argument("--explore-only", action="store_true",
                       help="Only explore the dataset without requiring result files")
    
    args = parser.parse_args()
    
    evaluator = GAIAEvaluator(
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        random_seed=args.random_seed
    )
    
    if args.explore_only:
        # Just explore the dataset
        evaluator.explore_dataset()
    else:
        # Try to run full evaluation, but don't require results
        evaluator.run_full_evaluation(
            result_dir=args.result_dir,
            agent_name_col=args.agent_name_col,
            require_results=False  # Don't fail if no results found
        )


if __name__ == "__main__":
    main()

