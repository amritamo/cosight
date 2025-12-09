# GAIA Benchmark Evaluation

This directory contains scripts for evaluating model accuracy on the GAIA benchmark dataset.

## Setup

1. Install required dependencies:
```bash
pip install datasets pandas huggingface_hub python-dotenv
```

2. Set up your environment variables in `.env`:
```bash
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

3. Create the output directory structure:
```bash
mkdir -p output/validation
```

## Usage

### Basic Usage

Run the evaluation script:
```bash
python scripts/evaluation/evaluate_gaia.py
```

### Quick Testing (3 Random Examples)

For quick testing, run on just 3 random examples:
```bash
python scripts/evaluation/evaluate_gaia.py --max-examples 3
```

### Dataset Exploration (No Results Required)

To explore the dataset without requiring result files:
```bash
# Explore full dataset
python scripts/evaluation/evaluate_gaia.py --explore-only

# Explore 3 random examples
python scripts/evaluation/evaluate_gaia.py --explore-only --max-examples 3
```

This mode will:
- Load and display sample questions from the dataset
- Show task distribution
- Not require any result files

### Generating Results

To actually evaluate your model, you need to generate result files first. You can use the helper script:

```bash
# Generate results for 3 random examples (quick test)
python scripts/evaluation/generate_results.py --max-examples 3

# Generate results for all examples (full evaluation)
python scripts/evaluation/generate_results.py
```

This will:
1. Load the GAIA dataset
2. Run your CoSight model on each question
3. Save results to `output/validation/cosight_results.jsonl`

Then run the evaluation:
```bash
python scripts/evaluation/evaluate_gaia.py --max-examples 3
```

### Command Line Options

```bash
python scripts/evaluation/evaluate_gaia.py \
    --output-dir output \
    --result-dir output/validation \
    --agent-name-col agent_name \
    --max-examples 3 \
    --random-seed 42
```

Options:
- `--output-dir`: Base output directory (default: "output")
- `--result-dir`: Specific directory containing result JSONL files (default: "output/validation")
- `--agent-name-col`: Column name for agent identifier in results (default: "agent_name")
- `--max-examples`: Maximum number of random examples to evaluate (default: None, uses all). Use 3 for quick testing.
- `--random-seed`: Random seed for sampling examples (default: 42)
- `--explore-only`: Only explore the dataset without requiring result files (default: False)

### Programmatic Usage

```python
from scripts.evaluation.evaluate_gaia import GAIAEvaluator

# Create evaluator (full dataset)
evaluator = GAIAEvaluator(output_dir="output")

# Or create evaluator with limited examples (for quick testing)
evaluator = GAIAEvaluator(output_dir="output", max_examples=3, random_seed=42)

# Explore dataset without results
eval_df = evaluator.explore_dataset()

# Run full evaluation (gracefully handles missing results)
result_df, summary = evaluator.run_full_evaluation(
    result_dir="output/validation",
    agent_name_col="agent_name",
    require_results=False  # Won't fail if no results found
)

# Or run steps individually
evaluator.load_dataset()
evaluator.load_results()
evaluator.merge_with_ground_truth()
evaluator.score_predictions()
evaluator.analyze_steps()
evaluator.analyze_tool_calls()
summary = evaluator.generate_summary()
```

## Input Format

The evaluation script expects result files in JSONL format with the following structure:

```json
{
  "question": "What is the capital of France?",
  "prediction": "Paris",
  "agent_name": "my_agent",
  "intermediate_steps": [
    {
      "llm_output": "I need to search for the capital of France...",
      "tool_calls": [...]
    }
  ]
}
```

Required fields:
- `question`: The question text (must match GAIA dataset)
- `prediction`: The model's final answer

Optional fields:
- `agent_name`: Identifier for the model/agent
- `intermediate_steps`: List of intermediate reasoning steps
- `start_time`, `end_time`: Timestamps (will be dropped)

## Output

The evaluation script generates:

1. **Consolidated answers file**: `output/validation/answers.jsonl`
   - All results merged with ground truth
   - Includes scoring columns: `is_correct`, `is_near_correct`

2. **Summary statistics**:
   - Average accuracy by agent
   - Breakdown by task type
   - Step count analysis
   - Tool call analysis

3. **Near misses**: Questions that were close but not exact matches

## Scoring

The evaluation uses two scoring metrics:

1. **is_correct**: Exact or near-exact match with ground truth
   - Normalizes answers (lowercase, remove punctuation)
   - Checks for exact match, substring match, or numeric equivalence

2. **is_near_correct**: Close but not exact matches
   - Checks for significant word overlap (>50%)
   - Checks for numeric proximity (within 10%)

## Customization

### Hard Questions

Edit `hard_questions.py` to add question IDs that are considered particularly difficult:

```python
HARD_QUESTIONS = [
    "question_123",
    "question_456",
]
```

### Scoring Function

Modify `gaia_scorer.py` to customize the scoring logic for your specific use case.

## Files

- `evaluate_gaia.py`: Main evaluation script
- `gaia_scorer.py`: Scoring functions for comparing predictions to ground truth
- `hard_questions.py`: List of hard question identifiers
- `README.md`: This file

