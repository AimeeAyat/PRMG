"""
Evaluate TruthfulQA (Honesty) - Category-based with 100 samples per category
Compares AT-PRMG fine-tuned model vs Qwen2.5-3B baseline
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Paths
BENCHMARK_PATH = Path(r"G:\Rabia-Salman\CPO\workspace\benchmarks\truthfulqa.json")
LOCAL_MODEL_PATH = r"G:\Rabia-Salman\CPO\workspace\checkpoints\rg_dpo_policy_final_old\merged_model"
BASE_MODEL_PATH = r"G:\Rabia-Salman\CPO\downloads\Qwen2.5-3B"
RESULTS_DIR = Path(r"G:\Rabia-Salman\CPO\workspace\results\benchmarks")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class TruthfulQAEvaluator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_model_response(self, question: str, model, tokenizer, max_new_tokens: int = 512) -> str:
        """Generate response from policy model."""
        messages = [{"role": "user", "content": question}]

        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"<s>[INST] {question} [/INST]"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def evaluate_with_gpt4(self, question: str, response: str,
                          correct_answer: str) -> Tuple[float, str]:
        """Evaluate using GPT-4 for honesty."""

        prompt = prompts.HONESTY_PROMPT.format(
            question=question,
            right_answer=correct_answer,
            response=response
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            evaluation = completion.choices[0].message.content

            # Extract score (0-10 scale)
            import re
            score_match = re.search(r"Score:\s*(\d+)", evaluation)
            score = float(score_match.group(1)) if score_match else 0.0

            reasoning_match = re.search(r"Reasoning:\s*(.+)", evaluation, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            return score, reasoning

        except Exception as e:
            print(f"Error evaluating: {e}")
            return 0.0, str(e)


def load_and_sample_truthfulqa(samples_per_category: int = 5) -> Dict[str, List[Dict]]:
    """Load TruthfulQA and sample by category."""

    print(f"\nLoading TruthfulQA from {BENCHMARK_PATH}")

    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Group by category
    categories = defaultdict(list)
    for item in data:
        cat = item.get('category', 'unknown')
        categories[cat].append(item)

    # Sample from each category
    sampled_data = {}
    random.seed(42)

    print(f"\nSampling {samples_per_category} examples per category:")
    for cat, items in sorted(categories.items()):
        n_samples = min(samples_per_category, len(items))
        sampled = random.sample(items, n_samples)
        sampled_data[cat] = sampled
        print(f"  {cat}: {len(items)} total -> {n_samples} sampled")

    return sampled_data


def evaluate_truthfulqa(evaluator: TruthfulQAEvaluator, model, tokenizer,
                       category_data: Dict[str, List[Dict]],
                       output_file: Path) -> Dict:
    """Evaluate TruthfulQA by category with incremental saving."""

    print("\n" + "="*80)
    print("EVALUATING TRUTHFULQA (HONESTY)")
    print("="*80)

    all_results = {}
    category_scores = {}

    # Load existing results if file exists
    if output_file.exists():
        print(f"\nLoading existing results from {output_file.name}")
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        category_scores = {cat: data['average_score'] for cat, data in all_results.items()}
        print(f"Found {len(all_results)} completed categories")

    for category, items in category_data.items():
        # Skip if already evaluated
        if category in all_results:
            print(f"\n[SKIPPED] {category} (already evaluated)")
            continue

        print(f"\n{'='*60}")
        print(f"Category: {category} ({len(items)} samples)")
        print(f"{'='*60}")

        results = []
        scores = []

        for item in tqdm(items, desc=f"Evaluating {category}"):
            question = item['question']
            correct_answer = item.get('best_answer', '')
            correct_answers = item.get('correct_answers', [])
            incorrect_answers = item.get('incorrect_answers', [])

            # Get model response
            model_response = evaluator.get_model_response(question, model, tokenizer)

            # Evaluate
            score, reasoning = evaluator.evaluate_with_gpt4(
                question=question,
                response=model_response,
                correct_answer=correct_answer
            )

            scores.append(score)
            results.append({
                "question": question,
                "category": category,
                "correct_answer": correct_answer,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
                "model_response": model_response,
                "score": score,
                "reasoning": reasoning
            })

        avg_score = sum(scores) / len(scores) if scores else 0.0
        category_scores[category] = avg_score
        all_results[category] = {
            "average_score": avg_score,
            "total_samples": len(results),
            "detailed_results": results
        }

        print(f"{category}: {avg_score:.2f}/10")

        # Save incrementally after each category
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Progress saved ({len(all_results)}/{len(category_data)} categories complete)")

    return all_results, category_scores


def create_comparison_visualization(finetuned_scores: Dict[str, float],
                                   baseline_scores: Dict[str, float]):
    """Create comparison bar charts for both models (split into multiple charts)."""

    # Get all categories (sorted by fine-tuned model score)
    all_categories = sorted(set(finetuned_scores.keys()) | set(baseline_scores.keys()))
    sorted_categories = sorted(all_categories,
                              key=lambda x: finetuned_scores.get(x, 0),
                              reverse=True)

    # Split into chunks of 8 categories each
    categories_per_chart = 8
    num_charts = (len(sorted_categories) + categories_per_chart - 1) // categories_per_chart

    for chart_idx in range(num_charts):
        start_idx = chart_idx * categories_per_chart
        end_idx = min((chart_idx + 1) * categories_per_chart, len(sorted_categories))

        chart_categories = sorted_categories[start_idx:end_idx]
        finetuned_vals = [finetuned_scores.get(cat, 0) for cat in chart_categories]
        baseline_vals = [baseline_scores.get(cat, 0) for cat in chart_categories]

        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, max(6, len(chart_categories) * 0.6)))

        y_pos = np.arange(len(chart_categories))
        width = 0.35

        ax.barh(y_pos - width/2, finetuned_vals, width,
                label='AT-PRMG (Fine-tuned)', color='#2ecc71')
        ax.barh(y_pos + width/2, baseline_vals, width,
                label='Qwen2.5-3B-Instruct', color='#3498db')

        ax.set_xlabel('Score (0-10)', fontsize=12)

        # Add chart number to title
        title = f'TruthfulQA: Model Comparison by Category ({chart_idx + 1}/{num_charts})'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(chart_categories, fontsize=10)
        ax.set_xlim(0, 10)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

        # Add score labels
        for i, (ft_val, bl_val) in enumerate(zip(finetuned_vals, baseline_vals)):
            ax.text(ft_val + 0.15, i - width/2, f'{ft_val:.2f}',
                    va='center', fontsize=9, color='#27ae60', fontweight='bold')
            ax.text(bl_val + 0.15, i + width/2, f'{bl_val:.2f}',
                    va='center', fontsize=9, color='#2980b9', fontweight='bold')

        plt.tight_layout()
        output_path = RESULTS_DIR / f"truthfulqa_comparison_part{chart_idx + 1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Chart {chart_idx + 1}/{num_charts} saved: {output_path.name}")
        plt.close()

    print(f"\nAll {num_charts} comparison charts saved")


def create_overall_comparison(finetuned_avg: float, baseline_avg: float):
    """Create simple bar chart comparing overall scores."""

    fig, ax = plt.subplots(figsize=(8, 6))

    models = ['AT-PRMG\n(Fine-tuned)', 'Qwen2.5-3B-Instruct\n(Baseline)']
    scores = [finetuned_avg, baseline_avg]
    colors = ['#2ecc71', '#3498db']

    bars = ax.bar(models, scores, color=colors, width=0.5)
    ax.set_ylabel('Average Score (0-10)', fontsize=12)
    ax.set_title('TruthfulQA: Overall Honesty Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=0.3)

    # Add score labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               f'{score:.2f}',
               ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add improvement indicator
    improvement = finetuned_avg - baseline_avg
    improvement_pct = (improvement / baseline_avg) * 100 if baseline_avg > 0 else 0

    if improvement > 0:
        ax.text(0.5, max(scores) * 0.5,
               f'+{improvement:.2f} ({improvement_pct:+.1f}%)\nimprovement',
               ha='center', fontsize=12, color='#27ae60', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='#27ae60', linewidth=2))
    elif improvement < 0:
        ax.text(0.5, max(scores) * 0.5,
               f'{improvement:.2f} ({improvement_pct:.1f}%)\nregression',
               ha='center', fontsize=12, color='#e74c3c', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='#e74c3c', linewidth=2))

    plt.tight_layout()
    output_path = RESULTS_DIR / "truthfulqa_overall_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nOverall comparison saved to {output_path}")
    plt.close()


def main():
    from dotenv import load_dotenv
    import os

    load_dotenv()

    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("="*80)
    print("TRUTHFULQA COMPARATIVE EVALUATION")
    print("Testing 100 samples per category")
    print("Comparing: AT-PRMG (Fine-tuned) vs Qwen2.5-3B (Baseline)")
    print("="*80)

    # Initialize evaluator
    evaluator = TruthfulQAEvaluator(api_key=API_KEY, model="gpt-4o-mini")

    # Load and sample data
    category_data = load_and_sample_truthfulqa(samples_per_category=100)

    # EVALUATE BASELINE MODEL (Qwen2.5-3B)
    print("\n" + "="*80)
    print("[1/2] EVALUATING BASELINE: Qwen2.5-3B-Instruct")
    print("="*80)

    print(f"\nLoading baseline model from {BASE_MODEL_PATH}...")
    baseline_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    baseline_model.eval()

    if baseline_tokenizer.pad_token is None:
        baseline_tokenizer.pad_token = baseline_tokenizer.eos_token

    print(f"Baseline model loaded on: {baseline_model.device}")

    baseline_output_path = RESULTS_DIR / "truthfulqa_baseline_results.json"
    baseline_results, baseline_scores = evaluate_truthfulqa(
        evaluator, baseline_model, baseline_tokenizer, category_data,
        output_file=baseline_output_path
    )

    baseline_avg = sum(baseline_scores.values()) / len(baseline_scores)
    print(f"\nBaseline Overall Average: {baseline_avg:.2f}/10")

    del baseline_model
    torch.cuda.empty_cache()

    # EVALUATE FINE-TUNED MODEL (AT-PRMG)
    print("\n" + "="*80)
    print("[2/2] EVALUATING FINE-TUNED MODEL: AT-PRMG")
    print("="*80)

    print(f"\nLoading merged model from {LOCAL_MODEL_PATH}...")
    finetuned_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    finetuned_model.eval()

    if finetuned_tokenizer.pad_token is None:
        finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token

    print(f"Fine-tuned model loaded on: {finetuned_model.device}")

    finetuned_output_path = RESULTS_DIR / "truthfulqa_finetuned_results1.json"
    finetuned_results, finetuned_scores = evaluate_truthfulqa(
        evaluator, finetuned_model, finetuned_tokenizer, category_data,
        output_file=finetuned_output_path
    )

    finetuned_avg = sum(finetuned_scores.values()) / len(finetuned_scores)
    print(f"\nFine-tuned Overall Average: {finetuned_avg:.2f}/10")

    # CREATE COMPARISON VISUALIZATIONS
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)

    create_comparison_visualization(finetuned_scores, baseline_scores)
    create_overall_comparison(finetuned_avg, baseline_avg)

    # PRINT SUMMARY
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80)

    print(f"\nOVERALL SCORES:")
    print(f"  AT-PRMG (Fine-tuned):  {finetuned_avg:.2f}/10")
    print(f"  Qwen2.5-3B (Baseline): {baseline_avg:.2f}/10")

    improvement = finetuned_avg - baseline_avg
    improvement_pct = (improvement / baseline_avg) * 100 if baseline_avg > 0 else 0

    if improvement > 0:
        print(f"  Improvement: +{improvement:.2f} ({improvement_pct:+.1f}%)")
    elif improvement < 0:
        print(f"  Regression: {improvement:.2f} ({improvement_pct:.1f}%)")

    print(f"\nTOP 5 CATEGORIES (Fine-tuned):")
    for cat, score in sorted(finetuned_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        baseline_score = baseline_scores.get(cat, 0)
        diff = score - baseline_score
        print(f"  {cat}: {score:.2f} (baseline: {baseline_score:.2f}, diff: {diff:+.2f})")

    print(f"\nLARGEST IMPROVEMENTS:")
    improvements = {cat: finetuned_scores[cat] - baseline_scores.get(cat, 0)
                   for cat in finetuned_scores.keys()}
    for cat, diff in sorted(improvements.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {cat}: {diff:+.2f} ({finetuned_scores[cat]:.2f} vs {baseline_scores.get(cat, 0):.2f})")

    print(f"\nLARGEST REGRESSIONS:")
    for cat, diff in sorted(improvements.items(), key=lambda x: x[1])[:5]:
        print(f"  {cat}: {diff:+.2f} ({finetuned_scores[cat]:.2f} vs {baseline_scores.get(cat, 0):.2f})")

    print("\n" + "="*80)
    print("FILES GENERATED:")
    print("="*80)
    print(f"  - truthfulqa_baseline_results.json")
    print(f"  - truthfulqa_finetuned_results1.json")
    print(f"  - truthfulqa_comparison_part1.png ... part5.png (5 charts, 8 categories each)")
    print(f"  - truthfulqa_overall_comparison.png")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
