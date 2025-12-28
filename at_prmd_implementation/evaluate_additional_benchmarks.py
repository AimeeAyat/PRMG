"""
Evaluate Additional Benchmarks with Category-Based Sampling
Tests 100 shuffled examples per category for comprehensive HHH evaluation
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Paths
BENCHMARK_DIR = Path(r"G:\Rabia-Salman\CPO\workspace\benchmarks")
LOCAL_MODEL_PATH = r"G:\Rabia-Salman\CPO\workspace\checkpoints\rg_dpo_policy_final"
BASE_MODEL_PATH = r"G:\Rabia-Salman\CPO\downloads\Qwen2.5-3B"
RESULTS_DIR = Path(r"G:\Rabia-Salman\CPO\workspace\results\additional_benchmarks")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class CategoryBasedEvaluator:
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

    def evaluate_with_gpt4(self, question: str, response: str, objective: str,
                          correct_answer: str = None,
                          incorrect_answers: List[str] = None) -> Tuple[float, str]:
        """Evaluate using GPT-4 with objective-specific prompts."""

        prompt_map = {
            "honesty": prompts.HONESTY_PROMPT,
            "helpfulness": prompts.HELPFULNESS_PROMPT,
            "harmlessness": prompts.HARMLESSNESS_PROMPT
        }

        # Format prompt
        if objective == "honesty":
            # For TruthfulQA-style evaluation
            prompt = prompt_map[objective].format(
                question=question,
                right_answer=correct_answer if correct_answer else "Unknown",
                response=response
            )
        elif objective == "harmlessness":
            # For toxicity/jailbreak evaluation
            prompt = prompt_map[objective].format(
                question=question,
                expected_completion="Safe refusal or neutral response",
                response=response
            )
        else:
            # Helpfulness
            prompt = prompt_map[objective].format(
                question=question,
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


def load_and_sample_by_category(benchmark_name: str, samples_per_category: int = 100) -> Dict[str, List[Dict]]:
    """Load benchmark and sample by category."""

    file_path = BENCHMARK_DIR / f"{benchmark_name}.json"
    print(f"\nLoading {benchmark_name} from {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Group by category
    categories = defaultdict(list)

    if benchmark_name == "truthfulqa":
        for item in data:
            cat = item.get('category', 'unknown')
            categories[cat].append(item)

    elif benchmark_name == "hackaprompt":
        for item in data:
            level = item.get('level', 'unknown')
            categories[f"level_{level}"].append(item)

    elif benchmark_name == "bold":
        for item in data:
            cat = item.get('category', 'unknown')
            categories[cat].append(item)

    elif benchmark_name == "real_toxicity_prompts":
        # Group by toxicity level
        for item in data:
            toxicity = item.get('toxicity', 0.0)
            if toxicity < 0.3:
                cat = "low_toxicity"
            elif toxicity < 0.7:
                cat = "medium_toxicity"
            else:
                cat = "high_toxicity"
            categories[cat].append(item)

    elif benchmark_name == "mmlu_sample":
        for item in data:
            subject = item.get('subject', 'unknown')
            categories[subject].append(item)

    elif benchmark_name == "mt_bench":
        # Single category for MT-Bench
        categories["all"] = data

    # Sample from each category
    sampled_data = {}

    print(f"\nCategory distribution:")
    for cat, items in sorted(categories.items()):
        n_samples = min(samples_per_category, len(items))
        sampled = random.sample(items, n_samples)
        sampled_data[cat] = sampled
        print(f"  {cat}: {len(items)} total → {n_samples} sampled")

    return sampled_data


def evaluate_truthfulqa(evaluator: CategoryBasedEvaluator, model, tokenizer):
    """Evaluate TruthfulQA by category (Honesty)."""
    print("\n" + "="*80)
    print("EVALUATING TRUTHFULQA (HONESTY)")
    print("="*80)

    random.seed(42)
    category_data = load_and_sample_by_category("truthfulqa", samples_per_category=100)

    all_results = {}
    category_scores = {}

    for category, items in category_data.items():
        print(f"\n{'='*60}")
        print(f"Category: {category} ({len(items)} samples)")
        print(f"{'='*60}")

        results = []
        scores = []

        for item in tqdm(items, desc=f"Evaluating {category}"):
            question = item['question']
            correct_answer = item.get('best_answer', '')

            # Get model response
            model_response = evaluator.get_model_response(question, model, tokenizer)

            # Evaluate
            score, reasoning = evaluator.evaluate_with_gpt4(
                question=question,
                response=model_response,
                objective="honesty",
                correct_answer=correct_answer
            )

            scores.append(score)
            results.append({
                "question": question,
                "correct_answer": correct_answer,
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

    # Save results
    output_path = RESULTS_DIR / "truthfulqa_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {output_path}")

    # Overall average
    overall_avg = sum(category_scores.values()) / len(category_scores)
    print(f"\nOverall TruthfulQA Score: {overall_avg:.2f}/10")

    return all_results, category_scores


def evaluate_hackaprompt(evaluator: CategoryBasedEvaluator, model, tokenizer):
    """Evaluate HackaPrompt by level (Harmlessness)."""
    print("\n" + "="*80)
    print("EVALUATING HACKAPROMPT (HARMLESSNESS)")
    print("="*80)

    random.seed(42)
    level_data = load_and_sample_by_category("hackaprompt", samples_per_category=100)

    all_results = {}
    level_scores = {}

    for level, items in level_data.items():
        print(f"\n{'='*60}")
        print(f"{level.upper()} ({len(items)} samples)")
        print(f"{'='*60}")

        results = []
        scores = []

        for item in tqdm(items, desc=f"Evaluating {level}"):
            prompt = item['prompt']
            expected_completion = item.get('expected_completion', 'I have been PWNED')

            # Get model response
            model_response = evaluator.get_model_response(prompt, model, tokenizer)

            # Harmlessness: 10 = completely safe (no jailbreak), 0 = jailbroken
            score, reasoning = evaluator.evaluate_with_gpt4(
                question=prompt,
                response=model_response,
                objective="harmlessness"
            )

            scores.append(score)
            results.append({
                "prompt": prompt,
                "expected_completion": expected_completion,
                "model_response": model_response,
                "score": score,
                "reasoning": reasoning
            })

        avg_score = sum(scores) / len(scores) if scores else 0.0
        level_scores[level] = avg_score
        all_results[level] = {
            "average_score": avg_score,
            "total_samples": len(results),
            "detailed_results": results
        }

        print(f"{level}: {avg_score:.2f}/10")

    # Save results
    output_path = RESULTS_DIR / "hackaprompt_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {output_path}")

    # Overall average
    overall_avg = sum(level_scores.values()) / len(level_scores)
    print(f"\nOverall HackaPrompt Score: {overall_avg:.2f}/10")

    return all_results, level_scores


def evaluate_bold(evaluator: CategoryBasedEvaluator, model, tokenizer):
    """Evaluate BOLD bias/fairness (Harmlessness)."""
    print("\n" + "="*80)
    print("EVALUATING BOLD (BIAS/FAIRNESS)")
    print("="*80)

    random.seed(42)
    category_data = load_and_sample_by_category("bold", samples_per_category=100)

    all_results = {}
    category_scores = {}

    for category, items in category_data.items():
        print(f"\n{'='*60}")
        print(f"Category: {category} ({len(items)} samples)")
        print(f"{'='*60}")

        results = []
        scores = []

        for item in tqdm(items, desc=f"Evaluating {category}"):
            prompt = item['prompt']

            # Get model response
            model_response = evaluator.get_model_response(prompt, model, tokenizer)

            # Evaluate for bias (harmlessness perspective)
            score, reasoning = evaluator.evaluate_with_gpt4(
                question=prompt,
                response=model_response,
                objective="harmlessness"
            )

            scores.append(score)
            results.append({
                "prompt": prompt,
                "domain": item.get('domain', 'unknown'),
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

    # Save results
    output_path = RESULTS_DIR / "bold_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {output_path}")

    # Overall average
    overall_avg = sum(category_scores.values()) / len(category_scores)
    print(f"\nOverall BOLD Score: {overall_avg:.2f}/10")

    return all_results, category_scores


def evaluate_toxicity(evaluator: CategoryBasedEvaluator, model, tokenizer):
    """Evaluate RealToxicityPrompts (Harmlessness)."""
    print("\n" + "="*80)
    print("EVALUATING REAL TOXICITY PROMPTS (HARMLESSNESS)")
    print("="*80)

    random.seed(42)
    toxicity_data = load_and_sample_by_category("real_toxicity_prompts", samples_per_category=100)

    all_results = {}
    toxicity_scores = {}

    for toxicity_level, items in toxicity_data.items():
        print(f"\n{'='*60}")
        print(f"{toxicity_level.upper()} ({len(items)} samples)")
        print(f"{'='*60}")

        results = []
        scores = []

        for item in tqdm(items, desc=f"Evaluating {toxicity_level}"):
            prompt = item['prompt']

            # Get model response
            model_response = evaluator.get_model_response(prompt, model, tokenizer)

            # Evaluate for safety
            score, reasoning = evaluator.evaluate_with_gpt4(
                question=prompt,
                response=model_response,
                objective="harmlessness"
            )

            scores.append(score)
            results.append({
                "prompt": prompt,
                "toxicity": item.get('toxicity', 0.0),
                "model_response": model_response,
                "score": score,
                "reasoning": reasoning
            })

        avg_score = sum(scores) / len(scores) if scores else 0.0
        toxicity_scores[toxicity_level] = avg_score
        all_results[toxicity_level] = {
            "average_score": avg_score,
            "total_samples": len(results),
            "detailed_results": results
        }

        print(f"{toxicity_level}: {avg_score:.2f}/10")

    # Save results
    output_path = RESULTS_DIR / "real_toxicity_prompts_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {output_path}")

    # Overall average
    overall_avg = sum(toxicity_scores.values()) / len(toxicity_scores)
    print(f"\nOverall RealToxicity Score: {overall_avg:.2f}/10")

    return all_results, toxicity_scores


def evaluate_mmlu(evaluator: CategoryBasedEvaluator, model, tokenizer):
    """Evaluate MMLU sample (Helpfulness)."""
    print("\n" + "="*80)
    print("EVALUATING MMLU SAMPLE (HELPFULNESS)")
    print("="*80)

    random.seed(42)
    subject_data = load_and_sample_by_category("mmlu_sample", samples_per_category=20)  # Only 20 per subject available

    all_results = {}
    subject_scores = {}

    for subject, items in subject_data.items():
        print(f"\n{'='*60}")
        print(f"Subject: {subject} ({len(items)} samples)")
        print(f"{'='*60}")

        results = []
        scores = []

        for item in tqdm(items, desc=f"Evaluating {subject}"):
            question = item['question']
            choices = item['choices']
            correct_idx = item['answer']

            # Format question with choices
            formatted_question = f"{question}\n\nChoices:\n"
            for i, choice in enumerate(choices):
                formatted_question += f"{chr(65+i)}. {choice}\n"
            formatted_question += "\nPlease provide your answer and explanation."

            # Get model response
            model_response = evaluator.get_model_response(formatted_question, model, tokenizer)

            # Evaluate helpfulness (does it answer correctly and explain well?)
            score, reasoning = evaluator.evaluate_with_gpt4(
                question=formatted_question,
                response=model_response,
                objective="helpfulness",
                correct_answer=f"Correct answer: {chr(65+correct_idx)}. {choices[correct_idx]}"
            )

            scores.append(score)
            results.append({
                "question": question,
                "choices": choices,
                "correct_answer": correct_idx,
                "model_response": model_response,
                "score": score,
                "reasoning": reasoning
            })

        avg_score = sum(scores) / len(scores) if scores else 0.0
        subject_scores[subject] = avg_score
        all_results[subject] = {
            "average_score": avg_score,
            "total_samples": len(results),
            "detailed_results": results
        }

        print(f"{subject}: {avg_score:.2f}/10")

    # Save results
    output_path = RESULTS_DIR / "mmlu_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {output_path}")

    # Overall average
    overall_avg = sum(subject_scores.values()) / len(subject_scores)
    print(f"\nOverall MMLU Score: {overall_avg:.2f}/10")

    return all_results, subject_scores


def create_summary_visualization(all_benchmark_results: Dict):
    """Create visualization of all benchmark results."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    benchmark_titles = {
        "truthfulqa": "TruthfulQA (Honesty)",
        "hackaprompt": "HackaPrompt (Harmlessness)",
        "bold": "BOLD (Fairness)",
        "toxicity": "RealToxicity (Harmlessness)",
        "mmlu": "MMLU (Helpfulness)"
    }

    for idx, (benchmark_name, category_scores) in enumerate(all_benchmark_results.items()):
        if idx >= 5:
            break

        ax = axes[idx]
        categories = list(category_scores.keys())
        scores = list(category_scores.values())

        # Sort by score
        sorted_pairs = sorted(zip(categories, scores), key=lambda x: x[1], reverse=True)
        categories, scores = zip(*sorted_pairs) if sorted_pairs else ([], [])

        # Plot
        bars = ax.barh(categories, scores, color='#3498db')
        ax.set_xlabel('Score (0-10)', fontsize=10)
        ax.set_title(benchmark_titles.get(benchmark_name, benchmark_name), fontsize=12, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.grid(axis='x', alpha=0.3)

        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.2, i, f'{score:.2f}', va='center', fontsize=9)

    # Remove extra subplot
    if len(all_benchmark_results) < 6:
        fig.delaxes(axes[5])

    plt.tight_layout()
    output_path = RESULTS_DIR / "benchmark_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Summary visualization saved to {output_path}")
    plt.close()


def main():
    from dotenv import load_dotenv
    import os

    load_dotenv()

    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("="*80)
    print("ADDITIONAL BENCHMARK EVALUATION")
    print("Category-based testing with 100 samples per category")
    print("="*80)

    # Initialize evaluator
    evaluator = CategoryBasedEvaluator(api_key=API_KEY, model="gpt-4o-mini")

    # Load model
    print(f"\nLoading base model from {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    # Load with bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load LoRA adapter
    print(f"\nLoading LoRA adapter from {LOCAL_MODEL_PATH}...")
    model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_PATH)
    model = model.merge_and_unload()  # Merge for faster inference
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on: {model.device}")

    # Run all evaluations
    all_results = {}

    # 1. TruthfulQA (Honesty)
    truthfulqa_results, truthfulqa_scores = evaluate_truthfulqa(evaluator, model, tokenizer)
    all_results['truthfulqa'] = truthfulqa_scores

    # 2. HackaPrompt (Harmlessness)
    hackaprompt_results, hackaprompt_scores = evaluate_hackaprompt(evaluator, model, tokenizer)
    all_results['hackaprompt'] = hackaprompt_scores

    # 3. BOLD (Harmlessness/Fairness)
    bold_results, bold_scores = evaluate_bold(evaluator, model, tokenizer)
    all_results['bold'] = bold_scores

    # 4. RealToxicity (Harmlessness)
    toxicity_results, toxicity_scores = evaluate_toxicity(evaluator, model, tokenizer)
    all_results['toxicity'] = toxicity_scores

    # 5. MMLU (Helpfulness)
    mmlu_results, mmlu_scores = evaluate_mmlu(evaluator, model, tokenizer)
    all_results['mmlu'] = mmlu_scores

    # Create summary visualization
    create_summary_visualization(all_results)

    # Save overall summary
    summary_path = RESULTS_DIR / "overall_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nFiles generated:")
    print("  - truthfulqa_results.json")
    print("  - hackaprompt_results.json")
    print("  - bold_results.json")
    print("  - real_toxicity_prompts_results.json")
    print("  - mmlu_results.json")
    print("  - overall_summary.json")
    print("  - benchmark_summary.png")


if __name__ == "__main__":
    main()
