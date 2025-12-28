"""
Download Benchmark Datasets for HHH Evaluation
Downloads: TruthfulQA, MT-Bench, HackaPrompt, and additional benchmarks
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Output directory
BENCHMARK_DIR = Path(r"G:\Rabia-Salman\CPO\workspace\benchmarks")
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("DOWNLOADING HHH EVALUATION BENCHMARKS")
print("="*80)

# ============================================================================
# 1. TruthfulQA (Honesty)
# ============================================================================
print("\n[1/6] Downloading TruthfulQA (Honesty)...")
try:
    truthfulqa = load_dataset("truthful_qa", "generation", split="validation")

    truthfulqa_data = []
    for item in tqdm(truthfulqa, desc="Processing TruthfulQA"):
        truthfulqa_data.append({
            "question": item["question"],
            "category": item["category"],
            "best_answer": item["best_answer"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
            "source": item.get("source", "unknown")
        })

    output_path = BENCHMARK_DIR / "truthfulqa.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(truthfulqa_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(truthfulqa_data)} questions to {output_path}")

    # Print category distribution
    categories = {}
    for item in truthfulqa_data:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n  Categories ({len(categories)} total):")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        print(f"    - {cat}: {count} questions")
    if len(categories) > 10:
        print(f"    ... and {len(categories)-10} more")

except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# 2. MT-Bench (Helpfulness)
# ============================================================================
print("\n[2/6] Downloading MT-Bench (Helpfulness)...")
try:
    # MT-Bench from FastChat
    mtbench = load_dataset("lmsys/mt_bench_human_judgments", split="train")

    # Extract unique questions (multi-turn)
    mt_data = []
    seen_questions = set()

    for item in tqdm(mtbench, desc="Processing MT-Bench"):
        question_id = item.get("question_id")
        if question_id not in seen_questions:
            mt_data.append({
                "question_id": question_id,
                "category": item.get("category", "unknown"),
                "turns": item.get("turns", []),
                "is_multi_turn": len(item.get("turns", [])) > 1
            })
            seen_questions.add(question_id)

    output_path = BENCHMARK_DIR / "mt_bench.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mt_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(mt_data)} questions to {output_path}")

except Exception as e:
    print(f"  ✗ Error downloading MT-Bench: {e}")
    print("  → Using fallback: Creating minimal MT-Bench dataset...")

    # Fallback: Create minimal MT-Bench
    mt_data = [
        {
            "question_id": 1,
            "category": "writing",
            "turns": [
                "Compose an engaging travel blog post about a recent trip to Hawaii.",
                "Rewrite your previous response in a more formal tone."
            ],
            "is_multi_turn": True
        },
        {
            "question_id": 2,
            "category": "reasoning",
            "turns": [
                "If a train travels 120 miles in 2 hours, what is its average speed?",
                "How long would it take to travel 300 miles at the same speed?"
            ],
            "is_multi_turn": True
        }
    ]

    output_path = BENCHMARK_DIR / "mt_bench.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mt_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Created fallback MT-Bench with {len(mt_data)} questions")

# ============================================================================
# 3. HackaPrompt (Harmlessness)
# ============================================================================
print("\n[3/6] Downloading HackaPrompt (Harmlessness)...")
try:
    hackaprompt = load_dataset("hackaprompt/hackaprompt-dataset", split="train")

    hacka_data = []
    for item in tqdm(hackaprompt, desc="Processing HackaPrompt"):
        hacka_data.append({
            "prompt": item["user_input"],
            "level": item["level"],
            "expected_completion": "I have been PWNED",  # Standard target
            "model": item.get("model", "unknown")
        })

    output_path = BENCHMARK_DIR / "hackaprompt.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hacka_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(hacka_data)} prompts to {output_path}")

    # Print level distribution
    levels = {}
    for item in hacka_data:
        level = item['level']
        levels[level] = levels.get(level, 0) + 1

    print(f"\n  Levels:")
    for level in sorted(levels.keys()):
        print(f"    - Level {level}: {levels[level]} prompts")

except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# 4. BOLD (Bias/Fairness for Harmlessness)
# ============================================================================
print("\n[4/6] Downloading BOLD (Bias/Fairness)...")
try:
    bold = load_dataset("AlexaAI/bold", split="train")

    bold_data = []
    for item in tqdm(list(bold)[:1000], desc="Processing BOLD"):  # Limit to 1000
        bold_data.append({
            "prompt": item["prompts"][0] if item["prompts"] else "",
            "category": item["category"],
            "domain": item.get("domain", "unknown")
        })

    output_path = BENCHMARK_DIR / "bold.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bold_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(bold_data)} prompts to {output_path}")

except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# 5. RealToxicityPrompts (Harmlessness)
# ============================================================================
print("\n[5/6] Downloading RealToxicityPrompts (Harmlessness)...")
try:
    toxicity = load_dataset("allenai/real-toxicity-prompts", split="train")

    toxicity_data = []
    for item in tqdm(list(toxicity)[:1000], desc="Processing RealToxicity"):
        if item.get("prompt") and item["prompt"].get("text"):
            toxicity_data.append({
                "prompt": item["prompt"]["text"],
                "toxicity": item["prompt"].get("toxicity", 0.0),
                "continuation": item.get("continuation", {}).get("text", "")
            })

    output_path = BENCHMARK_DIR / "real_toxicity_prompts.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(toxicity_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(toxicity_data)} prompts to {output_path}")

except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# 6. MMLU (General Knowledge for Helpfulness)
# ============================================================================
print("\n[6/6] Downloading MMLU subset (Helpfulness)...")
try:
    # Sample from a few MMLU subjects
    subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge"]
    mmlu_data = []

    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            for item in list(dataset)[:20]:  # 20 per subject
                mmlu_data.append({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "subject": subject
                })
        except:
            continue

    output_path = BENCHMARK_DIR / "mmlu_sample.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mmlu_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(mmlu_data)} questions to {output_path}")

except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)
print(f"\nBenchmarks saved to: {BENCHMARK_DIR}\n")
print("Files:")
for f in sorted(BENCHMARK_DIR.glob("*.json")):
    size_kb = f.stat().st_size / 1024
    print(f"  ✓ {f.name:30s} ({size_kb:>8.1f} KB)")

print("\n" + "="*80)
print("Next step: Run evaluate_additional_benchmarks.py")
print("="*80)
