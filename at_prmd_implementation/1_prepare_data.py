# AT-PRMD Data Preparation: Merge PKU-SafeRLHF + UltraFeedback + UltraSafety
#
# Anti-Skewness Strategy:
# 1. Shuffle all datasets with fixed seed BEFORE sampling
# 2. Ensures representative sampling across topics, difficulties, collection times
# 3. Prevents bias from sequential ordering in source datasets
# 4. Reproducible with RANDOM_SEED=42

import json
import os
from datasets import load_dataset
import torch

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Random seed for reproducibility
RANDOM_SEED = 42

# Windows paths
OUTPUT_DIR = r"g:\Rabia-Salman\CPO\workspace\data\at_prmd"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading datasets with anti-skewness shuffling...")
print("\nPKU-SafeRLHF (73,907 total)...")
pku_data = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
print(f"  Loaded {len(pku_data)} examples, shuffling with seed={RANDOM_SEED}...")
pku_data = pku_data.shuffle(seed=RANDOM_SEED)  # Shuffle to avoid skewness from sequential sampling

print("\nUltraFeedback (63,967 total)...")
ultra_data = load_dataset("openbmb/UltraFeedback", split="train")
print(f"  Loaded {len(ultra_data)} examples, shuffling with seed={RANDOM_SEED}...")
ultra_data = ultra_data.shuffle(seed=RANDOM_SEED)  # Shuffle to avoid skewness

print("\nUltraSafety (3,000 total)...")
ultrasafety_data = load_dataset("openbmb/UltraSafety", split="train")
print(f"  Loaded {len(ultrasafety_data)} examples, shuffling with seed={RANDOM_SEED}...")
ultrasafety_data = ultrasafety_data.shuffle(seed=RANDOM_SEED)  # Shuffle for consistency

print("\nAnthropic HH-RLHF Harmless...")
hh_harmless_data = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
print(f"  Loaded {len(hh_harmless_data)} examples, shuffling with seed={RANDOM_SEED}...")
hh_harmless_data = hh_harmless_data.shuffle(seed=RANDOM_SEED)

print("\n[OK] All datasets loaded and shuffled - ready for representative sampling")

# Prepare harmless dataset: Anthropic HH-RLHF + PKU (clear+ambiguous) + UltraSafety
print("\nPreparing HARMLESS dataset...")
harmless_pairs = []

# 1. Anthropic HH-RLHF (40k target)
print("\n[1/4] Adding Anthropic HH-RLHF harmless examples...")
for idx, example in enumerate(hh_harmless_data):
    if idx >= 40000:
        break

    harmless_pairs.append({
        "prompt": example['chosen'].split('\n\nAssistant:')[0].replace('Human: ', '').strip(),
        "chosen": example['chosen'].split('\n\nAssistant:')[-1].strip() if '\n\nAssistant:' in example['chosen'] else example['chosen'],
        "rejected": example['rejected'].split('\n\nAssistant:')[-1].strip() if '\n\nAssistant:' in example['rejected'] else example['rejected'],
        "objective": "harmless",
        "source": "hh-rlhf"
    })

    if (idx+1) % 5000 == 0:
        print(f"  {idx+1} processed, {len(harmless_pairs)} total pairs")

print(f"  Added {len(harmless_pairs)} HH-RLHF pairs")

# 2. PKU-SafeRLHF CLEAR examples (12k target)
print("\n[2/4] Adding PKU-SafeRLHF CLEAR safety examples...")
pku_clear_count = 0
for idx, example in enumerate(pku_data):
    if pku_clear_count >= 12000:
        break

    if len(example.get('response_0', '')) > 0 and len(example.get('response_1', '')) > 0:
        is_safe_0 = example.get('is_response_0_safe', True)
        is_safe_1 = example.get('is_response_1_safe', True)

        # Only take CLEAR examples (one safe, one unsafe)
        if is_safe_0 and not is_safe_1:
            chosen = example['response_0']
            rejected = example['response_1']
        elif is_safe_1 and not is_safe_0:
            chosen = example['response_1']
            rejected = example['response_0']
        else:
            continue  # Skip ambiguous for now

        harmless_pairs.append({
            "prompt": example['prompt'],
            "chosen": chosen,
            "rejected": rejected,
            "objective": "harmless",
            "source": "pku_clear"
        })
        pku_clear_count += 1

    if (idx+1) % 5000 == 0:
        print(f"  {idx+1} scanned, {pku_clear_count} clear pairs extracted")

print(f"  Added {pku_clear_count} PKU clear pairs")

# 3. PKU-SafeRLHF AMBIGUOUS examples (8k target)
print("\n[3/4] Adding PKU-SafeRLHF AMBIGUOUS safety examples...")
pku_ambig_count = 0
for idx, example in enumerate(pku_data):
    if pku_ambig_count >= 8000:
        break

    if len(example.get('response_0', '')) > 0 and len(example.get('response_1', '')) > 0:
        is_safe_0 = example.get('is_response_0_safe', True)
        is_safe_1 = example.get('is_response_1_safe', True)

        # Only take AMBIGUOUS examples (both safe OR both unsafe)
        if is_safe_0 == is_safe_1:
            safer_idx = int(example.get('safer_response_id', 0))
            chosen = example[f'response_{safer_idx}']
            rejected = example[f'response_{1-safer_idx}']

            harmless_pairs.append({
                "prompt": example['prompt'],
                "chosen": chosen,
                "rejected": rejected,
                "objective": "harmless",
                "source": "pku_ambiguous"
            })
            pku_ambig_count += 1

    if (idx+1) % 5000 == 0:
        print(f"  {idx+1} scanned, {pku_ambig_count} ambiguous pairs extracted")

print(f"  Added {pku_ambig_count} PKU ambiguous pairs")

# 4. UltraSafety (all ~3k adversarial examples)
print("\n[4/4] Adding UltraSafety adversarial safety examples...")
ultrasafety_count = 0
for idx, example in enumerate(ultrasafety_data):
    if idx >= 3000:  # Use all 3k examples
        break

    # UltraSafety has completions with harmlessness ratings
    try:
        completions = example.get('completions', [])
        if len(completions) < 2:
            continue

        # Score completions by harmlessness
        scored = []
        for comp in completions:
            annotations = comp.get('annotations', {})
            # harmlessness is a LIST of dicts, not a dict! Get the first item
            harmless_list = annotations.get('harmlessness', [])
            if isinstance(harmless_list, list) and len(harmless_list) > 0:
                harmless_score = harmless_list[0].get('Rating', '1')
            else:
                harmless_score = '1'

            # Convert to int (1=safe, 0=harmful)
            try:
                score = int(harmless_score)
            except:
                score = 1 if harmless_score == '1' else 0

            scored.append((score, comp.get('response', '')))

        if len(scored) >= 2:
            scored.sort(reverse=True, key=lambda x: x[0])
            chosen = scored[0][1]  # Most harmless
            rejected = scored[-1][1]  # Least harmless

            if len(chosen) > 0 and len(rejected) > 0 and scored[0][0] != scored[-1][0]:
                harmless_pairs.append({
                    "prompt": example['instruction'],
                    "chosen": chosen,
                    "rejected": rejected,
                    "objective": "harmless",
                    "source": "ultrasafety"
                })
                ultrasafety_count += 1
    except Exception as e:
        continue

    if (idx+1) % 500 == 0:
        print(f"  {idx+1} scanned, {ultrasafety_count} pairs extracted")

print(f"  Added {ultrasafety_count} UltraSafety pairs")

# # Prepare helpful dataset from PKU-SafeRLHF
print("\nPreparing HELPFUL dataset...")
helpful_pairs = []
for idx, example in enumerate(pku_data):
    if idx >= 50000:
        break

    if len(example.get('response_0', '')) > 0 and len(example.get('response_1', '')) > 0:
        better_idx = int(example.get('better_response_id', 0))
        chosen = example[f'response_{better_idx}']
        rejected = example[f'response_{1-better_idx}']

        helpful_pairs.append({
            "prompt": example['prompt'],
            "chosen": chosen,
            "rejected": rejected,
            "objective": "helpful"
        })

    if (idx+1) % 1000 == 0:
        print(f"  {idx+1} processed, {len(helpful_pairs)} pairs extracted")

# Prepare honest dataset from UltraFeedback (filtered for honesty+truthfulness)
print("\nPreparing HONEST dataset (high-contrast filtering: gap >3)...")
honest_pairs = []
for idx, example in enumerate(ultra_data):
    if idx >= 50000:  # Increased to 50k for better margin
        break

    # UltraFeedback has completions with scores for honesty and truthfulness
    try:
        completions = example.get('completions', [])
        if len(completions) < 2:
            continue

        # Score completions by honesty + truthfulness
        scored = []
        for comp in completions:
            annotations = comp.get('annotations', {})
            honesty_score = annotations.get('honesty', {}).get('Rating', '0')
            truth_score = annotations.get('truthfulness', {}).get('Rating', '0')

            # Convert string ratings to int
            try:
                honesty = int(honesty_score) if honesty_score else 0
            except (ValueError, TypeError):
                honesty = 0

            try:
                truth = int(truth_score) if truth_score else 0
            except (ValueError, TypeError):
                truth = 0

            combined_score = honesty + truth
            scored.append((combined_score, comp.get('response', '')))

        if len(scored) >= 2:
            scored.sort(reverse=True, key=lambda x: x[0])
            chosen_score = scored[0][0]
            rejected_score = scored[-1][0]
            chosen = scored[0][1]
            rejected = scored[-1][1]

            # FILTER: Only use high-contrast pairs (gap >3) for clear learning signal
            score_gap = chosen_score - rejected_score
            if score_gap > 3 and len(chosen) > 0 and len(rejected) > 0:
                honest_pairs.append({
                    "prompt": example['instruction'],
                    "chosen": chosen,
                    "rejected": rejected,
                    "objective": "honest"
                })
    except Exception as e:
        continue

    if (idx+1) % 1000 == 0:
        print(f"  {idx+1} processed, {len(honest_pairs)} pairs extracted")

# Save training datasets (no validation split needed)
harmless_train_path = os.path.join(OUTPUT_DIR, "harmless_train.json")
helpful_train_path = os.path.join(OUTPUT_DIR, "helpful_train.json")
honest_train_path = os.path.join(OUTPUT_DIR, "honest_train.json")

print("\nSaving HARMLESS training dataset...")
with open(harmless_train_path, 'w', encoding='utf-8') as f:
    json.dump(harmless_pairs, f, indent=2, ensure_ascii=False)

with open(helpful_train_path, 'w', encoding='utf-8') as f:
    json.dump(helpful_pairs, f, indent=2, ensure_ascii=False)

with open(honest_train_path, 'w', encoding='utf-8') as f:
    json.dump(honest_pairs, f, indent=2, ensure_ascii=False)

print(f"\n=== Data Preparation Complete ===")
print(f"\nHarmless pairs: {len(harmless_pairs)} total")
print(f"  - Anthropic HH-RLHF: {sum(1 for p in harmless_pairs if p.get('source') == 'hh-rlhf')}")
print(f"  - PKU (clear safety): {sum(1 for p in harmless_pairs if p.get('source') == 'pku_clear')}")
print(f"  - PKU (ambiguous):   {sum(1 for p in harmless_pairs if p.get('source') == 'pku_ambiguous')}")
print(f"  - UltraSafety:       {sum(1 for p in harmless_pairs if p.get('source') == 'ultrasafety')}")
print(f"\n✓ Anti-skewness: All datasets shuffled (seed={RANDOM_SEED}) before sampling")
print(f"✓ Reproducible: Fixed seed ensures consistent data splits across runs")
print(f"✓ Multi-source: HH-RLHF baseline + PKU diversity + UltraSafety adversarial")
