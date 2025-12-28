"""
Quick script to verify RM scores in precomputed files
"""
import json
import numpy as np

def check_file(obj_name):
    file_path = f"g:/Rabia-Salman/CPO/workspace/data/precomputed_ref_logprobs_final/{obj_name}_precomputed.json"

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        print(f"\n{'='*60}")
        print(f"Objective: {obj_name.upper()}")
        print(f"{'='*60}")
        print(f"Total samples: {len(data)}")

        # Check RM scores
        rm_keys = ['chosen_rm_harmless', 'chosen_rm_helpful', 'chosen_rm_honest',
                   'rejected_rm_harmless', 'rejected_rm_helpful', 'rejected_rm_honest']

        for key in rm_keys:
            scores = [sample[key] for sample in data]
            print(f"\n{key}:")
            print(f"  Mean:  {np.mean(scores):.4f}")
            print(f"  Std:   {np.std(scores):.4f}")
            print(f"  Min:   {np.min(scores):.4f}")
            print(f"  Max:   {np.max(scores):.4f}")
            print(f"  Zeros: {sum(s == 0.0 for s in scores)} / {len(scores)}")

        # Sample check
        print(f"\nSample 0 RM scores:")
        for key in rm_keys:
            print(f"  {key}: {data[0][key]:.4f}")

        print(f"\nSample {len(data)//2} RM scores:")
        for key in rm_keys:
            print(f"  {key}: {data[len(data)//2][key]:.4f}")

        # Health check
        print(f"\n{'='*60}")
        all_zeros = all(sample[key] == 0.0 for sample in data for key in rm_keys)
        if all_zeros:
            print("❌ BROKEN: All RM scores are 0.0!")
        else:
            print("✅ OK: RM scores look good")

    except FileNotFoundError:
        print(f"\n⚠️  File not found: {file_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    objectives = ["harmless", "helpful", "honest"]

    for obj in objectives:
        check_file(obj)

    print(f"\n{'='*60}\n")
