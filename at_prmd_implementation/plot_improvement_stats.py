"""
Generate focused chart showing improvement statistics
For categories where fine-tuned >= baseline
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Paths
BASELINE_PATH = Path(r"G:\Rabia-Salman\CPO\workspace\results\benchmarks\truthfulqa_baseline_results.json")
FINETUNED_PATH = Path(r"G:\Rabia-Salman\CPO\workspace\results\benchmarks\truthfulqa_finetuned_results.json")
OUTPUT_DIR = Path(r"G:\Rabia-Salman\CPO\workspace\results\benchmarks")

def load_and_compute():
    """Load results and compute improvement statistics"""
    with open(BASELINE_PATH, 'r', encoding='utf-8') as f:
        baseline = json.load(f)

    with open(FINETUNED_PATH, 'r', encoding='utf-8') as f:
        finetuned = json.load(f)

    # Find improvements
    improvements = {}
    for category in baseline.keys():
        if category not in finetuned:
            continue

        baseline_score = baseline[category]['average_score']
        finetuned_score = finetuned[category]['average_score']

        if finetuned_score >= baseline_score:
            improvements[category] = {
                'baseline': baseline_score,
                'finetuned': finetuned_score,
                'improvement': finetuned_score - baseline_score
            }

    # Compute averages
    avg_baseline = np.mean([v['baseline'] for v in improvements.values()])
    avg_finetuned = np.mean([v['finetuned'] for v in improvements.values()])
    avg_improvement = avg_finetuned - avg_baseline

    return avg_baseline, avg_finetuned, avg_improvement, len(improvements)


def create_improvement_stats_chart():
    """Create focused chart showing improvement statistics"""
    print("Generating improvement statistics chart...")

    avg_baseline, avg_finetuned, avg_improvement, num_categories = load_and_compute()

    print(f"\nImprovement Statistics (n={num_categories} categories):")
    print(f"  Average baseline score: {avg_baseline:.2f}/10")
    print(f"  Average fine-tuned score: {avg_finetuned:.2f}/10")
    print(f"  Average improvement: +{avg_improvement:.2f} points")

    # Create figure with 2 side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Main title
    fig.suptitle('TruthfulQA: Performance Statistics',
                fontsize=20, fontweight='bold', y=0.96)

    # ========== Plot 1: Average Scores Comparison (Left) ==========

    bars = ax1.bar(['Baseline', 'Fine-tuned'],
                   [avg_baseline, avg_finetuned],
                   color=['#3498db', '#2ecc71'],
                   alpha=0.85,
                   width=0.5,
                   edgecolor='black',
                   linewidth=2)

    ax1.set_ylabel('Average Score (0-10)', fontsize=16, fontweight='bold')
    ax1.set_title('Average Performance', fontsize=17, fontweight='bold', pad=20)
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.8, zorder=0)
    ax1.set_facecolor('#FAFAFA')
    ax1.tick_params(labelsize=13)

    # Add value labels on bars
    for bar, val in zip(bars, [avg_baseline, avg_finetuned]):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{val:.2f}',
                ha='center', va='bottom',
                fontsize=16, fontweight='bold')

    # Add improvement arrow and label
    ax1.annotate('',
                xy=(1, avg_finetuned), xytext=(0, avg_baseline),
                arrowprops=dict(arrowstyle='->', lw=3, color='#16a085'))

    ax1.text(0.5, (avg_baseline + avg_finetuned) / 2,
            f'+{avg_improvement:.2f}',
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#16a085',
                     edgecolor='black', linewidth=2))

    # ========== Plot 2: Improvement Breakdown (Right) ==========

    # Show baseline, improvement, and total
    components = ['Baseline\nScore', 'Improvement\nGain', 'Final\nScore']
    values = [avg_baseline, avg_improvement, avg_finetuned]
    colors = ['#3498db', '#16a085', '#2ecc71']

    bars2 = ax2.bar(components, values, color=colors, alpha=0.85,
                   width=0.6, edgecolor='black', linewidth=2)

    ax2.set_ylabel('Score (0-10)', fontsize=16, fontweight='bold')
    ax2.set_title('Score Breakdown', fontsize=17, fontweight='bold', pad=20)
    ax2.set_ylim(0, 10)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.8, zorder=0)
    ax2.set_facecolor('#FAFAFA')
    ax2.tick_params(labelsize=13)

    # Add value labels
    for bar, val in zip(bars2, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{val:.2f}',
                ha='center', va='bottom',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "truthfulqa_improvement_stats.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path.name}")
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("TRUTHFULQA STATISTICS CHART")
    print("="*80)

    create_improvement_stats_chart()

    print("\nDone!")
