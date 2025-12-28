"""
Compare TruthfulQA Results - Show Only Improvements
Extracts categories where fine-tuned model performs >= baseline
Creates bar charts with max 10 categories per chart
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Paths
BASELINE_PATH = Path(r"G:\Rabia-Salman\CPO\workspace\results\benchmarks\truthfulqa_baseline_results.json")
FINETUNED_PATH = Path(r"G:\Rabia-Salman\CPO\workspace\results\benchmarks\truthfulqa_finetuned_results.json")
OUTPUT_DIR = Path(r"G:\Rabia-Salman\CPO\workspace\results\benchmarks")

def load_results():
    """Load baseline and finetuned results"""
    print("Loading results...")

    with open(BASELINE_PATH, 'r', encoding='utf-8') as f:
        baseline = json.load(f)

    with open(FINETUNED_PATH, 'r', encoding='utf-8') as f:
        finetuned = json.load(f)

    print(f"  Baseline categories: {len(baseline)}")
    print(f"  Fine-tuned categories: {len(finetuned)}")

    return baseline, finetuned


def extract_improvements(baseline, finetuned):
    """Extract categories where finetuned >= baseline"""
    improvements = {}

    for category in baseline.keys():
        if category not in finetuned:
            continue

        baseline_score = baseline[category]['average_score']
        finetuned_score = finetuned[category]['average_score']

        # Only include if finetuned >= baseline
        if finetuned_score >= baseline_score:
            improvements[category] = {
                'baseline': baseline_score,
                'finetuned': finetuned_score,
                'improvement': finetuned_score - baseline_score
            }

    # Sort by improvement (descending)
    improvements = dict(sorted(improvements.items(),
                               key=lambda x: x[1]['improvement'],
                               reverse=True))

    print(f"\nFound {len(improvements)} categories where fine-tuned >= baseline:")
    for cat, scores in list(improvements.items())[:10]:
        print(f"  {cat}: {scores['finetuned']:.2f} vs {scores['baseline']:.2f} (+{scores['improvement']:.2f})")

    if len(improvements) > 10:
        print(f"  ... and {len(improvements) - 10} more")

    return improvements


def create_improvement_charts(improvements):
    """Create bar charts comparing scores (max 10 categories per chart)"""
    print("\nGenerating improvement charts...")

    categories = list(improvements.keys())
    num_categories = len(categories)

    if num_categories == 0:
        print("  No improvements found!")
        return

    # Split into chunks of 10
    categories_per_chart = 10
    num_charts = (num_categories + categories_per_chart - 1) // categories_per_chart

    for chart_idx in range(num_charts):
        start_idx = chart_idx * categories_per_chart
        end_idx = min((chart_idx + 1) * categories_per_chart, num_categories)

        chart_categories = categories[start_idx:end_idx]
        baseline_vals = [improvements[cat]['baseline'] for cat in chart_categories]
        finetuned_vals = [improvements[cat]['finetuned'] for cat in chart_categories]
        improvement_vals = [improvements[cat]['improvement'] for cat in chart_categories]

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(14, max(8, len(chart_categories) * 0.7)))

        y_pos = np.arange(len(chart_categories))
        width = 0.35

        # Plot bars
        bars1 = ax.barh(y_pos - width/2, baseline_vals, width,
                       label='Baseline (Qwen2.5-3B)', color='#3498db', alpha=0.85)
        bars2 = ax.barh(y_pos + width/2, finetuned_vals, width,
                       label='Fine-tuned (AT-PRMG)', color='#2ecc71', alpha=0.85)

        # Customize
        ax.set_xlabel('TruthfulQA Score (0-10)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Category', fontsize=14, fontweight='bold')

        # Title
        if num_charts > 1:
            title = f'TruthfulQA: Categories ({chart_idx + 1}/{num_charts})'
        else:
            title = 'TruthfulQA: Categories'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(chart_categories, fontsize=11)
        ax.set_xlim(0, 10)
        ax.legend(fontsize=12, loc='lower right', framealpha=0.95)
        ax.grid(axis='x', alpha=0.3, linewidth=0.8)
        ax.set_facecolor('#FAFAFA')
        ax.tick_params(labelsize=11)

        # Add score labels on bars
        for i, (bl_val, ft_val, imp) in enumerate(zip(baseline_vals, finetuned_vals, improvement_vals)):
            # Baseline score
            ax.text(bl_val + 0.12, i - width/2, f'{bl_val:.2f}',
                   va='center', fontsize=10, color='#2980b9', fontweight='bold')

            # Finetuned score
            ax.text(ft_val + 0.12, i + width/2, f'{ft_val:.2f}',
                   va='center', fontsize=10, color='#27ae60', fontweight='bold')

            # Improvement delta
            if imp > 0:
                ax.text(max(bl_val, ft_val) + 0.5, i, f'+{imp:.2f}',
                       va='center', fontsize=9, color='#16a085',
                       fontweight='bold', style='italic')

        plt.tight_layout()

        # Save
        if num_charts > 1:
            output_path = OUTPUT_DIR / f"truthfulqa_improvements_part{chart_idx + 1}.png"
        else:
            output_path = OUTPUT_DIR / "truthfulqa_improvements.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")
        plt.close()

    print(f"\nGenerated {num_charts} chart(s)")


def create_summary_stats(improvements):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)

    if len(improvements) == 0:
        print("\nNo categories showed improvement.")
        return

    total_categories = len(improvements)
    avg_baseline = np.mean([v['baseline'] for v in improvements.values()])
    avg_finetuned = np.mean([v['finetuned'] for v in improvements.values()])
    avg_improvement = np.mean([v['improvement'] for v in improvements.values()])

    print(f"\nCategories with improvement: {total_categories}")
    print(f"Average baseline score: {avg_baseline:.2f}/10")
    print(f"Average fine-tuned score: {avg_finetuned:.2f}/10")
    print(f"Average improvement: +{avg_improvement:.2f}")

    # Top 5 improvements
    print(f"\nTop 5 improvements:")
    for i, (cat, scores) in enumerate(list(improvements.items())[:5], 1):
        print(f"  {i}. {cat}: {scores['baseline']:.2f} -> {scores['finetuned']:.2f} (+{scores['improvement']:.2f})")

    # Perfect scores
    perfect_scores = [cat for cat, scores in improvements.items() if scores['finetuned'] >= 9.5]
    if perfect_scores:
        print(f"\nCategories with near-perfect scores (≥9.5):")
        for cat in perfect_scores:
            print(f"  - {cat}: {improvements[cat]['finetuned']:.2f}")

    print("\n" + "="*80)


def create_overall_comparison(improvements, baseline, finetuned):
    """Create overall comparison summary chart"""
    print("\nGenerating overall comparison...")

    # Calculate overall stats
    all_baseline_scores = [v['average_score'] for v in baseline.values()]
    all_finetuned_scores = [v['average_score'] for v in finetuned.values()]

    improved_categories = len(improvements)
    total_categories = len(baseline)
    degraded_categories = total_categories - improved_categories

    avg_baseline_all = np.mean(all_baseline_scores)
    avg_finetuned_all = np.mean(all_finetuned_scores)

    avg_baseline_improved = np.mean([v['baseline'] for v in improvements.values()]) if improvements else 0
    avg_finetuned_improved = np.mean([v['finetuned'] for v in improvements.values()]) if improvements else 0

    # Create summary chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('TruthfulQA: Overall Improvement Summary', fontsize=18, fontweight='bold')

    # Left plot: Overall average scores
    ax1.bar(['Baseline\n(All Categories)', 'Fine-tuned\n(All Categories)'],
           [avg_baseline_all, avg_finetuned_all],
           color=['#3498db', '#2ecc71'], alpha=0.85, width=0.6)
    ax1.set_ylabel('Average Score (0-10)', fontsize=14, fontweight='bold')
    ax1.set_title('Overall Performance', fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.8)
    ax1.set_facecolor('#FAFAFA')

    # Add value labels
    for i, (label, val) in enumerate(zip(['Baseline', 'Fine-tuned'], [avg_baseline_all, avg_finetuned_all])):
        ax1.text(i, val + 0.2, f'{val:.2f}', ha='center', fontsize=13, fontweight='bold')

    # Improvement annotation
    improvement = avg_finetuned_all - avg_baseline_all
    ax1.text(0.5, avg_baseline_all + (improvement/2), f'{improvement:+.2f}',
            ha='center', fontsize=12, fontweight='bold', color='#16a085')

    # Right plot: Category breakdown
    categories_data = [improved_categories, degraded_categories]
    colors = ['#2ecc71', '#e74c3c']
    labels = [f'Improved\n({improved_categories})', f'Degraded\n({degraded_categories})']

    wedges, texts, autotexts = ax2.pie(categories_data, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Category Performance', fontsize=15, fontweight='bold', pad=15)

    # Add summary text box
    summary_text = f"Improved Categories:\n"
    summary_text += f"  Avg: {avg_baseline_improved:.2f} -> {avg_finetuned_improved:.2f}\n"
    summary_text += f"  Gain: +{avg_finetuned_improved - avg_baseline_improved:.2f}"

    ax2.text(0.5, -1.3, summary_text, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2ecc71', linewidth=2),
            transform=ax2.transAxes)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "truthfulqa_overall_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def main():
    print("="*80)
    print("TRUTHFULQA IMPROVEMENT ANALYSIS")
    print("="*80)

    # Load data
    baseline, finetuned = load_results()

    # Extract improvements
    improvements = extract_improvements(baseline, finetuned)

    if len(improvements) == 0:
        print("\nNo improvements found. The fine-tuned model did not outperform")
        print("the baseline on any category.")
        return

    # Create visualizations
    create_improvement_charts(improvements)
    create_overall_comparison(improvements, baseline, finetuned)

    print("\nDone!")


if __name__ == "__main__":
    main()
