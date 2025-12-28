"""
Visualize Research-Grade Reward-Guided Multi-Task Policy Training
Reads TensorBoard logs and creates detailed plots for CRMD training

UPDATED FOR NEW IMPLEMENTATION:
- Fixed priority weights (not learnable)
- Margin-based loss (implicit + explicit, no constraints)
- Single RM per objective
- Per-objective accuracy and margins
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.ndimage import gaussian_filter1d

# Configuration
POLICY_CHECKPOINT_DIR = r"G:\Rabia-Salman\CPO\workspace\checkpoints\rg_dpo_policy_final"
OUTPUT_DIR = r"G:\Rabia-Salman\CPO\workspace\results\plots\policy_training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OBJECTIVES = ['harmless', 'helpful', 'honest']
COLORS = {
    'harmless': '#e73c7e',  # Pink/red for safety
    'helpful': '#34c8db',   # Cyan for helpfulness
    'honest': '#652ecc',    # Purple for honesty
}

PRIORITY_WEIGHTS = {
    'harmless': 1.0,   # Highest priority
    'helpful': 0.4,    # Lower priority
    'honest': 0.7      # Medium priority
}


def load_tensorboard_logs(checkpoint_dir):
    """Load training metrics from TensorBoard logs (handles multiple runs)"""
    print(f"\nLoading logs from: {checkpoint_dir}")

    runs_dir = os.path.join(checkpoint_dir, "runs")
    if not os.path.exists(runs_dir):
        print(f"ERROR: No runs directory found at {runs_dir}")
        return None

    # Find all run directories
    run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not run_dirs:
        print(f"ERROR: No run directories found")
        return None

    print(f"Found {len(run_dirs)} training run(s):")
    for i, run in enumerate(sorted(run_dirs), 1):
        print(f"  {i}. {run}")

    # Load metrics from ALL runs (for resume capability)
    all_metrics = {}

    for run_dir in sorted(run_dirs):
        actual_log_dir = os.path.join(runs_dir, run_dir)
        print(f"\nProcessing run: {run_dir}")

        try:
            ea = EventAccumulator(actual_log_dir)
            ea.Reload()

            scalar_tags = ea.Tags()["scalars"]
            print(f"  Found {len(scalar_tags)} scalar metrics")

            # Extract metrics
            for tag in scalar_tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                # Merge with existing data (for multi-run training)
                if tag in all_metrics:
                    all_metrics[tag]["steps"].extend(steps)
                    all_metrics[tag]["values"].extend(values)
                else:
                    all_metrics[tag] = {"steps": steps, "values": values}

        except Exception as e:
            print(f"  Warning: Failed to load run {run_dir}: {e}")
            continue

    if not all_metrics:
        print("ERROR: No metrics loaded from any run")
        return None

    # Sort by step for each metric (in case of resume)
    for tag in all_metrics:
        sorted_pairs = sorted(zip(all_metrics[tag]["steps"], all_metrics[tag]["values"]))
        all_metrics[tag]["steps"], all_metrics[tag]["values"] = zip(*sorted_pairs)
        all_metrics[tag]["steps"] = list(all_metrics[tag]["steps"])
        all_metrics[tag]["values"] = list(all_metrics[tag]["values"])

    return all_metrics


def smooth(values, sigma=2):
    """Smooth values using Gaussian filter"""
    if len(values) < 3:
        return values
    return gaussian_filter1d(values, sigma=sigma)


def downsample(steps, values, max_points=500):
    """Downsample data to max_points for cleaner visualization"""
    if len(steps) <= max_points:
        return steps, values

    # Take every nth point
    step_size = len(steps) // max_points
    indices = list(range(0, len(steps), step_size))

    # Always include last point
    if indices[-1] != len(steps) - 1:
        indices.append(len(steps) - 1)

    return [steps[i] for i in indices], [values[i] for i in indices]


def plot_per_objective_losses(metrics):
    """Plot base and weighted losses for each objective"""
    print("\nGenerating per-objective loss plots...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig.suptitle('Training Loss per Objective', fontsize=18, fontweight='bold', y=0.995)

    for idx, obj in enumerate(OBJECTIVES):
        ax = axes[idx]

        base_loss_key = f"train/{obj}/loss/base"
        weighted_loss_key = f"train/{obj}/loss/weighted"

        found_data = False

        # Plot base loss only (simpler)
        if base_loss_key in metrics:
            steps = metrics[base_loss_key]['steps']
            values = metrics[base_loss_key]['values']

            # Downsample and smooth
            steps_ds, values_ds = downsample(steps, values, max_points=500)
            smoothed = smooth(values_ds, sigma=5)

            ax.plot(steps_ds, smoothed, linewidth=3.5, color=COLORS[obj],
                   alpha=0.9, label=f'{obj.capitalize()} Loss')
            found_data = True

            # Add stats box
            vals = np.array(values)
            if len(vals) > 0:
                improvement = ((vals[0]-vals[-1])/vals[0]*100) if vals[0] != 0 else 0
                status = "✓ Improved" if improvement > 0 else "✗ Degraded"
                desc = f"Priority: {PRIORITY_WEIGHTS[obj]}\n"
                desc += f"Start: {vals[0]:.3f}\n"
                desc += f"End: {vals[-1]:.3f}\n"
                desc += f"Change: {improvement:+.1f}% {status}"
                ax.text(0.98, 0.97, desc, transform=ax.transAxes, fontsize=11,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white',
                                edgecolor=COLORS[obj], linewidth=2, alpha=0.9))

        if not found_data:
            ax.text(0.5, 0.5, f'No {obj} loss data', ha='center', va='center',
                   fontsize=14, color='gray')

        ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax.set_title(f'{obj.upper()}', fontsize=15, fontweight='bold',
                    color=COLORS[obj], pad=10)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_facecolor('#FAFAFA')

        # Cleaner tick labels
        ax.tick_params(labelsize=11)

    axes[-1].set_xlabel('Training Step', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "1_per_objective_losses.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_individual_objectives(metrics):
    """Plot loss and accuracy for each objective in separate files"""
    print("\nGenerating individual objective plots...")

    for obj in OBJECTIVES:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        fig.suptitle(f'{obj.upper()} Objective Training', fontsize=20, fontweight='bold', y=0.995)

        # ----- LOSS SUBPLOT -----
        loss_key = f"train/{obj}/loss/base"
        if loss_key in metrics:
            steps = metrics[loss_key]['steps']
            values = metrics[loss_key]['values']
            steps_ds, values_ds = downsample(steps, values, max_points=500)
            smoothed = smooth(values_ds, sigma=5)

            ax1.plot(steps_ds, smoothed, linewidth=4, color=COLORS[obj], alpha=0.9)

            # Stats
            vals = np.array(values)
            improvement = ((vals[0]-vals[-1])/vals[0]*100) if vals[0] != 0 else 0
            status = "✓ Improved" if improvement > 0 else "✗ Degraded"

            ax1.set_ylabel('Loss', fontsize=16, fontweight='bold')
            ax1.set_title(f'Loss (Priority Weight: {PRIORITY_WEIGHTS[obj]})', fontsize=16, fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.25, linewidth=0.8)
            ax1.set_facecolor('#FAFAFA')
            ax1.tick_params(labelsize=12)

            # Add text annotation
            desc = f"Start: {vals[0]:.3f}\nEnd: {vals[-1]:.3f}\nChange: {improvement:+.1f}% {status}"
            ax1.text(0.02, 0.98, desc, transform=ax1.transAxes, fontsize=13,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white',
                             edgecolor=COLORS[obj], linewidth=3, alpha=0.95))
        else:
            ax1.text(0.5, 0.5, 'No loss data', ha='center', va='center', fontsize=14, color='gray')

        # ----- ACCURACY SUBPLOT -----
        acc_key = f"train/{obj}/accuracy"
        if acc_key in metrics:
            steps = metrics[acc_key]['steps']
            values = np.array(metrics[acc_key]['values']) * 100
            steps_ds, values_ds = downsample(steps, list(values), max_points=500)
            smoothed = smooth(values_ds, sigma=5)

            ax2.plot(steps_ds, smoothed, linewidth=4, color=COLORS[obj], alpha=0.9)
            ax2.axhline(y=50, color='#555555', linestyle='--', alpha=0.6, linewidth=2.5,
                       label='Random Baseline')

            ax2.set_xlabel('Training Step', fontsize=16, fontweight='bold')
            ax2.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
            ax2.set_title('Preference Prediction Accuracy', fontsize=16, fontweight='bold', pad=15)
            ax2.set_ylim([0, 105])
            ax2.legend(fontsize=12, loc='lower right', framealpha=0.95)
            ax2.grid(True, alpha=0.25, linewidth=0.8)
            ax2.set_facecolor('#FAFAFA')
            ax2.tick_params(labelsize=12)

            # Add final value annotation
            if len(smoothed) > 0:
                desc = f"Final: {smoothed[-1]:.1f}%"
                ax2.text(0.02, 0.98, desc, transform=ax2.transAxes, fontsize=13,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white',
                                 edgecolor=COLORS[obj], linewidth=3, alpha=0.95))
        else:
            ax2.text(0.5, 0.5, 'No accuracy data', ha='center', va='center', fontsize=14, color='gray')

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"{obj}_training.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {obj}_training.png")
        plt.close()


def plot_accuracy(metrics):
    """Plot per-objective accuracy"""
    print("\nGenerating accuracy plots...")

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle('Preference Prediction Accuracy', fontsize=18, fontweight='bold')

    found_any = False
    for obj in OBJECTIVES:
        acc_key = f"train/{obj}/accuracy"

        if acc_key in metrics:
            steps = metrics[acc_key]['steps']
            values = np.array(metrics[acc_key]['values']) * 100  # Convert to percentage

            # Downsample and smooth
            steps_ds, values_ds = downsample(steps, list(values), max_points=500)
            smoothed = smooth(values_ds, sigma=5)

            ax.plot(steps_ds, smoothed, linewidth=3.5, color=COLORS[obj],
                   label=f'{obj.capitalize()}', alpha=0.9)
            found_any = True

            # Add final value annotation
            if len(smoothed) > 0:
                ax.annotate(f'{smoothed[-1]:.1f}%',
                           xy=(steps_ds[-1], smoothed[-1]),
                           xytext=(10, 0), textcoords='offset points',
                           fontsize=11, fontweight='bold', color=COLORS[obj])

    if found_any:
        ax.axhline(y=50, color='#555555', linestyle='--', alpha=0.6, linewidth=2.5,
                  label='Random Baseline', zorder=0)
        ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_ylim([-5, 105])
        ax.legend(fontsize=13, loc='lower right', framealpha=0.95)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_facecolor('#FAFAFA')
        ax.tick_params(labelsize=11)
    else:
        ax.text(0.5, 0.5, 'No accuracy data found', ha='center', va='center',
               fontsize=14, color='gray')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "2_accuracy.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_margins(metrics):
    """Plot implicit vs explicit margins"""
    print("\nGenerating margin analysis plots...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Margin Analysis: Implicit (Policy) vs Explicit (RM)',
                 fontsize=16, fontweight='bold')

    for idx, obj in enumerate(OBJECTIVES):
        ax = axes[idx]

        implicit_key = f"train/{obj}/margin/implicit"
        explicit_key = f"train/{obj}/margin/explicit"
        combined_key = f"train/{obj}/margin/combined"

        found_data = False

        if implicit_key in metrics:
            steps = metrics[implicit_key]['steps']
            values = metrics[implicit_key]['values']
            smoothed = smooth(values, sigma=3)
            ax.plot(steps, smoothed, linewidth=2.5, color=COLORS[obj],
                   label='Implicit (DPO)', alpha=0.8)
            found_data = True

        if explicit_key in metrics:
            steps = metrics[explicit_key]['steps']
            values = metrics[explicit_key]['values']
            smoothed = smooth(values, sigma=3)
            ax.plot(steps, smoothed, linewidth=2.5, color=COLORS[obj],
                   linestyle='--', label='Explicit (RM)', alpha=0.8)
            found_data = True

        if combined_key in metrics:
            steps = metrics[combined_key]['steps']
            values = metrics[combined_key]['values']
            smoothed = smooth(values, sigma=3)
            ax.plot(steps, smoothed, linewidth=3, color='black',
                   linestyle='-', label='Combined (Δ_total)', alpha=0.5)
            found_data = True

        if found_data:
            ax.axhline(y=0, color='red', linestyle=':', alpha=0.4, linewidth=1.5)

            # Add final values
            if implicit_key in metrics and explicit_key in metrics:
                impl_final = metrics[implicit_key]['values'][-1] if metrics[implicit_key]['values'] else 0
                expl_final = metrics[explicit_key]['values'][-1] if metrics[explicit_key]['values'] else 0
                alignment = "✓ Aligned" if impl_final * expl_final > 0 else "✗ Misaligned"
                desc = f"Implicit: {impl_final:.3f}\nExplicit: {expl_final:.3f}\n{alignment}"
                ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=9,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor=COLORS[obj], alpha=0.2))
        else:
            ax.text(0.5, 0.5, f'No {obj} margin data', ha='center', va='center', fontsize=12)

        ax.set_ylabel('Margin', fontsize=11)
        ax.set_title(f'{obj.upper()} Margins', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#F8F9FA')

    axes[-1].set_xlabel('Training Step', fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "3_margins.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_rewards(metrics):
    """Plot chosen vs rejected rewards"""
    print("\nGenerating reward plots...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Reward Model Scores: Chosen vs Rejected',
                 fontsize=16, fontweight='bold')

    for idx, obj in enumerate(OBJECTIVES):
        ax = axes[idx]

        chosen_key = f"train/{obj}/reward/chosen"
        rejected_key = f"train/{obj}/reward/rejected"
        chosen_norm_key = f"train/{obj}/reward/chosen_norm"
        rejected_norm_key = f"train/{obj}/reward/rejected_norm"

        found_data = False

        # Raw rewards
        if chosen_key in metrics:
            steps = metrics[chosen_key]['steps']
            values = metrics[chosen_key]['values']
            smoothed = smooth(values, sigma=3)
            ax.plot(steps, smoothed, linewidth=2, color=COLORS[obj],
                   label='Chosen (raw)', alpha=0.6, linestyle=':')
            found_data = True

        if rejected_key in metrics:
            steps = metrics[rejected_key]['steps']
            values = metrics[rejected_key]['values']
            smoothed = smooth(values, sigma=3)
            ax.plot(steps, smoothed, linewidth=2, color=COLORS[obj],
                   linestyle=':', label='Rejected (raw)', alpha=0.4)
            found_data = True

        # Normalized rewards (used in loss)
        if chosen_norm_key in metrics:
            steps = metrics[chosen_norm_key]['steps']
            values = metrics[chosen_norm_key]['values']
            smoothed = smooth(values, sigma=3)
            ax.plot(steps, smoothed, linewidth=2.5, color=COLORS[obj],
                   label='Chosen (normalized)', alpha=0.9)
            found_data = True

        if rejected_norm_key in metrics:
            steps = metrics[rejected_norm_key]['steps']
            values = metrics[rejected_norm_key]['values']
            smoothed = smooth(values, sigma=3)
            ax.plot(steps, smoothed, linewidth=2.5, color=COLORS[obj],
                   linestyle='--', label='Rejected (normalized)', alpha=0.7)
            found_data = True

        # Shade gap
        if chosen_norm_key in metrics and rejected_norm_key in metrics:
            steps = metrics[chosen_norm_key]['steps']
            chosen_vals = smooth(metrics[chosen_norm_key]['values'], sigma=3)
            rejected_vals = smooth(metrics[rejected_norm_key]['values'], sigma=3)
            ax.fill_between(steps, chosen_vals, rejected_vals, alpha=0.15, color=COLORS[obj])

        if found_data:
            ax.axhline(y=0, color='black', linestyle=':', alpha=0.4, linewidth=1.5)

            # Add gap stats
            if chosen_key in metrics and rejected_key in metrics:
                chosen_final = metrics[chosen_key]['values'][-1]
                rejected_final = metrics[rejected_key]['values'][-1]
                gap = chosen_final - rejected_final
                status = "✓ Good" if gap > 0 else "✗ Bad"
                desc = f"Raw Gap: {gap:.3f}\n{status}"
                ax.text(0.02, 0.98, desc, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor=COLORS[obj], alpha=0.2))
        else:
            ax.text(0.5, 0.5, f'No {obj} reward data', ha='center', va='center', fontsize=12)

        ax.set_ylabel('Reward Score', fontsize=11)
        ax.set_title(f'{obj.upper()} RM Scores', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#F8F9FA')

    axes[-1].set_xlabel('Training Step', fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "4_rewards.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_priority_weights(metrics):
    """Plot fixed priority weights over time (should be constant)"""
    print("\nGenerating priority weights plot...")

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('Fixed Priority Weights (Not Learnable)',
                 fontsize=16, fontweight='bold')

    found_any = False
    for obj in OBJECTIVES:
        weight_key = f"train/priority_weights/{obj}"

        if weight_key in metrics:
            steps = metrics[weight_key]['steps']
            values = metrics[weight_key]['values']

            ax.plot(steps, values, linewidth=3, color=COLORS[obj],
                   label=f'{obj.capitalize()} = {PRIORITY_WEIGHTS[obj]}',
                   marker='o', markersize=2, markevery=max(1, len(steps)//20))
            found_any = True

    if found_any:
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Priority Weight', fontsize=12)
        ax.set_title('Safety prioritization: harmless > honest > helpful', fontsize=13)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.2])
        ax.set_facecolor('#F8F9FA')
    else:
        ax.text(0.5, 0.5, 'No priority weight data found', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "5_priority_weights.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_training_overview(metrics):
    """Plot overall training metrics"""
    print("\nGenerating training overview...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Training Stability Metrics', fontsize=18, fontweight='bold')

    # 1. Total multitask loss
    ax = axes[0]
    loss_key = "train/loss/total_multitask"
    if loss_key in metrics:
        steps = metrics[loss_key]['steps']
        values = metrics[loss_key]['values']
        steps_ds, values_ds = downsample(steps, values, max_points=500)
        smoothed = smooth(values_ds, sigma=5)

        ax.plot(steps_ds, smoothed, linewidth=3.5, color='#E74C3C', alpha=0.9)

        if len(values) > 0:
            desc = f"Start: {values[0]:.3f}\nEnd: {values[-1]:.3f}\nMin: {min(values):.3f}\nMax: {max(values):.3f}"
            ax.text(0.02, 0.98, desc, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white',
                            edgecolor='#E74C3C', linewidth=2, alpha=0.9))
    else:
        ax.text(0.5, 0.5, 'No total loss found', ha='center', va='center', fontsize=14)

    ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Loss', fontsize=14, fontweight='bold')
    ax.set_title('Weighted Multitask Loss', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_facecolor('#FAFAFA')
    ax.tick_params(labelsize=11)

    # 2. Gradient norm
    ax = axes[1]
    grad_key = "train/grad_norm"
    if grad_key in metrics:
        steps = metrics[grad_key]['steps']
        values = metrics[grad_key]['values']
        steps_ds, values_ds = downsample(steps, values, max_points=500)
        smoothed = smooth(values_ds, sigma=3)

        ax.plot(steps_ds, smoothed, linewidth=3.5, color='#8E44AD', alpha=0.9)
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=2.5,
                  label='Clip Threshold')

        if len(values) > 0:
            avg_grad = np.mean(values)
            max_grad = np.max(values)
            status = "✓ Stable" if avg_grad < 0.3 else "⚠ UNSTABLE"
            desc = f"Avg: {avg_grad:.2f}\nMax: {max_grad:.2f}\n{status}"
            ax.text(0.02, 0.98, desc, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white',
                            edgecolor='#8E44AD', linewidth=2, alpha=0.9))
    else:
        ax.text(0.5, 0.5, 'No gradient norm found', ha='center', va='center', fontsize=14)

    ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontsize=14, fontweight='bold')
    ax.set_title('Gradient Stability (Clip: 0.3)', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=13, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_facecolor('#FAFAFA')
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "3_training_overview.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def save_training_summary(metrics):
    """Save training summary to text file"""
    print("\nGenerating training summary...")

    output_path = os.path.join(OUTPUT_DIR, "training_summary.txt")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESEARCH-GRADE REWARD-GUIDED MULTI-TASK POLICY TRAINING SUMMARY\n")
        f.write("Constitutional Reward Model Distillation (CRMD)\n")
        f.write("="*80 + "\n\n")

        # Overall loss
        loss_key = "train/loss/total_multitask"
        if loss_key in metrics:
            values = np.array(metrics[loss_key]['values'])
            f.write("OVERALL MULTITASK LOSS:\n")
            f.write(f"  Initial: {values[0]:.6f}\n")
            f.write(f"  Final: {values[-1]:.6f}\n")
            f.write(f"  Min: {values.min():.6f}\n")
            f.write(f"  Max: {values.max():.6f}\n")
            improvement = ((values[0]-values[-1])/values[0]*100) if values[0] != 0 else 0
            f.write(f"  Improvement: {improvement:.2f}%\n")
            f.write(f"  Steps: {len(values)}\n\n")

        # Per-objective analysis
        f.write("="*80 + "\n")
        f.write("PER-OBJECTIVE ANALYSIS\n")
        f.write("="*80 + "\n\n")

        for obj in OBJECTIVES:
            f.write(f"{obj.upper()} (Priority: {PRIORITY_WEIGHTS[obj]})\n")
            f.write("-" * 80 + "\n")

            # Loss
            base_loss_key = f"train/{obj}/loss/base"
            if base_loss_key in metrics:
                vals = np.array(metrics[base_loss_key]['values'])
                if len(vals) > 0:
                    improvement = ((vals[0]-vals[-1])/vals[0]*100) if vals[0] != 0 else 0
                    f.write(f"  Base Loss: {vals[0]:.4f} -> {vals[-1]:.4f} ({improvement:.1f}%)\n")

            # Accuracy
            acc_key = f"train/{obj}/accuracy"
            if acc_key in metrics:
                vals = np.array(metrics[acc_key]['values']) * 100
                if len(vals) > 0:
                    f.write(f"  Accuracy: {vals[0]:.1f}% -> {vals[-1]:.1f}%\n")

            # Margins
            impl_key = f"train/{obj}/margin/implicit"
            expl_key = f"train/{obj}/margin/explicit"
            if impl_key in metrics and expl_key in metrics:
                impl = np.array(metrics[impl_key]['values'])
                expl = np.array(metrics[expl_key]['values'])
                if len(impl) > 0 and len(expl) > 0:
                    alignment = "✓ Aligned" if impl[-1] * expl[-1] > 0 else "✗ Misaligned"
                    f.write(f"  Margins: Implicit={impl[-1]:.3f}, Explicit={expl[-1]:.3f} ({alignment})\n")

            # Rewards
            chosen_key = f"train/{obj}/reward/chosen"
            rejected_key = f"train/{obj}/reward/rejected"
            if chosen_key in metrics and rejected_key in metrics:
                chosen = np.array(metrics[chosen_key]['values'])
                rejected = np.array(metrics[rejected_key]['values'])
                if len(chosen) > 0 and len(rejected) > 0:
                    gap = chosen[-1] - rejected[-1]
                    status = "✓ Good" if gap > 0 else "✗ Bad"
                    f.write(f"  Reward Gap: {gap:.3f} ({status})\n")

            f.write("\n")

        # Gradient stats
        grad_key = "train/grad_norm"
        if grad_key in metrics:
            vals = np.array(metrics[grad_key]['values'])
            f.write("="*80 + "\n")
            f.write("GRADIENT STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"  Average: {vals.mean():.6f}\n")
            f.write(f"  Max: {vals.max():.6f}\n")
            f.write(f"  Clip: 0.3\n")
            f.write(f"  Status: {'✓ STABLE' if vals.mean() < 0.3 else '⚠ HIGH'}\n\n")

        f.write("="*80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write("  Base Model: Qwen2.5-3B\n")
        f.write("  Method: CRMD (margin-based, no constraints)\n")
        f.write("  Loss: -log σ(Δ_implicit + α*Δ_explicit)\n")
        f.write("  Alpha (RM weight): 1.0\n")
        f.write("  Beta (DPO temp): 0.1\n")
        f.write("  Priority: [harmless=1.0, helpful=0.4, honest=0.7] (FIXED)\n")
        f.write("  LoRA: r=8, alpha=16, rsLoRA\n")
        f.write("  Batch: 4, Grad Accum: 16\n")
        f.write("  LR: 5e-7, Grad Clip: 0.3\n")
        f.write("  Epochs: 2\n\n")

    print(f"  Saved: {output_path}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("RESEARCH-GRADE REWARD-GUIDED MULTI-TASK VISUALIZATION")
    print("="*80)

    if not os.path.exists(POLICY_CHECKPOINT_DIR):
        print(f"\nERROR: Checkpoint directory not found!")
        print(f"Expected: {POLICY_CHECKPOINT_DIR}")
        return

    # Load metrics
    metrics = load_tensorboard_logs(POLICY_CHECKPOINT_DIR)

    if not metrics:
        print("\nERROR: No metrics found!")
        return

    print(f"\nLoaded {len(metrics)} unique metrics")
    print("\nSample metrics:")
    for i, key in enumerate(sorted(metrics.keys())[:15], 1):
        data_points = len(metrics[key]['steps'])
        print(f"  {i}. {key} ({data_points} points)")

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_individual_objectives(metrics)
    plot_per_objective_losses(metrics)
    plot_accuracy(metrics)
    plot_training_overview(metrics)
    save_training_summary(metrics)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("\nGenerated:")
    print("  0. Individual objective plots (harmless/helpful/honest_training.png)")
    print("  1. Per-objective losses (combined)")
    print("  2. Accuracy comparison (all objectives)")
    print("  3. Training overview (loss + grad + LR)")
    print("  + Training summary (text)")


if __name__ == "__main__":
    main()
