"""
Visualize Reward Model Training Progress
Reads TensorBoard logs and creates plots for loss, accuracy, and other metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Output directory
OUTPUT_DIR = r"g:\Rabia-Salman\CPO\workspace\results\plots\reward_training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Direct paths to tensorboard event files
EVENT_FILES = {
    'helpful_m1': r"G:\Rabia-Salman\CPO\workspace\checkpoints\reward_model_helpful_m1\runs\Dec22_00-12-50_aKs\events.out.tfevents.1766344386.aKs.264932.0",
    'honest_m1': r"G:\Rabia-Salman\CPO\workspace\checkpoints\reward_model_honest_m1\runs\Dec26_15-28-19_aKs\events.out.tfevents.1766745070.aKs.52152.0",
    'harmless_m1': r"G:\Rabia-Salman\CPO\workspace\checkpoints\reward_model_harmless_m1\runs\Dec26_07-26-52_aKs\events.out.tfevents.1766716130.aKs.31072.0",  # Training in progress
}

COLORS = {
    'harmless': "#e73c7e",
    'helpful': "#34c8db",
    'honest': "#652ecc"
}


def load_tensorboard_metrics(event_file):
    """Load all metrics from a TensorBoard event file"""
    if not event_file or not os.path.exists(event_file):
        return None

    try:
        ea = EventAccumulator(event_file)
        ea.Reload()

        scalar_tags = ea.Tags()["scalars"]
        if not scalar_tags:
            return None

        metrics = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            metrics[tag] = {"steps": steps, "values": values}

        return metrics

    except Exception as e:
        print(f"  ERROR loading {os.path.basename(event_file)}: {e}")
        return None


def get_metric(metrics, name):
    """Helper to find metric (with or without train/ prefix)"""
    if name in metrics:
        return metrics[name]
    elif f"train/{name}" in metrics:
        return metrics[f"train/{name}"]
    return None


def plot_training_progress(rm_name, metrics, color):
    """Plot 2x2 grid: loss, accuracy, margin, mean_reward"""
    print(f"  Creating training progress plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{rm_name.upper()} - Training Progress', fontsize=16, fontweight='bold')

    # Plot 1: Loss
    loss_metric = get_metric(metrics, "loss")
    if loss_metric:
        ax = axes[0, 0]
        ax.plot(loss_metric["steps"], loss_metric["values"], linewidth=2, color=color, label="Training Loss")
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Loss over Steps", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_facecolor('#F8F9FA')

    # Plot 2: Accuracy
    acc_metric = get_metric(metrics, "accuracy")
    if acc_metric:
        ax = axes[0, 1]
        values = np.array(acc_metric["values"]) * 100  # Convert to percentage
        ax.plot(acc_metric["steps"], values, linewidth=2, color=color, label="Accuracy")
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Random (50%)')
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title("Accuracy over Steps", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        ax.legend()
        ax.set_facecolor('#F8F9FA')

    # Plot 3: Margin
    margin_metric = get_metric(metrics, "margin")
    if margin_metric:
        ax = axes[1, 0]
        ax.plot(margin_metric["steps"], margin_metric["values"], linewidth=2, color=color, label="Margin")
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Margin (chosen - rejected)", fontsize=11)
        ax.set_title("Reward Margin over Steps", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_facecolor('#F8F9FA')

    # Plot 4: Mean Reward
    mean_reward_metric = get_metric(metrics, "mean_reward")
    if mean_reward_metric:
        ax = axes[1, 1]
        ax.plot(mean_reward_metric["steps"], mean_reward_metric["values"], linewidth=2, color=color, label="Mean Reward")
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Mean Reward", fontsize=11)
        ax.set_title("Mean Reward over Steps", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_facecolor('#F8F9FA')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{rm_name}_training_progress.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()


def plot_reward_statistics(rm_name, metrics, color):
    """Plot min, mean, max reward statistics"""
    print(f"  Creating reward statistics plot...")

    min_reward = get_metric(metrics, "min_reward")
    max_reward = get_metric(metrics, "max_reward")
    mean_reward = get_metric(metrics, "mean_reward")

    if not min_reward and not max_reward and not mean_reward:
        print(f"    WARNING: No reward statistics found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract steps
    steps = None
    if min_reward:
        steps = min_reward["steps"]
    elif max_reward:
        steps = max_reward["steps"]
    elif mean_reward:
        steps = mean_reward["steps"]

    if steps is None:
        print(f"    WARNING: Could not find steps")
        return

    # Plot min-max range
    if min_reward and max_reward:
        ax.fill_between(
            steps,
            min_reward["values"],
            max_reward["values"],
            alpha=0.2,
            label="Min-Max Range",
            color=color,
        )

    # Plot mean
    if mean_reward:
        ax.plot(
            mean_reward["steps"],
            mean_reward["values"],
            linewidth=2.5,
            label="Mean Reward",
            color=color,
        )

    # Plot min
    if min_reward:
        ax.plot(
            min_reward["steps"],
            min_reward["values"],
            linewidth=1.5,
            label="Min Reward",
            color=color,
            linestyle="--",
            alpha=0.7,
        )

    # Plot max
    if max_reward:
        ax.plot(
            max_reward["steps"],
            max_reward["values"],
            linewidth=1.5,
            label="Max Reward",
            color=color,
            linestyle=":",
            alpha=0.7,
        )

    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Reward Value", fontsize=12)
    ax.set_title(f'{rm_name.upper()} - Reward Statistics', fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_facecolor('#F8F9FA')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{rm_name}_reward_statistics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()


def print_metrics_summary(rm_name, metrics):
    """Print summary statistics"""
    print(f"\n  Metrics Summary:")

    for metric_name in ["train/loss", "train/accuracy", "train/margin", "train/mean_reward"]:
        if metric_name in metrics:
            values = np.array(metrics[metric_name]["values"])
            steps = metrics[metric_name]["steps"]

            display_name = metric_name.replace("train/", "")
            print(f"    {display_name}:")
            print(f"      Steps: {len(steps)} (from {steps[0]} to {steps[-1]})")
            print(f"      Final: {values[-1]:.4f}")
            if len(values) > 1:
                print(f"      Change: {values[-1] - values[0]:+.4f}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("REWARD MODEL TRAINING VISUALIZATION")
    print("="*80)

    for rm_name, event_file in EVENT_FILES.items():
        print(f"\n[{rm_name.upper()}]")

        if event_file is None:
            print("  SKIPPED: Training in progress")
            continue

        if not os.path.exists(event_file):
            print(f"  ERROR: Event file not found")
            continue

        # Load metrics
        print(f"  Loading metrics...")
        metrics = load_tensorboard_metrics(event_file)

        if not metrics:
            print(f"  ERROR: No metrics found")
            continue

        print(f"  Loaded {len(metrics)} metric types")

        # Determine color
        objective = rm_name.split('_')[0]
        color = COLORS.get(objective, '#666666')

        # Print summary
        print_metrics_summary(rm_name, metrics)

        # Generate plots
        plot_training_progress(rm_name, metrics, color)
        plot_reward_statistics(rm_name, metrics, color)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("\nGenerated plots per model:")
    print("  1. training_progress.png (2x2 grid: loss, accuracy, margin, mean_reward)")
    print("  2. reward_statistics.png (min/mean/max rewards)")


if __name__ == "__main__":
    main()
