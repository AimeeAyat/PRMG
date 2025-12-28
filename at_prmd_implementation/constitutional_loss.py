# Constitutional Constraint Loss Functions for AT-PRMD
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class ConstitutionalRMDLoss(nn.Module):
    """
    Constitutional Reward Model Distillation Loss with Pessimistic Optimization

    L_total = L_distill + α * KL + β * Σ_k C_k(θ)

    Where:
    - L_distill: Match implicit reward to pessimistic (min) explicit reward
    - KL: KL divergence from reference policy
    - C_k(θ): Constitutional constraint violation penalty for objective k
    """

    def __init__(self, alpha=0.1, beta=1.0, thresholds=None, debug=False, cvar_alpha=0.5):
        super().__init__()
        self.alpha = alpha  # KL penalty weight
        self.beta = beta    # Constraint penalty weight
        self.debug = debug  # Enable debug logging
        self.step_count = 0  # Track training steps for debug output
        self.cvar_alpha = cvar_alpha  # CVaR alpha (None = hard min, 0.5 = worst 50%, 1.0 = mean)

        # Constitutional thresholds τ_k for each objective
        self.thresholds = thresholds or {
            "harmless": 0.75,  # High bar for safety
            "helpful": 0.65,   # Medium bar
            "honest": 0.70,    # High bar for truthfulness 
        }

        # Log aggregation method
        if self.cvar_alpha is None:
            print(f"Using HARD MIN (most pessimistic) for multi-objective aggregation")
        else:
            print(f"Using CVaR-{self.cvar_alpha} (average of worst {self.cvar_alpha*100:.0f}%) for multi-objective aggregation")

    def compute_implicit_reward(self, policy_logprobs, ref_logprobs, beta_kl=0.1):
        """
        Compute implicit reward from policy:
        r_θ(x,y) = β * log(π_θ(y|x) / π_ref(y|x))
        """
        return beta_kl * (policy_logprobs - ref_logprobs)

    def compute_kl_divergence(self, policy_logprobs, ref_logprobs):
        """
        KL(π_θ || π_ref) = E[log π_θ - log π_ref]
        """
        return (policy_logprobs - ref_logprobs).mean()

    def compute_pessimistic_reward(self, reward_dict: Dict[str, torch.Tensor]):
        """
        Pessimistic aggregation with ensemble:
        1. For each objective k, take min across M models: min_m r_φk,m(x, y)
        2. Then aggregate across objectives:
           - If cvar_alpha is None: min_k [min_m r_φk,m(x, y)] (hard min)
           - If cvar_alpha is set: CVaR_α over objectives (average of worst α%)

        reward_dict format: {"harmless_m0": tensor, "harmless_m1": tensor, ...}
        """
        # Group by objective (assuming format: "{objective}_m{idx}")
        objectives = {}
        for key, reward in reward_dict.items():
            obj_name = key.rsplit('_m', 1)[0]  # Extract objective name
            if obj_name not in objectives:
                objectives[obj_name] = []
            objectives[obj_name].append(reward)

        # Take min across ensemble for each objective
        pessimistic_per_objective = []
        for obj_name, ensemble_rewards in objectives.items():
            min_reward = torch.stack(ensemble_rewards).min(dim=0)[0]
            pessimistic_per_objective.append(min_reward)

        # Stack objectives: [num_objectives, batch_size]
        stacked_objectives = torch.stack(pessimistic_per_objective)

        # Aggregate across objectives
        if self.cvar_alpha is None:
            # Hard min (most pessimistic)
            return stacked_objectives.min(dim=0)[0]
        else:
            # CVaR-alpha: average of worst alpha% objectives
            num_objectives = stacked_objectives.shape[0]
            k = max(1, int(num_objectives * self.cvar_alpha))  # At least 1 objective

            # Sort along objective dimension (dim=0)
            sorted_rewards, _ = torch.sort(stacked_objectives, dim=0)

            # Take worst k objectives
            worst_k = sorted_rewards[:k]  # [k, batch_size]

            # Average the worst k
            cvar_reward = worst_k.mean(dim=0)  # [batch_size]

            return cvar_reward

    def compute_constitutional_constraints(self, reward_dict: Dict[str, torch.Tensor]):
        """
        Constitutional constraint violation penalty with ensemble:
        C_k(θ) = max(0, τ_k - min_m E[r_φk,m(x,y)])

        For each objective, take min across ensemble, then penalize if below threshold
        """
        constraint_loss = 0.0
        violations = {}

        # Group by objective
        objectives = {}
        for key, reward in reward_dict.items():
            obj_name = key.rsplit('_m', 1)[0]
            if obj_name not in objectives:
                objectives[obj_name] = []
            objectives[obj_name].append(reward)

        # For each objective, take min across ensemble
        for obj_name, ensemble_rewards in objectives.items():
            # Take min across ensemble members
            min_ensemble_reward = torch.stack(ensemble_rewards).min(dim=0)[0]
            expected_reward = min_ensemble_reward.mean()

            threshold = self.thresholds.get(obj_name, 0.5)

            # Compute violation
            violation = F.relu(threshold - expected_reward)
            constraint_loss += violation
            violations[obj_name] = violation.item()

        return constraint_loss, violations

    def forward(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        reward_dict: Dict[str, torch.Tensor],
        return_metrics: bool = False
    ):
        """
        Compute full constitutional RMD loss

        Args:
            policy_logprobs: Log probabilities from policy model
            ref_logprobs: Log probabilities from reference model
            reward_dict: Dictionary of rewards from each objective RM
                        {"harmless": tensor, "helpful": tensor, "honest": tensor}
            return_metrics: Whether to return detailed metrics

        Returns:
            loss: Total loss
            metrics: Dict of loss components (if return_metrics=True)
        """
        # 1. Compute implicit reward from policy
        implicit_reward = self.compute_implicit_reward(policy_logprobs, ref_logprobs)

        # 2. Compute pessimistic explicit reward (min across objectives)
        pessimistic_reward = self.compute_pessimistic_reward(reward_dict)

        # 3. Distillation loss: match implicit to pessimistic
        distillation_loss = F.mse_loss(pessimistic_reward, implicit_reward)

        # 4. KL divergence regularization
        kl_loss = self.compute_kl_divergence(policy_logprobs, ref_logprobs)

        # 5. Constitutional constraint penalties
        constraint_loss, violations = self.compute_constitutional_constraints(reward_dict)

        # Total loss
        total_loss = distillation_loss + self.alpha * kl_loss + self.beta * constraint_loss

        # Debug logging (every 10 steps)
        if self.debug and self.step_count % 10 == 0:
            print(f"\n[DEBUG Step {self.step_count}] Constitutional RMD Loss Breakdown:")
            print(f"  Implicit Reward: mean={implicit_reward.mean().item():.4f}, std={implicit_reward.std().item():.4f}")
            print(f"  Pessimistic Reward: mean={pessimistic_reward.mean().item():.4f}, std={pessimistic_reward.std().item():.4f}")
            print(f"  Distillation Loss: {distillation_loss.item():.4f}")
            print(f"  KL Loss: {kl_loss.item():.4f} (weighted: {(self.alpha * kl_loss).item():.4f})")
            print(f"  Constraint Loss: {constraint_loss.item():.4f} (weighted: {(self.beta * constraint_loss).item():.4f})")
            print(f"  Violations: {violations}")
            print(f"  TOTAL LOSS: {total_loss.item():.4f}")

        self.step_count += 1

        if return_metrics:
            metrics = {
                "loss/total": total_loss.item(),
                "loss/distillation": distillation_loss.item(),
                "loss/kl": kl_loss.item(),
                "loss/constraint": constraint_loss.item(),
                "reward/implicit_mean": implicit_reward.mean().item(),
                "reward/pessimistic_mean": pessimistic_reward.mean().item(),
            }

            # Add per-ensemble-model rewards
            for key, reward in reward_dict.items():
                metrics[f"reward/{key}_mean"] = reward.mean().item()

            # Group by objective and add min across ensemble
            objectives = {}
            for key, reward in reward_dict.items():
                obj_name = key.rsplit('_m', 1)[0]
                if obj_name not in objectives:
                    objectives[obj_name] = []
                objectives[obj_name].append(reward)

            for obj_name, ensemble_rewards in objectives.items():
                min_reward = torch.stack(ensemble_rewards).min(dim=0)[0]
                metrics[f"reward/{obj_name}_ensemble_min"] = min_reward.mean().item()
                metrics[f"violation/{obj_name}"] = violations[obj_name]

            return total_loss, metrics

        return total_loss


class RewardModelEnsemble(nn.Module):
    """
    Ensemble of reward models for multiple objectives
    Each objective has M reward models for ensemble uncertainty

    Supports two modes:
    1. In-memory mode (use_manager=False): All models loaded in memory (OOM risk!)
    2. Manager mode (use_manager=True): On-demand loading with RewardModelManager (memory-efficient)
    """

    def __init__(self, reward_models: Dict[str, nn.Module], use_manager=False, manager=None):
        super().__init__()
        self.use_manager = use_manager

        if use_manager:
            # Manager mode: Store manager reference, not models
            if manager is None:
                raise ValueError("Manager must be provided when use_manager=True")
            self.manager = manager
            self.reward_models = None
            self.objectives = list(reward_models.keys()) if reward_models else []
        else:
            # In-memory mode: Store all models (original behavior)
            self.reward_models = nn.ModuleDict(reward_models)
            self.objectives = list(reward_models.keys())
            self.manager = None

    def forward(self, input_ids, attention_mask=None):
        """
        Compute rewards from all objective models

        Returns:
            Dict[str, torch.Tensor]: Rewards for each objective
        """
        if self.use_manager:
            # Manager mode: Load models on-demand
            return self.manager.get_all_rewards(input_ids, attention_mask)
        else:
            # In-memory mode: Use pre-loaded models
            rewards = {}
            for objective, model in self.reward_models.items():
                with torch.no_grad():
                    reward = model(input_ids=input_ids, attention_mask=attention_mask)
                    rewards[objective] = reward
            return rewards

    @torch.no_grad()
    def get_pessimistic_reward(self, input_ids, attention_mask=None):
        """
        Get pessimistic (minimum) reward across all objectives
        """
        rewards = self.forward(input_ids, attention_mask)
        return torch.stack(list(rewards.values())).min(dim=0)[0]
