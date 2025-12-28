# Memory-Efficient Reward Model Manager
# Handles loading/offloading of multiple reward models to prevent OOM

import torch
import torch.nn as nn
from typing import Dict, List
import os
import gc


class RewardModelManager:
    """
    Manages multiple reward models with automatic loading/offloading to prevent OOM

    Strategies:
    1. Keep only 1-2 models on GPU at a time
    2. Offload unused models to CPU
    3. Load models on-demand
    4. Clear CUDA cache between swaps
    """

    def __init__(
        self,
        model_paths: Dict[str, str],
        base_model_class,
        device: str = "cuda",
        max_models_on_gpu: int = 2,
        use_cpu_offload: bool = True,
        use_8bit: bool = False
    ):
        """
        Args:
            model_paths: Dict mapping model_id to checkpoint path
                        e.g., {"harmless_m0": "path/to/model", ...}
            base_model_class: Class to instantiate reward models
            device: Primary device (cuda/cpu)
            max_models_on_gpu: Maximum number of models to keep on GPU simultaneously
            use_cpu_offload: If True, offload unused models to CPU instead of reloading
            use_8bit: If True, load models in 8-bit quantization (saves ~7GB per model, no accuracy loss)
        """
        self.model_paths = model_paths
        self.base_model_class = base_model_class
        self.device = device
        self.max_models_on_gpu = max_models_on_gpu
        self.use_cpu_offload = use_cpu_offload
        self.use_8bit = use_8bit

        # Model cache: {model_id: (model, device)}
        self.loaded_models = {}
        self.model_queue = []  # Track loading order for LRU eviction

        print(f"\n[Reward Model Manager] Initialized")
        print(f"  Device: {device}")
        print(f"  Max models on GPU: {max_models_on_gpu}")
        print(f"  CPU offloading: {use_cpu_offload}")
        print(f"  8-bit quantization: {use_8bit}")
        print(f"  Total models to manage: {len(model_paths)}")

    def _evict_if_needed(self):
        """Evict oldest model if we exceed max_models_on_gpu"""
        models_on_gpu = sum(
            1 for model, dev in self.loaded_models.values()
            if str(dev).startswith('cuda')
        )

        while models_on_gpu >= self.max_models_on_gpu and self.model_queue:
            # Evict oldest model
            oldest_id = self.model_queue.pop(0)
            if oldest_id in self.loaded_models:
                model, current_device = self.loaded_models[oldest_id]

                if self.use_cpu_offload and str(current_device).startswith('cuda'):
                    # Move to CPU
                    print(f"  [Offload] {oldest_id}: GPU -> CPU")
                    model.to('cpu')
                    self.loaded_models[oldest_id] = (model, 'cpu')
                    models_on_gpu -= 1
                else:
                    # Delete model
                    print(f"  [Evict] {oldest_id}: Deleting from memory")
                    del self.loaded_models[oldest_id]
                    del model
                    models_on_gpu -= 1

                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    def load_model(self, model_id: str) -> nn.Module:
        """
        Load a specific reward model (on-demand loading with caching)

        Args:
            model_id: Model identifier (e.g., "harmless_m0")

        Returns:
            Model loaded on self.device
        """
        # Check if already loaded
        if model_id in self.loaded_models:
            model, current_device = self.loaded_models[model_id]

            # Move to GPU if on CPU
            if str(current_device) != str(self.device) and self.device != 'cpu':
                print(f"  [Load] {model_id}: CPU -> {self.device}")
                model.to(self.device)
                self.loaded_models[model_id] = (model, self.device)

                # Update queue
                if model_id in self.model_queue:
                    self.model_queue.remove(model_id)
                self.model_queue.append(model_id)

            return model

        # Model not loaded - need to evict if necessary
        self._evict_if_needed()

        # Load model
        print(f"  [Load] {model_id}: Loading from disk -> {self.device}")
        model_path = self.model_paths[model_id]

        # Load model (will be implemented by caller)
        model = self._load_checkpoint(model_path, model_id)
        model.to(self.device)
        model.eval()  # Set to eval mode

        # Cache
        self.loaded_models[model_id] = (model, self.device)
        self.model_queue.append(model_id)

        return model

    def _load_checkpoint(self, model_path: str, model_id: str) -> nn.Module:
        """Load AT-PRMD reward model (SequenceClassification with LoRA)"""
        from transformers import AutoModelForSequenceClassification
        from peft import PeftModel

        load_in_8bit = self.use_8bit and self.device != "cpu"
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_peft_model = os.path.exists(adapter_config_path)

        if is_peft_model:
            print(f"    Loading PEFT SequenceClassification from {model_path}")
            model = PeftModel.from_pretrained(
                AutoModelForSequenceClassification.from_pretrained(
                    "g:/Rabia-Salman/CPO/downloads/Qwen2.5-3B",
                    num_labels=1,
                    torch_dtype=torch.bfloat16 if not load_in_8bit else None,
                    attn_implementation="sdpa",
                    device_map=None,
                    low_cpu_mem_usage=True,
                    load_in_8bit=load_in_8bit,
                    trust_remote_code=True,
                ),
                model_path,
                is_trainable=False
            )
            # ADD THIS BLOCK
            if model.config.pad_token_id is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                model.config.pad_token_id = tokenizer.pad_token_id
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=1,
                torch_dtype=torch.bfloat16 if not load_in_8bit else None,
                attn_implementation="sdpa",
                device_map=None,
                low_cpu_mem_usage=True,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
            )
            # ADD THIS BLOCK
            if model.config.pad_token_id is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                model.config.pad_token_id = tokenizer.pad_token_id

        model.eval()
        return model

    def get_rewards(
        self,
        model_ids: List[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rewards from multiple models efficiently

        Strategy: Load models sequentially, compute rewards, cache results

        Args:
            model_ids: List of model IDs to use
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dict mapping model_id to reward tensor [batch_size]
        """
        rewards = {}

        # Move inputs to CPU if using offloading (we'll move per model)
        if self.use_cpu_offload:
            input_ids_cpu = input_ids.cpu()
            attention_mask_cpu = attention_mask.cpu() if attention_mask is not None else None

        for model_id in model_ids:
            # Load model (handles offloading automatically)
            model = self.load_model(model_id)

            # Move inputs to model's device
            inputs_device = input_ids if not self.use_cpu_offload else input_ids_cpu.to(model.device)
            mask_device = attention_mask if attention_mask is None or not self.use_cpu_offload else attention_mask_cpu.to(model.device)

            # Compute reward (SequenceClassification returns logits)
            with torch.no_grad():
                outputs = model(input_ids=inputs_device, attention_mask=mask_device)
                reward = outputs.logits.squeeze(-1)  # Extract scalar reward from logits

            # Move reward back to original device and cache
            rewards[model_id] = reward.to(input_ids.device)

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return rewards

    def get_all_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards from ALL managed models"""
        return self.get_rewards(
            list(self.model_paths.keys()),
            input_ids,
            attention_mask
        )

    def clear_cache(self):
        """Clear all loaded models and free memory"""
        print("\n[Reward Model Manager] Clearing all models...")
        for model_id in list(self.loaded_models.keys()):
            model, _ = self.loaded_models[model_id]
            del model
            del self.loaded_models[model_id]

        self.model_queue.clear()

        # FIX: Check if torch still exists (might be None during shutdown)
        if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if gc is not None:
            gc.collect()

        print("  [OK] All models cleared")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if hasattr(self, 'loaded_models'):
                self.clear_cache()
        except:
            pass  # Ignore errors during shutdown

# Example usage
if __name__ == "__main__":
    print("Reward Model Manager - Memory Optimization")
    print("\nThis module provides:")
    print("  1. On-demand model loading")
    print("  2. Automatic CPU offloading")
    print("  3. LRU eviction policy")
    print("  4. Memory-efficient batch inference")
    print("\nUsage:")
    print("""
    # Initialize manager
    manager = RewardModelManager(
        model_paths={
            "harmless_m0": "path/to/harmless_m0",
            "harmless_m1": "path/to/harmless_m1",
            ...
        },
        base_model_class=YourRewardModelClass,
        max_models_on_gpu=2,  # Keep max 2 models on GPU
        use_cpu_offload=True
    )

    # Compute rewards (models loaded/offloaded automatically)
    rewards = manager.get_all_rewards(input_ids, attention_mask)
    # Returns: {"harmless_m0": tensor, "harmless_m1": tensor, ...}

    # Or load specific models
    rewards = manager.get_rewards(
        ["harmless_m0", "helpful_m0"],
        input_ids,
        attention_mask
    )
    """)
