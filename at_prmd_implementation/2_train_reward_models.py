# AT-PRMD: Train Reward Models for 3 Objectives
import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType

# Memory and CUDA optimization
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

MAX_SEQ_LENGTH = 1024
BASE_MODEL = r"g:/Rabia-Salman/CPO/downloads/Qwen2.5-3B"
DATA_DIR = r"g:\Rabia-Salman\CPO\workspace\data\at_prmd"

OBJECTIVE = sys.argv[1] if len(sys.argv) > 1 else "harmless"
ENSEMBLE_IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUTPUT_DIR = rf"g:\Rabia-Salman\CPO\workspace\checkpoints\reward_model_{OBJECTIVE}_m{ENSEMBLE_IDX}"

if __name__ == '__main__':
    print(f"\n=== Training Reward Model: {OBJECTIVE.upper()} (Ensemble {ENSEMBLE_IDX}/2) ===")

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    print(f"Tokenizer: pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}, side={tokenizer.padding_side}")

    # Load dataset (TRL expects text columns: chosen/rejected)
    train_file = os.path.join(DATA_DIR, f"{OBJECTIVE}_train.json")
    print(f"Loading: {train_file}")
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    print(f"Train size: {len(train_dataset)}")

    # Format text for instruction format (TRL will tokenize)
    def format_text(example):
        return {
            "chosen": f"[INST] {example['prompt']} [/INST] {example['chosen']}",
            "rejected": f"[INST] {example['prompt']} [/INST] {example['rejected']}",
        }

    train_dataset = train_dataset.map(format_text, remove_columns=train_dataset.column_names)
    print("Dataset formatted")

    # Training config
    batch_size = 4
    grad_accum = 16
    epochs = 2
    total_steps = (len(train_dataset) // (batch_size * grad_accum)) * epochs
    save_steps = max(1, total_steps // 20)

    print(f"Steps: {total_steps}, Save every: {save_steps}, Effective batch: {batch_size * grad_accum}")

    # LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        use_rslora=True,
    )

    # Training args
    training_args = RewardConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, total_steps // 20),
        weight_decay=0.01,
        bf16=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=save_steps,
        eval_strategy="no",
        save_total_limit=3,
        dataloader_num_workers=0,
        report_to="tensorboard",
        save_safetensors=False,
        max_length=MAX_SEQ_LENGTH,
    )

    # Load model first (older TRL versions)
    from transformers import AutoModelForSequenceClassification
    from peft import get_peft_model

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map={"": 0},
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"Device: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # TRL RewardTrainer
    print("Initializing TRL RewardTrainer...")
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Resume or fresh
    import glob, shutil
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if not checkpoints:
        tb_dir = os.path.join(OUTPUT_DIR, "runs")
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)
        print("\n=== Starting fresh ===")
        trainer.train()
    else:
        print("\n=== Resuming from checkpoint ===")
        trainer.train(resume_from_checkpoint=True)

    # Save
    print(f"\nSaving to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[OK] {OBJECTIVE} m{ENSEMBLE_IDX} complete! Loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")
