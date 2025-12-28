# evaluation_.py (UPDATED)

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class MultiObjectiveEvaluator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def get_model_response(self, question: str, model, tokenizer, max_new_tokens: int = 512) -> str:
        """Generate response from your policy model."""
        
        messages = [{"role": "user", "content": question}]
        
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt = f"<s>[INST] {question} [/INST]"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def get_model_response_multiturn(self, turns: List[str], model, tokenizer, max_new_tokens: int = 512) -> List[str]:
        """Generate responses for multi-turn conversation."""
        conversation = []
        responses = []
        
        for turn in turns:
            conversation.append({"role": "user", "content": turn})
            
            if hasattr(tokenizer, 'apply_chat_template'):
                prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = "\n".join([f"User: {m['content']}" if m['role'] == 'user' else f"Assistant: {m['content']}" 
                                for m in conversation])
                prompt += "\nAssistant:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            responses.append(response)
            conversation.append({"role": "assistant", "content": response})
        
        return responses

    def evaluate_benchmark(self, benchmark_data: List[Dict], 
                        model,
                        tokenizer, 
                        objective: str,
                        output_path: str) -> Dict:
        """Evaluate model on a benchmark dataset."""
        results = []
        
        # For granular tracking
        if objective == "helpfulness":
            turn_scores = {"turn_1": [], "turn_2": []}
        elif objective == "honesty":
            domain_scores = defaultdict(list)
        elif objective == "harmlessness":
            level_scores = defaultdict(list)
        
        for item in tqdm(benchmark_data, desc=f"Evaluating {objective}"):
            # Handle multi-turn (MT-Bench)
            if "turns" in item and item.get("is_multi_turn"):
                turns = item["turns"]
                responses = self.get_model_response_multiturn(turns, model, tokenizer)
                
                # Evaluate EACH turn separately
                for turn_idx, (turn_question, turn_response) in enumerate(zip(turns, responses)):
                    score, reasoning = self.evaluate_with_gpt4(
                        question=turn_question,
                        response=turn_response,
                        objective=objective,
                        turn_index=turn_idx + 1  # 1-indexed
                    )
                    
                    turn_scores[f"turn_{turn_idx + 1}"].append(score)
                    
                    results.append({
                        "turn": turn_idx + 1,
                        "question": turn_question,
                        "model_response": turn_response,
                        "score": score,
                        "reasoning": reasoning
                    })
            
            else:
                # Single-turn evaluation
                question = item.get("question") or item.get("prompt") or item.get("text")
                
                if not question or question.strip() == "":
                    continue
                
                # Add task-specific instruction for summarization
                if objective == "honesty" and item.get('task_type') == 'summarization':
                    question = f"Please summarize the following document:\n\n{question}"
                
                model_response = self.get_model_response(question, model, tokenizer)
                
                # Evaluate
                score, reasoning = self.evaluate_with_gpt4(
                    question=question, 
                    response=model_response, 
                    objective=objective,
                    right_answer=item.get('right_answer'),
                    expected_completion=item.get('expected_completion')
                )
                
                result = {
                    "question": question,
                    "model_response": model_response,
                    "score": score,
                    "reasoning": reasoning
                }
                
                # Track granular scores
                if objective == "honesty":
                    domain = item.get('domain', 'unknown')
                    domain_scores[domain].append(score)
                    result['domain'] = domain
                    
                elif objective == "harmlessness":
                    level = item.get('level', 'unknown')
                    level_scores[f"level_{level}"].append(score)
                    result['level'] = level
                
                results.append(result)
        
        # Calculate metrics based on objective
        if objective == "helpfulness":
            output = {
                "objective": objective,
                "turn_1": sum(turn_scores["turn_1"]) / len(turn_scores["turn_1"]) if turn_scores["turn_1"] else 0,
                "turn_2": sum(turn_scores["turn_2"]) / len(turn_scores["turn_2"]) if turn_scores["turn_2"] else 0,
                "average": (sum(turn_scores["turn_1"]) + sum(turn_scores["turn_2"])) / 
                          (len(turn_scores["turn_1"]) + len(turn_scores["turn_2"])) if turn_scores["turn_1"] else 0,
                "total_samples": len(results),
                "detailed_results": results
            }
            
        elif objective == "honesty":
            all_scores = [r['score'] for r in results]
            domain_avgs = {domain: sum(scores)/len(scores) for domain, scores in domain_scores.items()}
            
            output = {
                "objective": objective,
                "average": sum(all_scores) / len(all_scores) if all_scores else 0,
                **domain_avgs,  # education, bio-medicine, etc.
                "total_samples": len(results),
                "detailed_results": results
            }
            
        elif objective == "harmlessness":
            all_scores = [r['score'] for r in results]
            level_avgs = {level: sum(scores)/len(scores) for level, scores in level_scores.items()}
            
            output = {
                "objective": objective,
                "average": sum(all_scores) / len(all_scores) if all_scores else 0,
                **level_avgs,  # level_0, level_1, etc.
                "total_samples": len(results),
                "detailed_results": results
            }
        
        # Save results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Print summary
        print(f"\n{objective.capitalize()} Results:")
        if objective == "helpfulness":
            print(f"  Turn 1: {output['turn_1']:.2f}/10")
            print(f"  Turn 2: {output['turn_2']:.2f}/10")
            print(f"  Average: {output['average']:.2f}/10")
        elif objective == "honesty":
            print(f"  Overall Average: {output['average']:.2f}/10")
            for domain, score in domain_avgs.items():
                print(f"    {domain}: {score:.2f}/10")
        elif objective == "harmlessness":
            print(f"  Overall Average: {output['average']:.2f}/10")
            for level, score in level_avgs.items():
                print(f"    {level}: {score:.2f}/10")
        
        return output    
    
    def evaluate_with_gpt4(self, question: str, response: str, objective: str, 
                        right_answer: str = None, expected_completion: str = None,
                        turn_index: int = None) -> Tuple[int, str]:
        """Evaluate a single response using GPT-4."""
        prompt_map = {
            "helpfulness": prompts.HELPFULNESS_PROMPT,
            "honesty": prompts.HONESTY_PROMPT,
            "harmlessness": prompts.HARMLESSNESS_PROMPT
        }
        
        # Format prompt based on objective
        if objective == "honesty":
            prompt = prompt_map[objective].format(
                question=question, 
                right_answer=right_answer,
                response=response
            )
        elif objective == "harmlessness":
            prompt = prompt_map[objective].format(
                question=question,
                expected_completion=expected_completion,
                response=response
            )
        else:  # helpfulness
            prompt = prompt_map[objective].format(
                question=question,
                response=response
            )
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            evaluation = completion.choices[0].message.content
            
            score_match = re.search(r"Score:\s*(\d+)", evaluation)
            score = int(score_match.group(1)) if score_match else 0
            
            reasoning_match = re.search(r"Reasoning:\s*(.+)", evaluation, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            return score, reasoning
            
        except Exception as e:
            print(f"Error evaluating: {e}")
            return 0, str(e)


def load_benchmark(benchmark_name: str, data_path: str) -> List[Dict]:
    """Load benchmark data from file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        if data_path.endswith('.json'):
            data = json.load(f)
        elif data_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    return data

import random

def sample_benchmark_data(data: List[Dict], n_samples: int, seed: int = 42) -> List[Dict]:
    """Sample n_samples from data with fixed seed for reproducibility."""
    random.seed(seed)
    if len(data) <= n_samples:
        return data
    return random.sample(data, n_samples)


def sample_hackaprompt_by_levels(data: List[Dict], levels: List[int] = [3, 6, 10], 
                                  samples_per_level: int = 100, seed: int = 42) -> List[Dict]:
    """Sample balanced samples from specific HackaPrompt levels."""
    random.seed(seed)
    
    # Group by level
    level_groups = {}
    for item in data:
        level = item.get('level')
        if level in levels:
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(item)
    
    print("\nHackaPrompt level distribution (selected levels):")
    for level in levels:
        count = len(level_groups.get(level, []))
        print(f"  Level {level}: {count} total samples")
    
    # Sample from each level
    sampled = []
    for level in levels:
        items = level_groups.get(level, [])
        n = min(samples_per_level, len(items))
        sampled.extend(random.sample(items, n))
        print(f"  Sampled {n} from Level {level}")
    
    print(f"\nTotal sampled: {len(sampled)}")
    return sampled


def save_summary(model_dir: Path, summary: Dict):
    """Save summary scores incrementally."""
    summary_path = model_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


def evaluate_model(model_name: str, model_subdirectory: str, evaluator: MultiObjectiveEvaluator, 
                   benchmarks: Dict, model_dir: str):
    """Evaluate a single model on all benchmarks."""
    
    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    print(f"{'='*60}")
    
    if model_subdirectory:
        tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=model_subdirectory)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, subfolder=model_subdirectory,
            torch_dtype=torch.float16, device_map="auto"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print(f"Model loaded on: {model.device}")
    
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    all_results = {}
    
    for objective, benchmark_path in benchmarks.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {objective.upper()}")
        print(f"{'='*50}")
        
        benchmark_data = load_benchmark(objective, benchmark_path)
        
        # Sample based on objective
        if objective == "harmlessness":
            # Sample 100 from each of levels 3, 6, 10
            benchmark_data = sample_hackaprompt_by_levels(
                benchmark_data, 
                levels=[3, 6, 10], 
                samples_per_level=100, 
                seed=42
            )
            print(f"Sampled {len(benchmark_data)} from HackaPrompt")

        elif objective == "honesty":
            # Use annotated dataset
            print(f"Using annotated dataset with {len(benchmark_data)} samples")
        elif objective == "helpfulness":
            print(f"Using all {len(benchmark_data)} MT-Bench pairs")
        
        # Run evaluation
        results = evaluator.evaluate_benchmark(
            benchmark_data=benchmark_data,
            model=model,
            tokenizer=tokenizer,
            objective=objective,
            output_path=str(model_path / f"{objective}_results.json")
        )
        
        all_results[objective] = results
        summary[objective] = results  # Store full dict with granular scores
        
        save_summary(model_path, summary)
    
    # Save combined detailed results
    with open(model_path / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS FOR {model_name}")
    print(f"{'='*60}")
    
    for obj, scores in summary.items():
        print(f"\n{obj.capitalize()}:")
        if obj == "helpfulness":
            print(f"  Turn 1: {scores['turn_1']:.2f}/10")
            print(f"  Turn 2: {scores['turn_2']:.2f}/10")
            print(f"  Average: {scores['average']:.2f}/10")
        else:
            print(f"  Overall: {scores['average']:.2f}/10")
    
    return summary


def create_comparison_chart(your_model_summary: Dict, baseline_summary: Dict, output_path: str = "results/comparison_chart.png"):
    """Create bar chart comparing two models (overall averages)."""
    
    objectives = ['helpfulness', 'honesty', 'harmlessness']
    your_scores = [your_model_summary[obj]['average'] for obj in objectives]
    baseline_scores = [baseline_summary[obj]['average'] for obj in objectives]
    
    x = np.arange(len(objectives))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, your_scores, width, label='Your DPO Model', color='#2ecc71')
    bars2 = ax.bar(x + width/2, baseline_scores, width, label='Qwen2.5-3B Baseline', color='#3498db')
    
    ax.set_ylabel('Score (0-10)', fontsize=12)
    ax.set_title('Multi-Objective Alignment Evaluation Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([obj.capitalize() for obj in objectives])
    ax.legend()
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved to {output_path}")
    plt.close()


def main():
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-4o-mini"
    
    # Use annotated dataset for honesty
    BENCHMARKS = {
        "helpfulness": r"D:\Salman Ahmed LUMS\Personal\LUMS\ATML\PROJECT\PRMD\data\mt_bench.json",
        "honesty": r"D:\Salman Ahmed LUMS\Personal\LUMS\ATML\PROJECT\PRMD\data\annotated_honesty.json", 
        "harmlessness": r"D:\Salman Ahmed LUMS\Personal\LUMS\ATML\PROJECT\PRMD\data\hackaprompt.json"
    }
    
    evaluator = MultiObjectiveEvaluator(api_key=API_KEY, model=MODEL_NAME)
    
    # Evaluate your DPO model
    your_model_summary = evaluate_model(
        model_name="Aizelsheikh/Reward-Guided-DPO-Model",
        model_subdirectory="Reward-Guided-DPO-model",
        evaluator=evaluator,
        benchmarks=BENCHMARKS,
        model_dir="results/your_model"
    )
    
    # Evaluate Qwen baseline
    baseline_summary = evaluate_model(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        model_subdirectory=None,
        evaluator=evaluator,
        benchmarks=BENCHMARKS,
        model_dir="results/qwen_baseline"
    )
    
    create_comparison_chart(your_model_summary, baseline_summary)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()