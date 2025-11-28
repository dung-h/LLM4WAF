
import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model on a given dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/processed/diverse_sft_dataset.jsonl",
        help="Path to the training dataset in JSONL format.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="The name of the pre-trained model to fine-tune from Hugging Face.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to a local model to continue training from. Overrides model_name if specified.",
    )
    parser.add_argument(
        "--new_model_name",
        type=str,
        default="phi3-mini-sft-diverse-v1",
        help="The name for the newly fine-tuned model.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token for private models.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device during training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="The initial learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments/sft_finetuned_model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision training.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 mixed precision training (requires Ampere or newer GPU).",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default="data/processed/red_test_v6_multi_clean.jsonl",
        help="Path to the evaluation dataset in JSONL format.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10,
        help="Run evaluation every N steps.",
    )
    return parser.parse_args()

def format_prompt_phi3(sample):
    """Formats a sample from the dataset into the official Phi-3 prompt format."""
    instruction = sample.get('instruction', '')
    payload_val = sample.get('payload', '')

    # Defensively handle if payload is a list, joining it into a single string.
    if isinstance(payload_val, list):
        payload = " ".join(map(str, payload_val))
    else:
        payload = str(payload_val) # Ensure it's a string

    # Combine instruction for the user query
    user_query = f"{instruction}\n" # Add newline here

    # Phi-3 format
    model_prompt = f"<|user|>\n{user_query}<|end|>\n<|assistant|>\n"
    
    return [model_prompt + payload]

def main():
    """Main function to run the training script."""
    args = parse_args()

    # --- 1. Load Dataset ---
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"Dataset loaded with {len(dataset)} samples.")

    print(f"Loading evaluation dataset from: {args.eval_dataset_path}")
    eval_dataset = load_dataset("json", data_files=args.eval_dataset_path, split="train")
    print(f"Evaluation dataset loaded with {len(eval_dataset)} samples.")
    # --- 2. Model and Tokenizer Configuration ---
    model_path = args.base_model_path if args.base_model_path else args.model_name
    print(f"Loading model from: {model_path}")

    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        token=args.hf_token
    )
    model.config.use_cache = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 3. PEFT (LoRA) Configuration ---
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # --- 4. Training Arguments ---
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    # --- 5. Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=format_prompt_phi3,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # --- 6. Start Training ---
    print("Starting model training...")
    trainer.train()
    print("Training finished.")

    # --- 7. Save Model ---
    print(f"Saving trained model to {args.new_model_name}")
    trainer.model.save_pretrained(args.new_model_name)
    tokenizer.save_pretrained(args.new_model_name)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
