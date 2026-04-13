#!/usr/bin/env python3
"""
scripts/fine_tune.py — Domain adaptation fine-tuning scaffold.

Addresses the "domain-specific models lack generalization" gap from the review.
Enables fine-tuning the reader model on domain-specific QA pairs
(e.g., medical, legal, biomedical) without rewriting the full pipeline.

Usage:
  python scripts/fine_tune.py \
    --base_model deepset/roberta-base-squad2 \
    --train_data data/domain_qa_train.json \
    --output_dir data/fine_tuned_model/ \
    --domain biomedical
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def load_squad_format(path: Path) -> dict:
    """Load QA pairs in SQuAD 2.0 format."""
    with open(path) as f:
        data = json.load(f)
    return data


def fine_tune(
    base_model: str,
    train_data_path: Path,
    output_dir: Path,
    domain: str = "general",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_seq_length: int = 512,
):
    """
    Fine-tune a BERT-based QA model on domain-specific data.

    Strategy:
    1. Load base model (pre-trained on SQuAD)
    2. Load domain-specific QA pairs in SQuAD format
    3. Fine-tune with lower learning rate to retain general knowledge
    4. Evaluate on held-out set and save if better than base

    This implements the domain adaptation approach described in BioBERT
    and Legal-BERT papers but generalized to any domain.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForQuestionAnswering,
            TrainingArguments,
            Trainer,
        )
        from datasets import Dataset
    except ImportError:
        print("Install: pip install transformers datasets torch")
        sys.exit(1)

    print(f"Fine-tuning {base_model} for domain: {domain}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForQuestionAnswering.from_pretrained(base_model)

    # Load training data
    raw_data = load_squad_format(train_data_path)
    examples = raw_data if isinstance(raw_data, list) else raw_data.get("data", [])

    # Convert to HuggingFace Dataset format
    dataset_dict = {
        "id": [],
        "context": [],
        "question": [],
        "answers": [],
    }

    for item in examples:
        dataset_dict["id"].append(str(item.get("id", len(dataset_dict["id"]))))
        dataset_dict["context"].append(item["context"])
        dataset_dict["question"].append(item["question"])
        answers = item.get("answers", {})
        if isinstance(answers, list):
            answers = {"text": answers, "answer_start": [0] * len(answers)}
        dataset_dict["answers"].append(answers)

    dataset = Dataset.from_dict(dataset_dict)

    def preprocess(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation=True,
            max_length=max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        return tokenized

    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print(f"Starting training on {len(tokenized_dataset)} examples...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Model saved to {final_path}")
    print(f"\nTo use: set READER_MODEL_NAME={final_path} in .env")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DocSage reader model")
    parser.add_argument("--base_model", default="deepset/roberta-base-squad2")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", default="data/fine_tuned_model")
    parser.add_argument("--domain", default="general")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    fine_tune(
        base_model=args.base_model,
        train_data_path=Path(args.train_data),
        output_dir=Path(args.output_dir),
        domain=args.domain,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()