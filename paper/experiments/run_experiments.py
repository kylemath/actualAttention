"""
Run all experiments for RNN, LSTM, and Transformer (baseline vs bio-enhanced).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import (
    DEVICE,
    MODES,
    TRAINING,
    RNN_CONFIG,
    LSTM_CONFIG,
    TRANSFORMER_CONFIG,
    BIO_MECHANISMS,
    TASKS,
    SEEDS,
    PATHS,
)
from utils import (
    set_seed,
    save_results,
    count_parameters,
    compute_gradient_norm,
    get_memory_usage,
    format_time,
)
from tasks import (
    create_copy_task_dataloaders,
    create_language_model_dataloaders,
    create_sequential_mnist_dataloaders,
    evaluate_copy_accuracy,
)
from models import (
    StandardRNN,
    BioEnhancedRNN,
    StandardLSTM,
    BioEnhancedLSTM,
    StandardTransformer,
    BioEnhancedTransformer,
    StandardRNNClassifier,
    BioEnhancedRNNClassifier,
    StandardLSTMClassifier,
    BioEnhancedLSTMClassifier,
    StandardTransformerClassifier,
    BioEnhancedTransformerClassifier,
)
from models.transformer import create_causal_mask


def build_model(
    arch,
    task,
    d_model,
    num_heads,
    vocab_size=None,
    input_dim=None,
    num_classes=None,
    bio=False,
):
    """
    Build model for a given architecture and task.
    """
    if task in ["copy", "language_model"]:
        if arch == "rnn":
            model_cls = BioEnhancedRNN if bio else StandardRNN
            return model_cls(vocab_size, vocab_size, RNN_CONFIG["hidden_dim"], RNN_CONFIG["num_layers"], RNN_CONFIG["dropout"])
        if arch == "lstm":
            model_cls = BioEnhancedLSTM if bio else StandardLSTM
            return model_cls(vocab_size, vocab_size, LSTM_CONFIG["hidden_dim"], LSTM_CONFIG["num_layers"], LSTM_CONFIG["dropout"])
        if arch == "transformer":
            if bio:
                return BioEnhancedTransformer(
                    vocab_size,
                    d_model,
                    num_heads,
                    TRANSFORMER_CONFIG["num_layers"],
                    TRANSFORMER_CONFIG["dim_feedforward"],
                    TRANSFORMER_CONFIG["dropout"],
                    BIO_MECHANISMS,
                )
            return StandardTransformer(
                vocab_size,
                d_model,
                num_heads,
                TRANSFORMER_CONFIG["num_layers"],
                TRANSFORMER_CONFIG["dim_feedforward"],
                TRANSFORMER_CONFIG["dropout"],
            )

    if task == "sequential_mnist":
        if arch == "rnn":
            model_cls = BioEnhancedRNNClassifier if bio else StandardRNNClassifier
            return model_cls(input_dim, RNN_CONFIG["hidden_dim"], RNN_CONFIG["num_layers"], num_classes, RNN_CONFIG["dropout"])
        if arch == "lstm":
            model_cls = BioEnhancedLSTMClassifier if bio else StandardLSTMClassifier
            return model_cls(input_dim, LSTM_CONFIG["hidden_dim"], LSTM_CONFIG["num_layers"], num_classes, LSTM_CONFIG["dropout"])
        if arch == "transformer":
            if bio:
                return BioEnhancedTransformerClassifier(
                    input_dim,
                    d_model,
                    num_heads,
                    TRANSFORMER_CONFIG["num_layers"],
                    TRANSFORMER_CONFIG["dim_feedforward"],
                    num_classes,
                    TRANSFORMER_CONFIG["dropout"],
                    BIO_MECHANISMS,
                )
            return StandardTransformerClassifier(
                input_dim,
                d_model,
                num_heads,
                TRANSFORMER_CONFIG["num_layers"],
                TRANSFORMER_CONFIG["dim_feedforward"],
                num_classes,
                TRANSFORMER_CONFIG["dropout"],
            )

    raise ValueError(f"Unsupported combination: {arch} + {task}")


def train_epoch(model, loader, optimizer, criterion, task, device, src_mask=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    grad_norms = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if task == "sequential_mnist":
            # inputs: (batch, 784) -> (batch, 784, 1)
            inputs = inputs.unsqueeze(-1)
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.size(0)
        else:
            # language modeling / copy task
            outputs = model(inputs, src_mask) if src_mask is not None else model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            vocab_size = outputs.size(-1)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            preds = outputs.argmax(dim=-1)
            if task == "copy":
                batch_acc = evaluate_copy_accuracy(preds, targets)
                total_correct += batch_acc * targets.size(0)
                total_count += targets.size(0)
            else:
                total_correct += (preds == targets).sum().item()
                total_count += targets.numel()

        # Bio-enhanced models may add auxiliary loss
        if hasattr(model, "get_total_loss"):
            loss = model.get_total_loss(loss)

        loss.backward()
        grad_norms.append(compute_gradient_norm(model))
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["grad_clip"])
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy, float(np.mean(grad_norms))


@torch.no_grad()
def eval_epoch(model, loader, criterion, task, device, src_mask=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if task == "sequential_mnist":
            inputs = inputs.unsqueeze(-1)
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.size(0)
        else:
            outputs = model(inputs, src_mask) if src_mask is not None else model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            vocab_size = outputs.size(-1)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            preds = outputs.argmax(dim=-1)
            if task == "copy":
                batch_acc = evaluate_copy_accuracy(preds, targets)
                total_correct += batch_acc * targets.size(0)
                total_count += targets.size(0)
            else:
                total_correct += (preds == targets).sum().item()
                total_count += targets.numel()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def run_single_experiment(arch, task, mode, bio, seed):
    set_seed(seed)
    config = MODES[mode]
    lm_mask = None

    # Setup data
    if task == "copy":
        train_loader, val_loader, vocab_size = create_copy_task_dataloaders(
            config["num_sequences"],
            max(config["num_sequences"] // 5, 100),
            TASKS["copy"]["sequence_length"],
            TASKS["copy"]["vocab_size"],
            config["batch_size"],
            TASKS["copy"]["blank_length"],
        )
        model = build_model(
            arch,
            task,
            d_model=config["model_dim"],
            num_heads=config["num_heads"],
            vocab_size=vocab_size,
            bio=bio,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    elif task == "language_model":
        train_loader, val_loader, vocab_size = create_language_model_dataloaders(
            data_dir="data/enwik8",
            seq_len=TASKS["language_model"]["sequence_length"],
            batch_size=config["batch_size"],
        )
        model = build_model(
            arch,
            task,
            d_model=config["model_dim"],
            num_heads=config["num_heads"],
            vocab_size=vocab_size,
            bio=bio,
        )
        criterion = nn.CrossEntropyLoss()
        lm_mask = None
        if arch == "transformer":
            seq_len = TASKS["language_model"]["sequence_length"]
            lm_mask = create_causal_mask(seq_len, DEVICE)

    elif task == "sequential_mnist":
        train_loader, val_loader, input_dim = create_sequential_mnist_dataloaders(
            data_dir="data/mnist",
            batch_size=config["batch_size"],
            permute=TASKS["sequential_mnist"]["permute"],
        )
        model = build_model(
            arch,
            task,
            d_model=config["model_dim"],
            num_heads=config["num_heads"],
            input_dim=1,
            num_classes=10,
            bio=bio,
        )
        criterion = nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unknown task: {task}")

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )

    results = {
        "arch": arch,
        "task": task,
        "bio": bio,
        "seed": seed,
        "num_params": count_parameters(model),
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "grad_norm": [],
        "memory_mb": [],
    }

    start_time = time.time()
    for epoch in range(config["epochs"]):
        src_mask = lm_mask if task == "language_model" else None
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, optimizer, criterion, task, DEVICE, src_mask=src_mask
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, task, DEVICE, src_mask=src_mask
        )

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_acc"].append(train_acc)
        results["val_acc"].append(val_acc)
        results["grad_norm"].append(grad_norm)
        results["memory_mb"].append(get_memory_usage())

        if epoch % 10 == 0 or epoch == config["epochs"] - 1:
            elapsed = format_time(time.time() - start_time)
            print(
                f"[{arch.upper()}|{task}|{'bio' if bio else 'base'}] "
                f"Epoch {epoch+1}/{config['epochs']} "
                f"loss={train_loss:.4f} val={val_loss:.4f} "
                f"acc={val_acc:.3f} time={elapsed}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run experiments across architectures")
    parser.add_argument("--mode", type=str, default="fast", choices=MODES.keys())
    parser.add_argument("--tasks", nargs="+", default=["copy", "language_model", "sequential_mnist"])
    parser.add_argument("--archs", nargs="+", default=["rnn", "lstm", "transformer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--run-id", type=str, default="run")
    parser.add_argument("--output-dir", type=str, default=PATHS["data"])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    for task in args.tasks:
        for arch in args.archs:
            for bio in [False, True]:
                for seed in args.seeds:
                    result = run_single_experiment(arch, task, args.mode, bio, seed)
                    tag = f"{args.run_id}_{arch}_{task}_{'bio' if bio else 'base'}_seed{seed}"
                    save_results(result, f"{tag}.pkl", base_dir=args.output_dir)
                    all_results.append(result)

    # Save summary
    summary = {
        "num_runs": len(all_results),
        "tasks": args.tasks,
        "archs": args.archs,
        "mode": args.mode,
    }
    save_results(summary, f"{args.run_id}_summary.pkl", base_dir=args.output_dir)


if __name__ == "__main__":
    main()
