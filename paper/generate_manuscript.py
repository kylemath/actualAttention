"""
Generate figures and manuscript from saved experiment data.
"""
import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("data")
FIG_DIR = Path("figures")
TEX_DIR = Path("tex")


def load_results(data_dir: Path):
    results = []
    for pkl_file in data_dir.glob("*.pkl"):
        if pkl_file.name.endswith("_summary.pkl"):
            continue
        with open(pkl_file, "rb") as f:
            results.append(pickle.load(f))
    return results


def results_to_dataframe(results):
    rows = []
    for r in results:
        rows.append({
            "arch": r["arch"],
            "task": r["task"],
            "bio": "bio" if r["bio"] else "base",
            "seed": r["seed"],
            "final_val_loss": r["val_loss"][-1] if r["val_loss"] else np.nan,
            "final_val_acc": r["val_acc"][-1] if r["val_acc"] else np.nan,
            "epochs": len(r["val_loss"]),
        })
    return pd.DataFrame(rows)


def plot_performance_bars(df, fig_dir: Path):
    """
    Multi-panel bar chart: final loss by task and architecture (base vs bio).
    """
    tasks = sorted(df["task"].unique())
    archs = ["rnn", "lstm", "transformer"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(14, 4), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        sub = df[df["task"] == task]
        means = sub.groupby(["arch", "bio"])["final_val_loss"].mean().unstack()
        means = means.reindex(archs)
        means.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(f"{task} (final loss)")
        ax.set_xlabel("")
        ax.set_ylabel("Loss")
        ax.legend(title="")

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "figure1_architecture_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_convergence_curves(results, fig_dir: Path):
    """
    Plot convergence curves for each architecture on the copy task.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    archs = ["rnn", "lstm", "transformer"]

    for ax, arch in zip(axes, archs):
        base_runs = [r for r in results if r["arch"] == arch and r["task"] == "copy" and not r["bio"]]
        bio_runs = [r for r in results if r["arch"] == arch and r["task"] == "copy" and r["bio"]]

        if base_runs:
            base_mean = np.mean([r["train_loss"] for r in base_runs], axis=0)
            ax.plot(base_mean, label="Base", linewidth=2)

        if bio_runs:
            bio_mean = np.mean([r["train_loss"] for r in bio_runs], axis=0)
            ax.plot(bio_mean, label="Bio", linewidth=2)

        ax.set_title(f"{arch.upper()} (copy task)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.legend()

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "figure2_convergence_curves.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_accuracy_summary(df, fig_dir: Path):
    """
    Multi-panel accuracy comparison by task.
    """
    tasks = sorted(df["task"].unique())
    archs = ["rnn", "lstm", "transformer"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(14, 4), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        sub = df[df["task"] == task]
        means = sub.groupby(["arch", "bio"])["final_val_acc"].mean().unstack()
        means = means.reindex(archs)
        means.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(f"{task} (final accuracy)")
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy")
        ax.legend(title="")

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "figure3_accuracy_summary.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def setup_plotting():
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["font.size"] = 12
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


def generate_figures(results, fig_dir: Path):
    setup_plotting()
    df = results_to_dataframe(results)
    if df.empty:
        print("No results found. Skipping figure generation.")
        return
    plot_performance_bars(df, fig_dir)
    plot_convergence_curves(results, fig_dir)
    plot_accuracy_summary(df, fig_dir)


def generate_tex():
    """
    No-op placeholder: LaTeX files are static.
    """
    print("LaTeX files are static. Update tex/ as needed.")


def main():
    parser = argparse.ArgumentParser(description="Generate figures and manuscript")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--compile-pdf", action="store_true")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--fig-dir", type=str, default=str(FIG_DIR))
    parser.add_argument("--summary-md", type=str, default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(exist_ok=True)

    results = load_results(data_dir)
    generate_figures(results, fig_dir)

    if not args.figures_only:
        generate_tex()

    if args.compile_pdf:
        # Delegate to shell script
        import subprocess
        subprocess.run(["bash", "tex/compile.sh"], check=False)

    if args.summary_md:
        df = results_to_dataframe(results)
        if not df.empty:
            summary = df.groupby(["arch", "bio"]).agg(
                loss=("final_val_loss", "mean"),
                acc=("final_val_acc", "mean"),
            )
            lines = [
                "# Cycle Summary",
                "",
                f"- Runs: {len(df)}",
                "",
                "## Mean Final Metrics",
                "",
            ]
            for (arch, bio), row in summary.iterrows():
                lines.append(
                    f"- {arch} | {bio}: loss={row['loss']:.4f}, acc={row['acc']:.4f}"
                )
            Path(args.summary_md).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
