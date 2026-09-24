# Parallel Experiment Configuration

Optimized for M5 Studio (16 cores, 128GB RAM) to complete all experiments in 1-2 hours.

## Quick Start

```bash
cd paper
make experiments-parallel
```

This runs all 18 experiments simultaneously:
- 3 architectures (RNN, LSTM, Transformer)
- 2 variants (base, bio)
- 3 seeds (42, 123, 456)

## Configuration: "parallel" Mode

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 30 | Enough to show convergence differences |
| **Sequences** | 2,000 | 5x more than cycle, enough data |
| **Batch size** | 32 | Balanced memory/speed |
| **Model dim** | 256 | Half of "full", faster training |
| **Sequence length** | 128 | Standard copy task length |

## Expected Runtime

**Per experiment:** 5-8 minutes (sequential)
**All 18 in parallel:** ~6-10 minutes on 16 cores

With your 16-core system:
- Each experiment gets ~1 core
- All complete simultaneously
- **Total time: ~10-15 minutes**

## Memory Usage

- RNN: ~100-200 MB per run
- LSTM: ~200-400 MB per run
- Transformer: ~400-800 MB per run
- **Total for 18 parallel: ~6-10 GB** (well within 128GB)

## Comparison to Other Modes

| Mode | Epochs | Sequences | Model Dim | Time (Sequential) | Purpose |
|------|--------|-----------|-----------|-------------------|---------|
| `cycle` | 2 | 200 | 128 | ~30 sec/run | Quick testing |
| `parallel` | 30 | 2,000 | 256 | ~6 min/run | **Fast iteration** |
| `fast` | 20 | 1,000 | 256 | ~4 min/run | Development |
| `full` | 100 | 10,000 | 512 | ~2+ hours/run | Final results |

## Usage

### Run all experiments in parallel
```bash
make experiments-parallel
```

### Run with custom workers
```bash
cd paper && source venv/bin/activate
python experiments/run_parallel.py --mode parallel --workers 12
```

### Monitor progress
```bash
# Watch CPU usage
htop

# Check output directory
watch -n 5 'ls -lh data/parallel*.pkl | wc -l'
```

## After Completion

Generate figures with the parallel results:

```bash
cd paper && source venv/bin/activate
python experiments/generate_figures.py --cycle parallel_results
make paper
```

## What the Data Shows

The "parallel" mode is designed to show:
1. **Convergence speed differences** (primary metric)
2. **Final accuracy** (all models should reach >95% on copy task)
3. **Training stability** (variance across seeds)

With 30 epochs and 2,000 sequences:
- Bio-RNN should reach low loss in ~5-10 epochs
- Base-RNN should reach low loss in ~20-25 epochs
- This ratio demonstrates the bio-enhancement effect

## Troubleshooting

### "RuntimeError: CUDA out of memory"
You don't have CUDA, but if you did:
```python
DEVICE = torch.device('cpu')  # Force CPU
```

### "Too many open files"
```bash
ulimit -n 4096
```

### Experiments too slow
Reduce to 2 seeds instead of 3:
```bash
python experiments/run_parallel.py --seeds 42 123
```

## Design Philosophy

The key insight: We care about **relative differences** between base and bio variants, not absolute performance numbers.

30 epochs with 2,000 sequences gives us:
- ✅ Clear convergence curves
- ✅ Stable final metrics
- ✅ Multiple seeds for statistics
- ✅ Fast enough to iterate (< 15 min)
- ❌ Not state-of-the-art absolute performance (don't need it!)

This is perfect for a research paper comparing architectures.
