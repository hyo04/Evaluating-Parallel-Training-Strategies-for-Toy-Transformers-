# Parallel-Training-Strategies-for-Toy-Transformers-Sequential-Data-and-Pipeline
PyTorch implementation of a toy Transformer model with sequential, data-parallel, and pipeline-parallel (1F1B) training. Includes experiments analyzing memory consumption and throughput trade-offs across different hyperparameters. These experiments were designed not only to measure raw performance but also to analyze trade-offs between different parallel training strategies, demonstrating the ability to evaluate system-level efficiency in large-scale model training.

This project explores training efficiency under different parallelism strategies:
- Sequential execution
- Data parallelism
- Pipeline parallelism (1F1B scheduling)

## Key Results
- **Data parallelism** gives the highest throughput across most settings.
- **Pipeline parallelism** shows improved scaling with sequence length, but lower overall throughput.
- **Sequential execution** is limited by single-GPU utilization.

📊 For detailed results & analysis, see [Full Report](analysis/report.md).



