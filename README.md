# Parallel-Training-Strategies-for-Toy-Transformers-Sequential-Data-and-Pipeline
PyTorch implementation of a toy Transformer model with sequential, data-parallel, and pipeline-parallel (1F1B) training. Includes experiments analyzing memory consumption and throughput trade-offs across different hyperparameters. These experiments were designed not only to measure raw performance but also to analyze trade-offs between different parallel training strategies, demonstrating the ability to evaluate system-level efficiency in large-scale model training.

This project explores training efficiency under different parallelism strategies:
- Sequential execution
- Data parallelism
- Pipeline parallelism (1F1B scheduling)


## Key Results
- **Data parallelism** gives the highest throughput across most settings, but has low GPU utilization.
- **Pipeline parallelism** shows improved scaling with sequence length, but lower overall throughput.
- **Sequential execution** experiences a moderately high and consistent throughput and GPU utilization.  

📊 For detailed results & analysis, see [Full Report](analysis/report.md).


## 🔎 Limitations & Future Work
- Experiments were conducted on only 2 NVIDIA RTX 2080 GPUs, which limits scalability insights for larger clusters.  
- Only three strategies were tested (sequential, data parallelism, pipeline parallelism); tensor parallelism and hybrid strategies were not explored.  
- The Transformer model was intentionally small (d_model=512, n_layers=12) for tractability. Larger models may exhibit different scaling behavior.  

**Future directions** include extending experiments to 8+ GPUs, testing hybrid parallelization strategies, profiling communication overhead with tools such as Nsight Systems, and scaling to larger model sizes.





