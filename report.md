# Full analysis report

Results were measured on two NVIDIA GeForce RTX 2080 GPUs with a baseline model configuration with vocab_size = 1000, seq_len = 256, d_model = 512, n_heads = 8, n_layers = 12, ffn_mult = 4, dropout = 0.1, batch_size = 32.

### Batch size VS Throughput 

<img src="graphs/batch_size_throughput.png" alt="Batch size vs Throughput" width="600"/>

### Sequence length VS Throughput

<img src="graphs/seq_throughput.png" alt="Sequence length vs Throughput" width="600"/>

### Number of layers VS Throughput

<img src="graphs/n_layers_throughput.png" alt="Number of layers vs Throughput" width="600"/>