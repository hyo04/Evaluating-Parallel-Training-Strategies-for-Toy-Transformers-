import torch
import random
import time 
import numpy as np
import platform
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import pynvml 


#--------------------------------------MODEL-----------------------------------------

# --- Config ---
class TransformerConfig:
    def __init__(self,
                 vocab_size=1000,       #1000
                 seq_len=256,           #256
                 d_model=512,           #512
                 n_heads=4,             #8
                 n_layers=35,           #12
                 ffn_mult=4,            #4
                 dropout=0.1,           #0.1
                 batch_size=32):        #32
      
      self.vocab_size = vocab_size
      self.seq_len = seq_len
      self.d_model = d_model
      self.n_heads = n_heads
      self.n_layers = n_layers
      self.ffn_mult = ffn_mult
      self.dropout = dropout
      self.batch_size = batch_size

# --- Embeddings ---
class TokenPositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        #embedding layer for the tokenID
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        #embedding layer for the position 
        self.pos_embedding = nn.Embedding(config.seq_len, config.d_model)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        #device=token_ids.device makes sure that the position matrix is created on the same device as the token_ids 
        #unsqueeze(0) adds a new dimension to match the token embedding dimension 
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        token_embeds = self.token_embedding(token_ids)
        pos_embeds = self.pos_embedding(positions)
        return token_embeds + pos_embeds
        

# --- Multi-Head Attention ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
         #split the vector into differerent heads -> size per head is d_model/heads 
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads

        #linear layers to produce q,k,v
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)

        #linear layer for the output after concatenating heads 
        self.out_linear = nn.Linear(self.d_model, self.d_model)


    def forward(self, x, mask=None):
        '''
        x: (batch_size, seq_len, d_model)
        mask: optional mask to prevent attending to certain tokens
        '''
        #compute Q,K,V weights 
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        batch_size, seq_len, d_model = x.shape
        head_dim = self.head_dim
        n_heads = self.n_heads

        #split into multiple heads -> reshape 
        x = x.view(batch_size, seq_len, n_heads, head_dim)
        x = x.permute(0,2,1,3)
        q = q.view(batch_size, seq_len, n_heads, head_dim)
        q = q.permute(0,2,1,3)
        k = k.view(batch_size, seq_len, n_heads, head_dim)
        k = k.permute(0,2,1,3)
        v = v.view(batch_size, seq_len, n_heads, head_dim)
        v = v.permute(0,2,1,3)

        #each scores[b,h,i,j] = how much token i attends to token j in head h of batch b 
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        #softmax along the last dimension (sum to 1 for attention weights)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        #return the shape to the original
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        #final output
        output = self.out_linear(context)

        return output 


# --- Feed-Forward Network (FFN) ---
class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_model * config.ffn_mult

        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = MultiHeadSelfAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        return out 


# --- Transformer Encoder Stack ---
class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed = TokenPositionalEmbedding(config)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, mask=None):
        # 1. Embed tokens (add position info)
        x = self.embed(x)
        
        # 2. Pass through each TransformerBlock
        for layer in self.layers:
            x = layer(x)
        
        # 3. Apply final normalization
        x = self.norm(x)

        return x

# --- Output Head ---
# final linear projection from d_model to vocab_size 
class OutputHead(nn.Module):
    def __init__(self, config: TransformerConfig, embedding_layer):
        super().__init__()
        self.proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x):
        # Input: (batch, seq_len, d_model)
        # Output: (batch, seq_len, vocab_size)
        logits = self.proj(x)
        return logits

# --- Full Model ---
class MinimalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.head = OutputHead(config, self.encoder.embed)


    def forward(self, token_ids, mask=None):
        hidden = self.encoder(token_ids)       # (batch, seq_len, d_model)
        logits = self.head(hidden)     # (batch, seq_len, vocab_size)
        return logits

#function for logging the statistics 
def log_gpu_stats():
    utilization = []
    memory_used = []
    for handle in gpu_handles:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization.append(util.gpu)      # % GPU utilization
        memory_used.append(mem.used / 1024**2)  # MB
    return utilization, memory_used

#---------------------------------BASELINE---------------------------------------------

# --- Config & Model ---
config = TransformerConfig()
model = MinimalTransformer(config) 

#need the parameters to be on device0 before it can be copied to the other devices using DataParallel
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0")
else:
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# --- Synthetic Dataset Generator ---
def random_batch(config, device):

    # full sequence including the "next-token" target
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len + 1), device=device)
    # inputs: everything except last token
    x = tokens[:, :-1]
    # targets: everything except first token (shifted left by one)
    y = tokens[:, 1:]

    return x, y

# --- Baseline Training Loop ----

# Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


step_logs = []

pynvml.nvmlInit()
num_gpus = torch.cuda.device_count()
gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]

total_start = time.time()

num_steps = 5
for step in range(1, num_steps+1):
    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)
    
    x, y = random_batch(config, device)
    
    step_start.record()  # Start timing

    mem_usage = {}
    
    # forward pass
    start_fwd = time.time()
    logits = model(x)
    #loss tensor should be on the same device as the final logits (device 1)
    y = y.reshape(-1).to(device)
    loss = criterion(logits.reshape(-1, config.vocab_size), y)

    torch.cuda.synchronize()
    fwd_time = time.time() - start_fwd

    # backward pass
    torch.cuda.synchronize()
    start_bwd = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    step_end.record()  # End timing
    torch.cuda.synchronize()
    
    # Step time & throughput
    step_time_ms = step_start.elapsed_time(step_end)  # ms
    step_time_sec = step_time_ms / 1000.0
    total_tokens = config.batch_size * config.seq_len
    throughput = total_tokens / step_time_sec
    throughput_per_gpu = throughput / num_gpus


    # GPU utilization & memory
    utilization, mem_usage = log_gpu_stats()

    # Store metrics
    step_logs.append({
        "step": step,
        "throughput": throughput,
        "throughput_per_gpu": throughput_per_gpu,
        "gpu_utilization": utilization,
        "gpu_memory_MB": mem_usage,
        "step_time_sec": step_time_sec
    })

total_end = time.time()


# --- Aggregate ---
avg_throughput = sum([log['throughput'] for log in step_logs]) / len(step_logs)
avg_throughput_per_gpu = sum([log['throughput_per_gpu'] for log in step_logs]) / len(step_logs)
avg_util = [sum([log['gpu_utilization'][i] for log in step_logs])/len(step_logs) 
            for i in range(num_gpus)]
max_mem = [max([log['gpu_memory_MB'][i] for log in step_logs])
            for i in range(num_gpus)]

print("\n=== SEQUENTIAL BASELINE SUMMARY ===")
print(f"Avg throughput: {avg_throughput:.1f} tokens/sec")
print(f"Avg throughput per GPU: {avg_throughput_per_gpu:.1f} tokens/sec")
print(f"Avg GPU utilization per GPU: {avg_util}")
print(f"Max GPU memory per GPU (MB): {max_mem}")
print(f"Total time: {(total_end - total_start)*1000:.2f} ms")



    




