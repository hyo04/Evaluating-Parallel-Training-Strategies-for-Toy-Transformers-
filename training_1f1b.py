import torch
import random
import time 
import numpy as np
import platform
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import pynvml 

micro_batch_size = 16           #baseline 2 

#--------------------------------------MODEL-----------------------------------------
# --- Config ---
class TransformerConfig:
    def __init__(self,
                 vocab_size=1000,       #baseline 1000
                 seq_len=256,           #baseline 256
                 d_model=512,           #baseline 512
                 n_heads=8,             #baseline 8
                 n_layers=12,           #baseline 12
                 ffn_mult=4,            #baseline 4
                 dropout=0.1,           #baseline 0.1
                 batch_size=32):        #baseline 32
      
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


#----------------------------------PIPELINE--------------------------------------------

class PipelineWrapper(nn.Module):
    def __init__(self, model: MinimalTransformer, config: TransformerConfig):
        super().__init__()
        
        # More balanced distribution across 4 stages
        layers_per_stage = max(1, config.n_layers // 3)

        # Stage 1: Embedding + some early layers (GPU 0)
        self.stage1 = nn.Sequential(
            model.encoder.embed,
            *model.encoder.layers[:layers_per_stage]
        )

        # Stage 2: Middle layers (GPU 0) 
        self.stage2 = nn.Sequential(*model.encoder.layers[layers_per_stage:2*layers_per_stage])

        # Stage 3: Remaining layers (GPU 1)
        self.stage3 = nn.Sequential(*model.encoder.layers[2*layers_per_stage:])

        # Stage 4: Norm + output head (GPU 1)
        self.norm = model.encoder.norm
        self.stage4 = model.head
        self.config = config


    # ---------------- Training forward  ----------------
    
    #each stage is called independently 
    def stage1_forward(self, x):
        return self.stage1(x)
    
    def stage2_forward(self, x):
        return self.stage2(x)
    
    def stage3_forward(self, x):
        return self.stage3(x)

    def stage4_forward(self, x):
        x = self.norm(x)
        logits = self.stage4(x)
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

def run_sequential_baseline():
    
    # Move entire model to GPU 0
    sequential_model = MinimalTransformer(config).to(devices[0])
    optimizer_seq = torch.optim.Adam(sequential_model.parameters(), lr=1e-4)

    # Metrics storage
    step_logs = []

    pynvml.nvmlInit()
    num_gpus = torch.cuda.device_count()
    gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]

    baseline_start = time.time()
    
    for step in range(num_steps):
        x, y = random_batch(config, devices[0])

        step_start = torch.cuda.Event(enable_timing=True)
        step_end   = torch.cuda.Event(enable_timing=True)
        step_start.record()

        # Forward
        logits = sequential_model(x)

        # Loss
        loss = criterion(logits.reshape(-1, config.vocab_size), y.reshape(-1))

        # Backward
        optimizer_seq.zero_grad()
        loss.backward()
        optimizer_seq.step()

        step_end.record()
        torch.cuda.synchronize()  # wait for GPU kernels

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

        '''
        print(f"Step {step}: {step_time_ms:.2f} ms, "
              f"{throughput:.1f} tokens/sec, "
              f"GPU utilization: {utilization}, "
              f"Memory MB: {mem_usage}")
        '''

        
    baseline_end = time.time()
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
    print(f"Total time: {(baseline_end - baseline_start)*1000:.2f} ms")

    return step_logs
    

# --- Config & Model ---
config = TransformerConfig()
model = MinimalTransformer(config)
num_steps = 5

#------------------Initialize for GPU usage tracking ----------------
pynvml.nvmlInit()
num_gpus = torch.cuda.device_count()
gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]



# --- Synthetic Dataset Generator ---
def random_batch(config, device):

    # full sequence including the "next-token" target
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len + 1), device=device)
    # inputs: everything except last token
    x = tokens[:, :-1]
    # targets: everything except first token (shifted left by one)
    y = tokens[:, 1:]

    return x, y

devices = [torch.device("cuda:0"), torch.device("cuda:1")]
criterion = nn.CrossEntropyLoss()


# --- Baseline Training Loop ----

#communication stream 
comm_stream0 = torch.cuda.Stream(devices[0])
comm_stream1 = torch.cuda.Stream(devices[1])

#assignment of stages to each device 
pipeline = PipelineWrapper(model, config)
pipeline.stage1.to(devices[0])
pipeline.stage2.to(devices[0])
pipeline.stage3.to(devices[1])
pipeline.norm.to(devices[1])
pipeline.stage4.to(devices[1])


# Loss + optimizer
optimizer = torch.optim.Adam(
    list(pipeline.stage1.parameters()) + 
    list(pipeline.stage2.parameters()) +
    list(pipeline.stage3.parameters()) + 
    list(pipeline.stage4.parameters()), 
    lr=1e-4 
)

batch_size = 32
token_ids = torch.randint(0, config.vocab_size, (batch_size, config.seq_len)).to(devices[0])

#micro-batching

num_micro_batches = batch_size // micro_batch_size

#add micro-batch buffers to store the activations before being passed to the next stage 
stage2_buffers = [None] * num_micro_batches
stage3_buffers = [None] * num_micro_batches

fwd_start_events = [[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(num_micro_batches)]
fwd_end_events   = [[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(num_micro_batches)]

bwd_start_events = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(num_micro_batches)]
bwd_end_events   = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(num_micro_batches)]

comm_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_micro_batches)]
comm_end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(num_micro_batches)]


#activations for the next stage 
stage1_buffers = [None] * num_micro_batches
stage2_original_buffers = [None] * num_micro_batches    #on GPU0
stage2_buffers = [None] * num_micro_batches             #on GPU1 (communicated via comm_stream0)
stage3_buffers = [None] * num_micro_batches
stage4_buffers = [None] * num_micro_batches

gpu1_fwd_queue = deque()
gpu1_bwd_queue = deque()
gpu0_bwd_queue = deque()


grad_act2_buffers = [None] *num_micro_batches
grad_act3_buffers = [None] *num_micro_batches

total_start = time.time()

step_logs = []

num_steps = 5
for step in range(1, num_steps+1):

    # Reset state variables (flags) for this step
    gpu0_fwd_idx = 0
    all_backward_done = False
    gpu0_fwd_done = [False] * num_micro_batches
    gpu1_bwd_done = [False] * num_micro_batches
    gpu0_bwd_done = [False] * num_micro_batches
    fwd_done = [False] * num_micro_batches
    gpu1_next_fwd_enqueue = 0
    gpu1_next_bwd_enqueue = 0
    gpu0_next_bwd_enqueue = 0
    
    # Clear queues for this step
    gpu1_fwd_queue.clear()
    gpu1_bwd_queue.clear()
    gpu0_bwd_queue.clear()

    mem_usage = {}
    optimizer.zero_grad()

    #work on a new data set each step 
    x, y = random_batch(config, devices[0])

    #batch split along the batch dimension 
    micro_x = x.split(micro_batch_size, dim=0)
    micro_y = y.split(micro_batch_size, dim=0)

    # --- ADD: Step wall-clock events ---
    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)
    step_start.record()

    ## Controller loop (+processing multiple micro-batches per iteration)
    while not all_backward_done:
        
        work_done_this_iteration = False
        
        #----------------------------------------------- FORWARD --------------------------------------------------
        
        # GPU0 FWD: Process ALL ready micro-batches, not just one (change from for -> while)
        while gpu0_fwd_idx < num_micro_batches and not gpu0_fwd_done[gpu0_fwd_idx]:
            # Stage 1
            fwd_start_events[gpu0_fwd_idx][0].record()
            act1 = pipeline.stage1_forward(micro_x[gpu0_fwd_idx])
            act1.retain_grad() 
            fwd_end_events[gpu0_fwd_idx][0].record()
            stage1_buffers[gpu0_fwd_idx] = act1

            # Stage 2  
            fwd_start_events[gpu0_fwd_idx][1].record()
            act2 = pipeline.stage2_forward(act1)
            act2.retain_grad() 
            fwd_end_events[gpu0_fwd_idx][1].record()
            
            stage2_original_buffers[gpu0_fwd_idx] = act2  

            # Async communication
            comm_start_events[gpu0_fwd_idx].record()
            with torch.cuda.stream(comm_stream0):
                stage2_buffers[gpu0_fwd_idx] = act2.to(devices[1], non_blocking=True)
            comm_end_events[gpu0_fwd_idx].record()

            gpu0_fwd_done[gpu0_fwd_idx] = True
            work_done_this_iteration = True
            gpu0_fwd_idx += 1

        # GPU1 FWD: Process ALL ready micro-batches from queue
        # Update queue with ALL ready items
        while gpu1_next_fwd_enqueue < gpu0_fwd_idx:
            i = gpu1_next_fwd_enqueue
            if gpu0_fwd_done[i] and stage2_buffers[i] is not None and not fwd_done[i]:
                gpu1_fwd_queue.append(i)
            gpu1_next_fwd_enqueue += 1

        # Process entire GPU1 forward queue
        while gpu1_fwd_queue: 
            gpu1_fwd_idx = gpu1_fwd_queue.popleft()
            torch.cuda.current_stream(devices[1]).wait_stream(comm_stream0)

            fwd_start_events[gpu1_fwd_idx][2].record()
            act3 = pipeline.stage3_forward(stage2_buffers[gpu1_fwd_idx])
            fwd_end_events[gpu1_fwd_idx][2].record()
            act3.retain_grad()
            stage3_buffers[gpu1_fwd_idx] = act3

            fwd_start_events[gpu1_fwd_idx][3].record()
            logits = pipeline.stage4_forward(act3)
            fwd_end_events[gpu1_fwd_idx][3].record()

            stage4_buffers[gpu1_fwd_idx] = logits
            fwd_done[gpu1_fwd_idx] = True
            work_done_this_iteration = True

        #----------------------------------------------- BACKWARD --------------------------------------------------

        # GPU1 BWD: Update queue and process ALL ready backward passes
        while gpu1_next_bwd_enqueue < num_micro_batches:
            j = gpu1_next_bwd_enqueue
            if fwd_done[j] and stage4_buffers[j] is not None and not gpu1_bwd_done[j]:
                gpu1_bwd_queue.append(j) 
            gpu1_next_bwd_enqueue += 1

        # Process entire GPU1 backward queue
        while gpu1_bwd_queue:
            gpu1_bwd_idx = gpu1_bwd_queue.popleft()
            torch.cuda.current_stream(devices[1]).wait_stream(comm_stream0)

            logits = stage4_buffers[gpu1_bwd_idx]
            y_mb_device = micro_y[gpu1_bwd_idx].to(devices[1])
            loss = criterion(logits.reshape(-1, config.vocab_size), y_mb_device.reshape(-1))
            
            bwd_start_events[gpu1_bwd_idx][0].record()
            loss.backward()
            bwd_end_events[gpu1_bwd_idx][0].record()

            grad_act3 = stage3_buffers[gpu1_bwd_idx].grad
            with torch.cuda.stream(comm_stream1):
                grad_act3_buffers[gpu1_bwd_idx] = grad_act3.to(devices[0], non_blocking=True)

            stage3_buffers[gpu1_bwd_idx].grad = None
            gpu1_bwd_done[gpu1_bwd_idx] = True
            work_done_this_iteration = True

        # GPU0 BWD: Update queue and process ALL ready backward passes  
        while gpu0_next_bwd_enqueue < num_micro_batches:
            k = gpu0_next_bwd_enqueue
            if gpu1_bwd_done[k] and stage2_buffers[k] is not None and not gpu0_bwd_done[k]:
                gpu0_bwd_queue.append(k)
            gpu0_next_bwd_enqueue += 1

        # Process entire GPU0 backward queue
        while gpu0_bwd_queue:
            gpu0_bwd_idx = gpu0_bwd_queue.popleft()
            torch.cuda.current_stream(devices[0]).wait_stream(comm_stream1)

            grad_act2 = grad_act3_buffers[gpu0_bwd_idx]
            bwd_start_events[gpu0_bwd_idx][1].record()
            if stage2_original_buffers[gpu0_bwd_idx].grad is None:
                stage2_original_buffers[gpu0_bwd_idx].backward(grad_act2)
            else:
                # Accumulate gradients if already computed
                stage2_original_buffers[gpu0_bwd_idx].grad += grad_act2
            bwd_end_events[gpu0_bwd_idx][1].record()

            gpu0_bwd_done[gpu0_bwd_idx] = True
            work_done_this_iteration = True
        
        all_backward_done = all(gpu0_bwd_done)
        
        # Prevent infinite loop if no work was done
        if not work_done_this_iteration and not all_backward_done:
            time.sleep(0.001)  # Small sleep to prevent busy waiting
            

    optimizer.step()

    # ---- Sync + measure times per step ----
    torch.cuda.synchronize(devices[0]); torch.cuda.synchronize(devices[1])

    step_end.record()
    torch.cuda.synchronize()  # wait for event to complete
    step_time = step_start.elapsed_time(step_end)   #in ms 
    

    # --- calculate throughput ---
    total_tokens = config.batch_size * config.seq_len
    step_time_ms = step_start.elapsed_time(step_end)  # float in ms
    step_time_sec = step_time_ms / 1000.0             # convert to seconds
    throughput = total_tokens / step_time_sec  
    throughput_per_gpu = throughput / num_gpus

    
    # --- GPU utilization and memory ---
    utilization, mem_usage = log_gpu_stats()
    
    # --- store metrics ---
    step_logs.append({
        "step": step,
        "throughput": throughput,
        "throughput_per_gpu": throughput_per_gpu,
        "gpu_utilization": utilization,
        "gpu_memory_MB": mem_usage,
        "step_time_sec": step_time_ms * 1000.0
    })



avg_throughput = sum([log['throughput'] for log in step_logs]) / len(step_logs)
avg_throughput_per_gpu = sum([log['throughput_per_gpu'] for log in step_logs]) / len(step_logs)
avg_util = [sum([log['gpu_utilization'][i] for log in step_logs])/len(step_logs) 
            for i in range(num_gpus)]
avg_mem = [max([log['gpu_memory_MB'][i] for log in step_logs]) 
           for i in range(num_gpus)]


print(f"Avg throughput: {avg_throughput:.1f} tokens/sec")
print(f"Avg throughput per GPU: {avg_throughput_per_gpu:.1f} tokens/sec")
print(f"Avg GPU utilization per GPU: {avg_util}")
print(f"Max GPU memory per GPU (MB): {avg_mem}")


#run_sequential_baseline()







