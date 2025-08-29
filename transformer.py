import torch
import random
import time 
import numpy as np
import platform
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


#--------------------------------------MODEL-----------------------------------------

# --- Config ---
class TransformerConfig:
    def __init__(self,
                 vocab_size=1000,
                 seq_len=256,
                 d_model=512,
                 n_heads=4,
                 n_layers=35,
                 ffn_mult=4,
                 dropout=0.1,
                 batch_size=32):
      
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
    

#---------------------------------BASELINE---------------------------------------------

def run_sequential_baseline():
    """Run the same model sequentially on GPU 0 for comparison"""
    #print("Running sequential baseline...")
    
    # Move entire model to GPU 0
    sequential_model = MinimalTransformer(config).to(devices[0])
    optimizer_seq = torch.optim.Adam(sequential_model.parameters(), lr=1e-4)

    baseline_start = time.time()
    baseline_results = []
    
    for step in range(num_steps):
        x, y = random_batch(config, devices[0])
        step_start = time.time()

        torch.cuda.synchronize()
        fwd_start = time.time()
        logits = sequential_model(x)
        torch.cuda.synchronize()
        fwd_time = time.time() - fwd_start
        
        loss = criterion(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        
        torch.cuda.synchronize()
        bwd_start = time.time()
        optimizer_seq.zero_grad()
        loss.backward()
        optimizer_seq.step()
        torch.cuda.synchronize()
        bwd_time = time.time()-bwd_start
        
        step_end = time.time()
        step_time = (step_end - step_start) * 1000
        #print(f"Sequential Step {step+1}: {step_time:.2f}ms")

        # throughput
        total_tokens = config.batch_size * config.seq_len
        throughput = total_tokens / (step_time / 1000.0)  # tokens/sec

        print(f"\n=== SEQUENTIAL STEP {step+1} ===")
        print(f"  Forward    = {fwd_time:.2f} ms")
        print(f"  Backward   = {bwd_time:.2f} ms")
        print(f"  Step time  = {step_time:.2f} ms")
        print(f"  Throughput = {throughput:.1f} tokens/sec")
        print(f"  Memory     = {mem_usage}")
        
    
    baseline_end = time.time()
    baseline_total = (baseline_end - baseline_start) * 1000

    #per step 
    
    print(f"\n=== SEQUENTIAL BASELINE SUMMARY ===")
    print(f"Total time: {baseline_total:.2f}ms")
    print(f"Average step time: {baseline_total/num_steps:.2f}ms")
    
    seq_throughputs = []  # This should be initialized at the beginning of the function
    # ... (after throughput calculation in the loop, add:)
    # seq_throughputs.append(throughput)

    avg_seq_throughput = sum(seq_throughputs) / len(seq_throughputs) if seq_throughputs else 0
    print(f"Average sequential throughput: {avg_seq_throughput:.1f} tokens/sec")

    return baseline_total, baseline_total/num_steps, avg_seq_throughput
    

# --- Config & Model ---
config = TransformerConfig()
model = MinimalTransformer(config)

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

devices = [torch.device("cuda:0"), torch.device("cuda:1")]

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
criterion = nn.CrossEntropyLoss()
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
micro_batch_size = 2
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

#Bookkeeping / flags
gpu0_fwd_done = [None] * num_micro_batches
gpu1_bwd_done = [None] * num_micro_batches
gpu0_bwd_done = [None] * num_micro_batches
fwd_done = [None] * num_micro_batches
bwd_done = [None] * num_micro_batches

#activations for the next stage 
stage1_buffers = [None] * num_micro_batches
stage2_original_buffers = [None] * num_micro_batches    #on GPU0
stage2_buffers = [None] * num_micro_batches             #on GPU1 (communicated via comm_stream0)
stage3_buffers = [None] * num_micro_batches
stage4_buffers = [None] * num_micro_batches

gpu0_fwd_idx = 0
bwd_idx = 0
all_backward_done = False

gpu1_fwd_queue = deque()
gpu1_bwd_queue = deque()
gpu0_bwd_queue = deque()

gpu1_next_fwd_enqueue = 0
gpu1_next_bwd_enqueue = 0
gpu0_next_bwd_enqueue = 0

grad_act2_buffers = [None] *num_micro_batches
grad_act3_buffers = [None] *num_micro_batches

total_start = time.time()

throughputs = []

num_steps = 5
for step in range(1, num_steps+1):
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

    # --- ADD: Record step end after sync ---
    step_end.record()
    torch.cuda.synchronize()  # wait for event to complete
    step_time = step_start.elapsed_time(step_end)  # true wall-clock ms

    # average fwd/bwd times across microbatches for this step
    fwd_times = []
    bwd_times = []
    for mb_idx in range(num_micro_batches):
        fwd_times.extend([e1.elapsed_time(e2) for e1, e2 in zip(fwd_start_events[mb_idx], fwd_end_events[mb_idx])])
        bwd_times.extend([e1.elapsed_time(e2) for e1, e2 in zip(bwd_start_events[mb_idx], bwd_end_events[mb_idx])])

    avg_fwd_time = sum(fwd_times) / len(fwd_times) if fwd_times else 0
    avg_bwd_time = sum(bwd_times) / len(bwd_times) if bwd_times else 0

    # memory usage
    for i in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.memory_allocated(i) / 1e6
        mem_max = torch.cuda.max_memory_allocated(i) / 1e6
        mem_usage[f"gpu{i}"] = {"alloc_MB": mem_alloc, "max_MB": mem_max}

    # throughput
    total_tokens = config.batch_size * config.seq_len
    throughput = total_tokens / (step_time / 1000.0)  # tokens/sec

    throughputs.append(throughput)

    print(f"\n=== PIPELINE STEP {step} ===")
    print(f"  Avg Fwd (per microbatch) = {avg_fwd_time:.2f} ms")
    print(f"  Avg Bwd (per microbatch) = {avg_bwd_time:.2f} ms")
    print(f"  Step wall-clock          = {step_time:.2f} ms")
    print(f"  Throughput = {throughput:.1f} tokens/sec")
    print(f"  Memory     = {mem_usage}")

total_end = time.time()
total_time = (total_end - total_start) * 1000
avg_step_time = total_time / num_steps

'''
print(f"\n=== PIPELINE PERFORMANCE SUMMARY ===")
print(f"Total time: {total_time:.2f} ms")
print(f"Average step time: {avg_step_time:.2f} ms")

baseline_total, baseline_avg = run_sequential_baseline()
print(f"\n=== SPEEDUP ANALYSIS ===")
print(f"Pipeline vs Sequential speedup: {baseline_avg/avg_step_time:.2f}x")
'''

avg_pipeline_throughput = sum(throughputs) / len(throughputs) if throughputs else 0

print(f"\n=== PIPELINE PERFORMANCE SUMMARY ===")
print(f"Total time: {total_time:.2f} ms")
print(f"Average step time: {avg_step_time:.2f} ms")
print(f"Average pipeline throughput: {avg_pipeline_throughput:.1f} tokens/sec")

baseline_total, baseline_avg, avg_seq_throughput = run_sequential_baseline()
print(f"\n=== SPEEDUP ANALYSIS ===")
print(f"Pipeline vs Sequential speedup: {baseline_avg/avg_step_time:.2f}x")
print(f"Throughput improvement: {avg_pipeline_throughput/avg_seq_throughput:.2f}x")




