from toy_transformer import *
import time 
import numpy as np
import platform
from collections import deque
import pynvml 

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
micro_batch_size = 2
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

