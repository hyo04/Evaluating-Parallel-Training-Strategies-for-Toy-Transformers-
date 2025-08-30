from toy_transformer import *
import pynvml 
import time


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



    




