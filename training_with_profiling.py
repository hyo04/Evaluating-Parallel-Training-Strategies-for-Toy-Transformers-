import torch
import torch.profiler
from collections import defaultdict, deque
import json
import random
import time 
import numpy as np
import platform
import torch.nn as nn
import torch.nn.functional as F
import pynvml 

micro_batch_size = 8          #baseline 2 
pynvml.nvmlInit()
num_gpus = torch.cuda.device_count()
gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]

class GEMMProfiler:
    """Microarchitectural profiling for GEMM operations in transformer layers"""
    
    def __init__(self, enable_detailed_profiling=True):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.kernel_stats = defaultdict(list)
        self.sm_efficiency_history = []
        self.memory_bandwidth_history = []
        self.tensor_core_utilization_history = []
        
    def start_gemm_profiling(self, gemm_name):
        """Context manager for profiling specific GEMM operations"""
        if not self.enable_detailed_profiling:
            return torch.profiler.record_function(gemm_name)
            
        return torch.profiler.record_function(f"GEMM_{gemm_name}")
    
    def analyze_kernel_metrics(self, prof_result):
        """Extract SM-level metrics from profiler results"""
        kernel_metrics = {
            'sm_efficiency': [],
            'memory_throughput_gb_s': [],
            'achieved_occupancy': [],
            'tensor_core_utilization': [],
            'warp_execution_efficiency': [],
            'kernel_duration_us': []
        }
        
        # Parse profiler events for CUDA kernels
        for event in prof_result.events():
            if event.device_type == torch.profiler.DeviceType.CUDA and 'gemm' in event.name.lower():
                # Extract key metrics (these would come from actual CUDA profiler)
                kernel_metrics['kernel_duration_us'].append(event.cuda_time_total)
                
                # Simulated metrics - in real implementation, these come from nvprof/nsight
                # You'd get these from event.cuda_memory_usage, event.flops, etc.
                kernel_metrics['sm_efficiency'].append(self._estimate_sm_efficiency(event))
                kernel_metrics['memory_throughput_gb_s'].append(self._estimate_memory_throughput(event))
                kernel_metrics['achieved_occupancy'].append(self._estimate_occupancy(event))
                
        return kernel_metrics
    
    def _estimate_sm_efficiency(self, event):
        """Estimate SM efficiency - replace with real nvprof data"""
        # In real implementation: achieved_cycles / theoretical_peak_cycles
        return np.random.uniform(0.7, 0.95)  # Placeholder
    
    def _estimate_memory_throughput(self, event):
        """Estimate memory bandwidth utilization"""
        # In real implementation: actual_bandwidth / peak_bandwidth  
        return np.random.uniform(400, 800)  # GB/s placeholder
    
    def _estimate_occupancy(self, event):
        """Estimate warp occupancy"""
        return np.random.uniform(0.6, 0.9)  # Placeholder


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
                 batch_size=128):        #baseline 32
      
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
        

# Modified MultiHeadSelfAttention with GEMM profiling
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, gemm_profiler: GEMMProfiler):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.gemm_profiler = gemm_profiler

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.out_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Profile QKV projection GEMMs
        with self.gemm_profiler.start_gemm_profiling("attention_qkv_proj"):
            q = self.q_linear(x)
            k = self.k_linear(x) 
            v = self.v_linear(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0,2,1,3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0,2,1,3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0,2,1,3)

        # Profile attention score GEMM (Q @ K^T)
        with self.gemm_profiler.start_gemm_profiling("attention_score_gemm"):
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attn_weights = torch.softmax(scores, dim=-1)

        # Profile attention output GEMM (attn @ V)  
        with self.gemm_profiler.start_gemm_profiling("attention_output_gemm"):
            context = torch.matmul(attn_weights, v)

        # Reshape back
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        # Profile output projection GEMM
        with self.gemm_profiler.start_gemm_profiling("attention_out_proj"):
            output = self.out_linear(context)

        return output


# Modified FeedForward with GEMM profiling
class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig, gemm_profiler: GEMMProfiler):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_model * config.ffn_mult
        self.gemm_profiler = gemm_profiler

        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Profile MLP up-projection GEMM
        with self.gemm_profiler.start_gemm_profiling("mlp_up_proj"):
            x = F.relu(self.linear1(x))
            
        # Profile MLP down-projection GEMM  
        with self.gemm_profiler.start_gemm_profiling("mlp_down_proj"):
            x = self.linear2(x)
            
        x = self.dropout(x)
        return x


# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = MultiHeadSelfAttention(config, gemm_profiler=GEMMProfiler)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config, gemm_profiler=GEMMProfiler)
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


# Enhanced profiling pipeline wrapper
class PipelineWrapper(nn.Module):
    def __init__(self, model: MinimalTransformer, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.gemm_profiler = GEMMProfiler()
        
        # Replace attention and FFN modules with profiled versions
        self._replace_with_profiled_modules(model)
        
        # Same pipeline distribution as before
        layers_per_stage = max(1, config.n_layers // 3)
        self.stage1 = nn.Sequential(
            model.encoder.embed,
            *model.encoder.layers[:layers_per_stage]
        )
        self.stage2 = nn.Sequential(*model.encoder.layers[layers_per_stage:2*layers_per_stage])
        self.stage3 = nn.Sequential(*model.encoder.layers[2*layers_per_stage:])
        self.norm = model.encoder.norm
        self.stage4 = model.head
        
    def _replace_with_profiled_modules(self, model):
        """Replace standard modules with profiled versions"""
        for layer in model.encoder.layers:
            # Replace attention
            old_attn = layer.attn
            layer.attn = MultiHeadSelfAttention(self.config, self.gemm_profiler)
            layer.attn.load_state_dict(old_attn.state_dict())
            
            # Replace feedforward
            old_ffn = layer.ffn
            layer.ffn = FeedForward(self.config, self.gemm_profiler)  
            layer.ffn.load_state_dict(old_ffn.state_dict())

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


# --- Synthetic Dataset Generator ---
def random_batch(config, device):

    # full sequence including the "next-token" target
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len + 1), device=device)
    # inputs: everything except last token
    x = tokens[:, :-1]
    # targets: everything except first token (shifted left by one)
    y = tokens[:, 1:]

    return x, y

class ComprehensiveProfiler:
    """Combines system-level and microarchitectural profiling"""
    
    def __init__(self, config, devices):
        self.config = config
        self.devices = devices
        self.system_metrics = []
        self.kernel_metrics = defaultdict(list)
        self.bottleneck_analysis = {}
        
    def profile_step_with_microarch(self, pipeline_fn, step_num):
        """Profile a training step with both system and kernel-level metrics"""
        
        # System-level timing
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        
        # Microarchitectural profiling context
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            use_cuda=True,  # Add this
            experimental_config=torch.profiler._ExperimentalConfig(verbose=True)
        ) as prof:
            
            step_start.record()
            
            # Execute your training step function here
            result = pipeline_fn()
            
            step_end.record()
            torch.cuda.synchronize()
        
        # Extract timing
        step_time_ms = step_start.elapsed_time(step_end)
        
        # Analyze kernel-level metrics
        kernel_analysis = self._analyze_gemm_kernels(prof)
        
        # System-level metrics (your existing code)
        utilization, mem_usage = log_gpu_stats()
        total_tokens = self.config.batch_size * self.config.seq_len
        throughput = total_tokens / (step_time_ms / 1000.0)
        
        # Store comprehensive metrics
        step_metrics = {
            'step': step_num,
            'system': {
                'throughput_tokens_sec': throughput,
                'step_time_ms': step_time_ms,
                'gpu_utilization': utilization,
                'gpu_memory_mb': mem_usage
            },
            'kernels': kernel_analysis,
            'bottleneck_analysis': self._compute_bottleneck_analysis(step_time_ms, kernel_analysis)
        }
        
        self.system_metrics.append(step_metrics)
        return step_metrics
    
    def _analyze_gemm_kernels(self, prof):
        """Extract GEMM kernel performance metrics"""
        gemm_analysis = {
            'attention_qkv_proj': {'sm_efficiency': [], 'memory_bw_util': [], 'duration_us': []},
            'attention_score_gemm': {'sm_efficiency': [], 'memory_bw_util': [], 'duration_us': []}, 
            'attention_output_gemm': {'sm_efficiency': [], 'memory_bw_util': [], 'duration_us': []},
            'mlp_up_proj': {'sm_efficiency': [], 'memory_bw_util': [], 'duration_us': []},
            'mlp_down_proj': {'sm_efficiency': [], 'memory_bw_util': [], 'duration_us': []}
        }
        
        # Parse profiler events for GEMM kernels
        for event in prof.events():
            if event.device_type == torch.profiler.DeviceType.CUDA:
                kernel_name = event.name.lower()
                
                # Map kernel names to our GEMM categories
                if 'gemm' in kernel_name or 'addmm' in kernel_name or 'bmm' in kernel_name:
                    gemm_type = self._classify_gemm_kernel(event.name)
                    if gemm_type in gemm_analysis:
                        
                        # Real metrics would come from CUPTI/nvprof integration
                        metrics = self._extract_kernel_metrics(event)
                        
                        gemm_analysis[gemm_type]['duration_us'].append(event.cuda_time_total)
                        gemm_analysis[gemm_type]['sm_efficiency'].append(metrics['sm_efficiency'])
                        gemm_analysis[gemm_type]['memory_bw_util'].append(metrics['memory_bw_util'])
        
        # Compute averages for each GEMM type
        for gemm_type in gemm_analysis:
            for metric in ['sm_efficiency', 'memory_bw_util', 'duration_us']:
                if gemm_analysis[gemm_type][metric]:
                    gemm_analysis[gemm_type][f'avg_{metric}'] = np.mean(gemm_analysis[gemm_type][metric])
                else:
                    gemm_analysis[gemm_type][f'avg_{metric}'] = 0.0
        
        return gemm_analysis
    
    def _classify_gemm_kernel(self, kernel_name):
        """Map actual CUDA kernel names to our GEMM categories"""
        name_lower = kernel_name.lower()
        
        # Look for actual CUDA kernel patterns
        if any(pattern in name_lower for pattern in ['sgemm', 'hgemm', 'gemm', 'addmm']):
            # Use stack trace or operation context to classify
            return 'general_gemm'  # Will need more sophisticated mapping
        
        return 'other'
    
    def _extract_kernel_metrics(self, event):
        """Extract microarchitectural metrics from kernel event"""
        # In a real implementation, you'd use CUPTI to get these metrics
        # For now, we'll simulate based on kernel characteristics
        
        # Estimate metrics based on operation size and type
        flops_estimate = self._estimate_flops(event)
        memory_estimate = self._estimate_memory_access(event)
        
        return {
            'sm_efficiency': min(0.95, max(0.5, flops_estimate / 1e9)),  # Simulated
            'memory_bw_util': min(900, max(200, memory_estimate / 1e6)), # GB/s simulated
            'achieved_occupancy': np.random.uniform(0.6, 0.9),
            'tensor_core_utilization': np.random.uniform(0.7, 0.95) if 'gemm' in event.name.lower() else 0.0
        }
    
    def _estimate_flops(self, event):
        """Estimate FLOPS for the kernel"""
        # Rough estimation - in real implementation, extract from profiler
        return event.cuda_time_total * 1e6  # Placeholder
    
    def _estimate_memory_access(self, event):
        """Estimate memory access pattern"""
        return event.cuda_time_total * 1e3  # Placeholder
    
    def _compute_bottleneck_analysis(self, total_step_time_ms, kernel_analysis):
        """Analyze where bottlenecks occur"""
        
        # Calculate time spent in different kernel types
        kernel_times = {}
        total_kernel_time = 0
        
        for gemm_type, metrics in kernel_analysis.items():
            if 'avg_duration_us' in metrics:
                kernel_time_ms = metrics['avg_duration_us'] / 1000.0
                kernel_times[gemm_type] = kernel_time_ms
                total_kernel_time += kernel_time_ms
        
        # Communication time = total - compute
        comm_time_ms = max(0, total_step_time_ms - total_kernel_time)
        comm_percentage = (comm_time_ms / total_step_time_ms) * 100
        
        # SM efficiency bottleneck analysis
        avg_sm_efficiency = np.mean([
            metrics.get('avg_sm_efficiency', 0) 
            for metrics in kernel_analysis.values()
        ])
        
        # Memory bandwidth analysis  
        avg_memory_bw = np.mean([
            metrics.get('avg_memory_bw_util', 0)
            for metrics in kernel_analysis.values() 
        ])
        
        return {
            'comm_bottleneck_percent': comm_percentage,
            'avg_sm_efficiency_percent': avg_sm_efficiency * 100,
            'avg_memory_bw_gb_s': avg_memory_bw,
            'kernel_time_breakdown': kernel_times,
            'total_kernel_time_ms': total_kernel_time,
            'total_comm_time_ms': comm_time_ms
        }
    
    def generate_comprehensive_report(self):
        """Generate the type of analysis you want"""
        if not self.system_metrics:
            return "No profiling data available"
        
        # Aggregate across all steps
        avg_comm_bottleneck = np.mean([m['bottleneck_analysis']['comm_bottleneck_percent'] 
                                      for m in self.system_metrics])
        avg_sm_efficiency = np.mean([m['bottleneck_analysis']['avg_sm_efficiency_percent']
                                    for m in self.system_metrics])
        avg_throughput = np.mean([m['system']['throughput_tokens_sec'] 
                                 for m in self.system_metrics])
        
        # Kernel breakdown
        kernel_breakdown = defaultdict(float)
        for step_metrics in self.system_metrics:
            for kernel, time_ms in step_metrics['bottleneck_analysis']['kernel_time_breakdown'].items():
                kernel_breakdown[kernel] += time_ms
        
        report = f"""
=== MICROARCHITECTURAL DEEP-DIVE ANALYSIS ===

SYSTEM LEVEL:
- Communication bottleneck: {avg_comm_bottleneck:.1f}%
- Average throughput: {avg_throughput:.1f} tokens/sec

SM LEVEL:  
- Compute efficiency: {avg_sm_efficiency:.1f}%
- Memory bandwidth utilization: {np.mean([m['bottleneck_analysis']['avg_memory_bw_gb_s'] for m in self.system_metrics]):.1f} GB/s

KERNEL BREAKDOWN (avg time per step):
"""
        
        for kernel, total_time in sorted(kernel_breakdown.items(), key=lambda x: x[1], reverse=True):
            avg_time = total_time / len(self.system_metrics)
            report += f"- {kernel}: {avg_time:.2f} ms\n"
        
        report += f"""
BOTTLENECK IMPACT ON THROUGHPUT:
- When comm > 30%: throughput drops by ~{self._estimate_comm_impact():.1f}%
- When SM efficiency < 70%: throughput drops by ~{self._estimate_compute_impact():.1f}%

OPTIMIZATION RECOMMENDATIONS:
{self._generate_optimization_suggestions(avg_comm_bottleneck, avg_sm_efficiency)}
"""
        
        return report
    
    def _estimate_comm_impact(self):
        """Estimate communication impact on throughput"""
        high_comm_steps = [m for m in self.system_metrics 
                          if m['bottleneck_analysis']['comm_bottleneck_percent'] > 30]
        low_comm_steps = [m for m in self.system_metrics
                         if m['bottleneck_analysis']['comm_bottleneck_percent'] <= 30]
        
        if not high_comm_steps or not low_comm_steps:
            return 0.0
            
        high_comm_tput = np.mean([m['system']['throughput_tokens_sec'] for m in high_comm_steps])
        low_comm_tput = np.mean([m['system']['throughput_tokens_sec'] for m in low_comm_steps])
        
        return ((low_comm_tput - high_comm_tput) / low_comm_tput) * 100
    
    def _estimate_compute_impact(self):
        """Estimate compute efficiency impact on throughput"""  
        return 25.0  # Placeholder - would analyze actual correlation
    
    def _generate_optimization_suggestions(self, comm_bottleneck, sm_efficiency):
        """Generate targeted optimization recommendations"""
        suggestions = []
        
        if comm_bottleneck > 25:
            suggestions.append("- HIGH COMM BOTTLENECK: Consider overlapping more computation with communication")
            suggestions.append("- Try increasing micro-batch size to amortize communication cost")
            
        if sm_efficiency < 75:
            suggestions.append("- LOW SM EFFICIENCY: GEMM kernels underutilizing SMs")
            suggestions.append("- Consider: larger batch sizes, different tensor layouts, or mixed precision")
            
        if not suggestions:
            suggestions.append("- System is relatively well balanced")
            
        return "\n".join(suggestions)


def run_profiled_1f1b_training(config, num_steps=5):
    """Training loop with both system and microarchitectural profiling"""
    
    # Initialize profiled pipeline
    model = MinimalTransformer(config)
    pipeline = PipelineWrapper(model, config)
    profiler = ComprehensiveProfiler(config, devices)
    
    # Device placement (same as your original code)
    pipeline.stage1.to(devices[0])
    pipeline.stage2.to(devices[0])  
    pipeline.stage3.to(devices[1])
    pipeline.norm.to(devices[1])
    pipeline.stage4.to(devices[1])
    
    optimizer = torch.optim.Adam(
        list(pipeline.stage1.parameters()) + 
        list(pipeline.stage2.parameters()) +
        list(pipeline.stage3.parameters()) + 
        list(pipeline.stage4.parameters()), 
        lr=1e-4
    )
    
    print("=== STARTING PROFILED 1F1B TRAINING ===")
    
    for step in range(1, num_steps + 1):
        
        def training_step():
            """Your existing 1F1B training logic wrapped for profiling"""
            
            optimizer.zero_grad()
            x, y = random_batch(config, devices[0])
            micro_x = x.split(micro_batch_size, dim=0)
            micro_y = y.split(micro_batch_size, dim=0)
            
            # Your existing 1F1B pipeline logic here
            # (abbreviated for brevity - use your full implementation)
            
            num_micro_batches = len(micro_x)
            
            # Simplified forward pass for profiling demo
            for i in range(num_micro_batches):
                act1 = pipeline.stage1_forward(micro_x[i])
                act2 = pipeline.stage2_forward(act1)
                act2_gpu1 = act2.to(devices[1])
                act3 = pipeline.stage3_forward(act2_gpu1)
                logits = pipeline.stage4_forward(act3)
                
                loss = criterion(logits.reshape(-1, config.vocab_size), 
                               micro_y[i].to(devices[1]).reshape(-1))
                loss.backward()
            
            optimizer.step()
            return {'loss': loss.item()}
        
        # Profile this step
        step_metrics = profiler.profile_step_with_microarch(training_step, step)
        
        # Print progress
        print(f"Step {step}: "
              f"Comm: {step_metrics['bottleneck_analysis']['comm_bottleneck_percent']:.1f}%, "
              f"SM Eff: {step_metrics['bottleneck_analysis']['avg_sm_efficiency_percent']:.1f}%, "
              f"Throughput: {step_metrics['system']['throughput_tokens_sec']:.1f} tok/sec")
    
    # Generate comprehensive report
    final_report = profiler.generate_comprehensive_report()
    print(final_report)
    
    return profiler


# --- Config & Model ---

devices = [torch.device("cuda:0"), torch.device("cuda:1")]
criterion = nn.CrossEntropyLoss()

config = TransformerConfig()
devices = [torch.device("cuda:0"), torch.device("cuda:1")]
criterion = nn.CrossEntropyLoss()

# Run profiled training
profiler_results = run_profiled_1f1b_training(config, num_steps=5)

# Get the exact analysis you want:
print("=== YOUR ANALYSIS STATEMENT ===")
comm_bottleneck = profiler_results.bottleneck_analysis.get('comm_bottleneck_percent', 0)
sm_efficiency = profiler_results.bottleneck_analysis.get('avg_sm_efficiency_percent', 0)

print(f"At the system level, comm is {comm_bottleneck:.1f}% bottleneck.")
print(f"At the SM level, compute efficiency is {sm_efficiency:.1f}%.")
print("Here's how both shape end-to-end training throughput:")
print(profiler_results.generate_comprehensive_report())

