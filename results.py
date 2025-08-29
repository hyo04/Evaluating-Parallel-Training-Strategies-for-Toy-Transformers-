import matplotlib.pyplot as plt

'''
# ---------------- batch_size VS throughput --------------
batch_size = [128, 256, 512] 

sequential = [30985.8, 30384.9, 28173.9]
data = [36467.8, 50566.2, 52946.9]
pipeline = [6257.6, 12052.4, 23859.0]


plt.figure(figsize =(7,5))
plt.plot(batch_size, sequential, marker='x',linewidth=2, label = "Sequential")
plt.plot(batch_size, data, marker='x',linewidth=2, label = "Data parallel" )
plt.plot(batch_size, pipeline, marker='x',linewidth=2, label = "Pipeline parallel")


plt.title("Sequence length VS Throughput")
plt.xlabel("Sequence length")
plt.ylabel("Throughput (tokens/sec)")
plt.legend()
plt.show()


# -------------- micro_batch_size VS throughput ----------------
nmicro_batch_size = [2, 4, 6, 8, 16]
mb_throughputs = [12437.8, 23390.3, 31863.8, 32207.5, 33428.9]

plt.figure(figsize=(7,5))
plt.plot(batch_size, seq_throughput, marker='x', linewidth=2)

plt.title("Micro batch size VS Throughput")
plt.xlabel("Micro batch size")
plt.ylabel("Throughput (tokens/sec)")
plt.legend("Bseline configuration")
plt.show()
'''


# ------------- batch size vs throughput -----------------------
# DATA PARALLEL -> SEQUENTIAL -> PIPELINE PARALLEL (pipeline does the worst)
batch_size = [16,32,64,128]

sequential = [27853.2, 30628.4, 32051.9, 32060]
data = [40617.2, 51082.1, 58568.8, 48285.9]
pipeline = [12131.0, 12479.7, 12658.0, 12881.0]

plt.figure(figsize =(7,5))
plt.plot(batch_size, sequential, marker='x',linewidth=2, label = "Sequential")
plt.plot(batch_size, data, marker='x',linewidth=2, label = "Data parallel" )
plt.plot(batch_size, pipeline, marker='x',linewidth=2, label = "Pipeline parallel")

plt.title("Sequence length VS Throughput")
plt.xlabel("Sequence length")
plt.ylabel("Throughput (tokens/sec)")
plt.legend()
plt.show()