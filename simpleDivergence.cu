#include <cuda_runtime.h>
#include <cstdlib>
#include <device_launch_parameters.h>


__global__ void mathKernel1(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    //b is floating point number 0, a is also set to the same as b (right to left)
    a = b = 0.0f;
    if (tid % 2 == 0) {
    a = 100.0f;
    } else {
    b = 200.0f;
    }
    c[tid] = a + b;
}


__global__ void mathKernel2(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else { 
        b = 200.0f;
    }
    c[tid] = a + b;
}


int main(int argc, char **argv) {

    // set up device
    int dev = 0;        //specify which gpu device to use 
    cudaDeviceProp deviceProp;      //cudaDeviceProp is a structure type that holds the information about the gpu device 
    cudaGetDeviceProperties(&deviceProp, dev);      //fetch the properties of the GPU specified by dev and stores it in deivceProp
    printf("%s using Device %d: %s\n", argv[0],dev, deviceProp.name);

    // set up data size
    int size = 64;  
    int blocksize = 64;
    if(argc > 1) blocksize = atoi(argv[1]);
    if(argc > 2) size = atoi(argv[2]);
    printf("Data size %d ", size);

    //set up the size 
    // set up execution configuration
    dim3 block (blocksize,1);
    dim3 grid ((size+block.x-1)/block.x,1);
    printf("Execution Configure (block %d grid %d)\n",block.x, grid.x);

    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);
    
    // run a warmup kernel to remove overhead
    size_t iStart,iElaps;
    cudaDeviceSynchronize();
    iStart = seconds();
    warmingup<<<grid, block>>> (d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("warmup <<< %4d %4d >>> elapsed %d sec \n",grid.x,block.x, iElaps );

    // run kernel 1
    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel1 <<< %4d %4d >>> elapsed %d sec \n",grid.x,block.x,iElaps );

    // run kernel 2
    iStart = seconds();
    mathKernel2<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds () - iStart;
    printf("mathKernel2 <<< %4d %4d >>> elapsed %d sec \n",grid.x,block.x,iElaps );

    // free gpu memory and reset divece
    cudaFree(d_C);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}