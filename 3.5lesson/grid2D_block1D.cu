/*********************************************************************************************
 * file name  : grid2D_block1D.cu
 * author     : 权 双
 * date       : 2023-08-14
 * brief      : 组织线程模型：二维网格一维线程块计算二维矩阵加法
***********************************************************************************************/

#include <stdio.h>
#include "../tools/common.cuh"

__global__ void addMatrix(int *A, int *B, int *C, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    // 1、设置GPU设备
    setGPU();

    // 2、分配主机内存和设备内存，并初始化
    int nx = 16;
    int ny = 8;
    int nxy = nx * ny;
    size_t stBytesCount = nxy * sizeof(int);
     
     // （1）分配主机内存，并初始化
    int *ipHost_A, *ipHost_B, *ipHost_C;
    ipHost_A = (int *)malloc(stBytesCount);
    ipHost_B = (int *)malloc(stBytesCount);
    ipHost_C = (int *)malloc(stBytesCount);
    if (ipHost_A != NULL && ipHost_B != NULL && ipHost_C != NULL)
    {
        for (int i = 0; i < nxy; i++)
            {
                ipHost_A[i] = i;
                ipHost_B[i] = i + 1;
            }
        memset(ipHost_C, 0, stBytesCount); 
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }
    

    // （2）分配设备内存，并初始化
    int *ipDevice_A, *ipDevice_B, *ipDevice_C;
    ErrorCheck(cudaMalloc((int**)&ipDevice_A, stBytesCount), __FILE__, __LINE__); 
    ErrorCheck(cudaMalloc((int**)&ipDevice_B, stBytesCount), __FILE__, __LINE__); 
    ErrorCheck(cudaMalloc((int**)&ipDevice_C, stBytesCount), __FILE__, __LINE__); 
    if (ipDevice_A != NULL && ipDevice_B != NULL && ipDevice_C != NULL)
    {
        ErrorCheck(cudaMemcpy(ipDevice_A, ipHost_A, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__); 
        ErrorCheck(cudaMemcpy(ipDevice_B, ipHost_B, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__); 
        ErrorCheck(cudaMemcpy(ipDevice_C, ipHost_C, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__); 
    }   
    else
    {
        printf("Fail to allocate memory\n");
        free(ipHost_A);
        free(ipHost_B);
        free(ipHost_C);
        exit(1);
    }

    // calculate on GPU
    dim3 block(4, 1);
    dim3 grid((nx + block.x -1) / block.x, ny);
    printf("Thread config:grid:<%d, %d>, block:<%d, %d>\n", grid.x, grid.y, block.x, block.y);
    
    addMatrix<<<grid, block>>>(ipDevice_A, ipDevice_B, ipDevice_C, nx, ny);  // 调用内核函数
    
    ErrorCheck(cudaMemcpy(ipHost_C, ipDevice_C, stBytesCount, cudaMemcpyDeviceToHost), __FILE__, __LINE__); 
    for (int i = 0; i < 10; i++)
    {
        printf("id=%d, matrix_A=%d, matrix_B=%d, result=%d\n", i + 1,ipHost_A[i], ipHost_B[i], ipHost_C[i]);
    }

    free(ipHost_A);
    free(ipHost_B);
    free(ipHost_C);

    ErrorCheck(cudaFree(ipDevice_A), __FILE__, __LINE__); 
    ErrorCheck(cudaFree(ipDevice_B), __FILE__, __LINE__); 
    ErrorCheck(cudaFree(ipDevice_C), __FILE__, __LINE__); 

    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__); 
    return 0;
}
