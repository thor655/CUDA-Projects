/**
*   CS6023: GPU Programming 
*   Assignment 2
*   
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree() 
*   to free up memory as soon as you're done with an allocation. 
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__
void convolution2D(long int *h_mat, long int *h_filter, long int* h_ans, int m, int n, int k){
    extern __shared__ long int filter_shared[];
    int ii,jj;
    //Parallely Copy the filter to shared memory.
    for(ii = 0 ; ii<ceil((double)k*k/blockDim.x); ii++){
        if(ii*blockDim.x+threadIdx.x < k*k)
            filter_shared[ii*blockDim.x+threadIdx.x] = h_filter[ii*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    long int total=0, i, j;
    for(ii=0; ii<k ; ii++){
        for(jj=0; jj<k; jj++){
            i = blockIdx.x  + ii - (k-1)/2;
            j = threadIdx.x + jj - (k-1)/2;
            if(i>=0 && i<m && j>=0 && j<n){
                total += filter_shared[ii*k+jj] * h_mat[i*n+j];  
                //coalescing at j
                //filter in shared
            }
        }
    } 
    h_ans[blockDim.x*blockIdx.x+threadIdx.x] = total;
}

int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/
    
    long int *h_mat_gpu, *h_filter_gpu, *h_ans_gpu;

    cudaMalloc(&h_mat_gpu, m*n*sizeof(long int));
    cudaMemcpy(h_mat_gpu,h_mat,m*n*sizeof(long int),cudaMemcpyHostToDevice);

    cudaMalloc(&h_filter_gpu, k*k*sizeof(long int));
    cudaMemcpy(h_filter_gpu,h_filter,k*k*sizeof(long int),cudaMemcpyHostToDevice);

    cudaMalloc(&h_ans_gpu, m*n*sizeof(long int));

    cudaDeviceSynchronize();

    auto shared_memory_size = k*k*sizeof(long int);
    cudaDeviceSynchronize();
    cudaFuncSetCacheConfig(convolution2D, cudaFuncCachePreferShared);
    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch
    convolution2D<<<m,n,shared_memory_size>>>(h_mat_gpu,h_filter_gpu,h_ans_gpu,m,n,k);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    cudaMemcpy(h_ans, h_ans_gpu, m*n*sizeof(long int), cudaMemcpyDeviceToHost);
    
    std::chrono::duration<double> elapsed1 = end - start;
    printf("Time : %f\n",elapsed1.count());

    cudaFree(h_mat_gpu);
    cudaFree(h_filter_gpu);
    cudaFree(h_ans_gpu);
    
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */


    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}
