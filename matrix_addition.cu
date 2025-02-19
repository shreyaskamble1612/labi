#include<cuda_runtime.h>
#include<iostream>

#define N 16 

__global__ void matrixAdd(int *A,int *B,int *C,int width){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < width && col < width){
		int index = row * width + col;
		C[index] = A[index] + B[index];
	}
}

int main(){
	int size = N*N*sizeof(int);
	int A[N][N],B[N][N],C[N][N];
	int *d_A.*d_B,*d_C;
	
	for(int i = 0;i<N;i++){
	    for(int j = 0;j<N;j++){
		A[i][j] = i+j;
		B[i][j] = i-j;
		}
	}

	cudaMalloc((void **)&d_A,size);
	cudaMalloc((void **)&d_B,size);
	cudaMalloc((void **)&d_C,size);

	cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);
	
	
	dim3 threadPerBLock(16,16);
	dim3 blocksPerGrid((N + threadsPerBLock.x - 1) / threadsPerBLock.x,
	(N + threadPerBLock.y - 1) /threadPerBLock,y);
	
	
	matrixAdd<<<blockPerGrid,threadPerBlock>>>(d_A,d_B,d_C,N);
	cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);
	
	coud<<"Result matrix c:\n";
	for(int i = 0;i<N;i++){
	   for(int j - 0;j<N;j++){
		cout<<C[i][j]<<" ";		
		}
		cout<<endl;
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	
	return 0;
}
