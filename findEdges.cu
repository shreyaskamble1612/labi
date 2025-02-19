#include <iostream>
#include <cuda_runtime.h>

#define WIDTH 512
#define HEIGHT 512

// Sobel kernel
__global__ void sobelEdgeDetection(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int Gx = 0;
        int Gy = 0;

        int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixel = input[(y + i) * width + (x + j)];
                Gx += sobelX[i + 1][j + 1] * pixel;
                Gy += sobelY[i + 1][j + 1] * pixel;
            }
        }
        int magnitude = sqrt(Gx * Gx + Gy * Gy);

        magnitude = min(max(magnitude, 0), 255);
        
        output[y * width + x] = magnitude;
    }
}

int main() {
    unsigned char* h_input = new unsigned char[WIDTH * HEIGHT];
    unsigned char* h_output = new unsigned char[WIDTH * HEIGHT];

    unsigned char *d_input, *d_output;
    
    cudaMalloc(&d_input, WIDTH * HEIGHT * sizeof(unsigned char));
    cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(unsigned char));

    cudaMemcpy(d_input, h_input, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    sobelEdgeDetection<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, HEIGHT);

    cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

//nvcc sobel_edge_detection.cu -o sobel

