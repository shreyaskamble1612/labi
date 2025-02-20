#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void rgbToGrayKernel(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = d_input[idx];
        unsigned char g = d_input[idx + 1];
        unsigned char b = d_input[idx + 2];
        d_output[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void convertToGrayGPU(const Mat &input, Mat &output) {
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t img_size = width * height * channels * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, img_size);
    cudaMalloc((void **)&d_output, gray_size);

    cudaMemcpy(d_input, input.data, img_size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    rgbToGrayKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    cudaMemcpy(output.data, d_output, gray_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    Mat input = imread("input.jpg", IMREAD_COLOR);
    if (input.empty()) {
        printf("Could not open or find the image!\n");
        return -1;
    }
    Mat output(input.rows, input.cols, CV_8UC1);
    
    convertToGrayGPU(input, output);
    imwrite("output.jpg", output);
    
    printf("Grayscale image saved as output.jpg\n");
    return 0;
}

//nvcc grayscale_cuda.cu -o grayscale -lopencv_core -lopencv_imgcodecs -lopencv_highgui

