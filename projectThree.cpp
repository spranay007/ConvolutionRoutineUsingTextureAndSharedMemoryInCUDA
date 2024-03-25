#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "convolutionTexture_common.h"

using namespace std;

int dimX, dimY, dimK;

int main(int argc, char** argv) {
    float* h_Kernel, * h_Input, * h_OutputGPU, * h_OutputCPU, * h_Buffer ;
    //cudaArray is an opaque block of memory that is optimized for binding to textures.
    cudaArray* a_Src;
    //Cuda texture object
    cudaTextureObject_t texSrc;
    //Returns a channel descriptor using the specified format
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    float* d_Output;
    
    StopWatchInterface* hTimer = NULL;
    float elapsedSeconds = 0;
    float milliseconds = 0;
    printf("[%s] - Starting...\n", argv[0]);

    // Parse command-line arguments for image dimensions and kernel size
    if (argc != 7) {
        printf("Usage: %s -i <dimX> -j <dimY> -k <dimK>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            dimX = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-j") == 0) {
            dimY =  atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-k") == 0) {
            dimK = atoi(argv[++i]);
        }
    }
    int kernelLengthUser = dimK;
    int kernelRadiusUser = 0;
    if (dimK % 2 == 0)
    {
        kernelRadiusUser = (kernelLengthUser / 2) - 1; //incase of even
    }
    else
    {
        kernelRadiusUser = kernelLengthUser / 2;    //incase of odd
    }
    cout << "Kernel Length Input : " << kernelLengthUser << endl;
    cout << "Kernel Radius User : " << kernelRadiusUser << endl;

    findCudaDevice(argc, (const char**)argv);
    sdkCreateTimer(&hTimer);
    printf("Initializing data...\n");

    // Allocate host memory
    h_Kernel = (float*)malloc(kernelLengthUser * sizeof(float));
    h_Input = (float*)malloc(dimX * dimY * sizeof(float));
    h_OutputGPU = (float*)malloc(dimX * dimY * sizeof(float));

    h_Buffer = (float*)malloc(dimX * dimY * sizeof(float));
    h_OutputCPU = (float*)malloc(dimX * dimY * sizeof(float));

    // Allocate device memory
    checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, dimX, dimY));
    checkCudaErrors(cudaMalloc((void**)&d_Output, dimX * dimY * sizeof(float)));

    // Generate random values for kernel and input image
    // Seed the random number generator
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < kernelLengthUser; i++) {
        h_Kernel[i] = (float)(rand() % 16);
    }
    for (int i = 0; i < dimX * dimY; i++) {
        h_Input[i] = (float)(rand() % 16);
    }

    // Set convolution kernel
    setConvolutionKernel(h_Kernel, kernelLengthUser);

    // Copy input image data to device
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, h_Input, dimX * dimY * sizeof(float), cudaMemcpyHostToDevice));

    // Create texture object
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = a_Src;
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.readMode = cudaReadModeElementType;
    checkCudaErrors(cudaCreateTextureObject(&texSrc, &texRes, &texDesc, NULL));
    
    printf("Running convolution Rows and Coloumns on the GPU \n");
    // Start timer
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Launch GPU kernels for convolution
    convolutionRowsGPU(d_Output, a_Src, dimX, dimY, texSrc, kernelRadiusUser);
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, d_Output,dimX * dimY * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    convolutionColumnsGPU(d_Output, a_Src, dimX, dimY, texSrc, kernelRadiusUser);
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    sdkStopTimer(&hTimer);
    milliseconds = sdkGetTimerValue(&hTimer);
    printf("GPU convolution time: %f ms, %f Mpix/s\n", milliseconds, dimX * dimY * 1e-6 / (0.001 * milliseconds));

    // Calculate total number of FLOPs for each kernel
    // Each element-wise multiplication and addition counts as 2 FLOPs
    int numFlopsPerElement = 2;
    int numElements = dimX * dimY;
    int numFlops = numFlopsPerElement * numElements * dimK; // Assuming dimK is the number of elements in the kernel
    elapsedSeconds = milliseconds / 1000.0;
    // Calculate GFLOPS
    float gflops = (float)numFlops / (elapsedSeconds * 1.0e9f);

    // Print GFLOPS
    printf("GFLOPS: %.2f\n", gflops);


    // Copy GPU result back to host
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, dimX * dimY * sizeof(float), cudaMemcpyDeviceToHost));
    
    /////////////////////////////////CPU Verification Code Starts////////////////////////////////////////////////
    printf("Checking the results...\n");
    printf("...running convolutionRowsCPU()\n");
    convolutionRowsCPU(h_Buffer, h_Input, h_Kernel, dimX, dimY, kernelRadiusUser);

    printf("...running convolutionColumnsCPU()\n");
    convolutionColumnsCPU(h_OutputCPU, h_Buffer, h_Kernel, dimX, dimY, kernelRadiusUser);

    double delta = 0;
    double sum = 0;

    for (unsigned int i = 0; i < dimX * dimY; i++) {
        sum += h_OutputCPU[i] * h_OutputCPU[i];
        delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
        if (h_OutputGPU[i] != h_OutputCPU[i])
        {
//            cout << "failed at " << i << "!!" << endl;
            //cout << "difference at " << i << "is : " << h_OutputGPU[i] - h_OutputCPU[i] <<endl;
        }
    }
    
    double L2norm = sqrt(delta / sum);
    printf("Relative L2 norm: %E\n", L2norm);
    /////////////////////////////////CPU Verification Code Ends////////////////////////////////////////////////

    printf("Shutting down...\n");

    // Cleanup
    cudaDestroyTextureObject(texSrc);
    cudaFreeArray(a_Src);
    cudaFree(d_Output);
    free(h_Kernel);
    free(h_Input);
    free(h_OutputGPU);
    sdkDeleteTimer(&hTimer);

    return 0;
}
