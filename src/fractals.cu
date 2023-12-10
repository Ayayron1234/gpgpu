#include "fractals.h"

#include "utils/ExternalResource.h"
#include "IO.h"

#include <stdio.h>

__device__ Float maxf(Float a, Float b) {
    return (a > b) ? a : b;
}

__device__ Float minf(Float a, Float b) {
    return (a < b) ? a : b;
}

/**
 * CUDA kernel function: Computes the color of a pixel in the Mandelbrot or Julia set and stores it in the buffer.
 *
 * @param buffer - The output buffer storing RGB values of pixels.
 * @param options - The Mandelbrot set properties and camera configuration.
 * @param maxIterations - The maximum number of iterations for the fractal computation.
 */
__global__ void calcMandelbrot(IO::RGB* buffer, Options options, int maxIterations) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * options.windowWidth + col;

    Float wph = (Float)options.windowWidth / (Float)options.windowHeight;
    Float x0 = ((Float)col / (Float)options.windowWidth) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.x;
    Float y0 = (((Float)row / (Float)options.windowHeight) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.y) / wph;

    vec2 z, c;

    // Determine the initial values for the fractal computation based on the selected type
    if (options.type == FractalType::Mandelbrot) {
        z = options.GetProperty();
        c = { x0, y0 };
    }
    else {
        z = { x0, y0 };
        c = options.GetProperty();
    }

    // Iterate to determine the color of the pixel
    int iter = 0;
    Float xtemp = 0;
    while ((z.x * z.x + z.y * z.y <= 4.0f) && (iter < maxIterations)) {
        xtemp = z.x * z.x - z.y * z.y + c.x;
        z.y = 2.0f * z.x * z.y + c.y;
        z.x = xtemp;
        iter++;
    }

    // Compute the color intensity based on the iteration count and the length of the complex number
    Float color = 
        //5.0 * 
        ((Float)iter 
         * 255.l / (Float)maxIterations
         -  log2f(maxf(1.f, log2f(length(z))))
        );

    // Store the RGB color values in the buffer, clamped to the range [0, 255]
    buffer[i].r = minf(255.f, color);
    buffer[i].g = minf(255.f, color);
    buffer[i].b = minf(255.f, color);
}

cudaError_t mandelbrotCuda(Options options, int maxIterations) {
    IO::RGB* gpuBuffer = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed? ");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&gpuBuffer, options.windowWidth * options.windowHeight * sizeof(IO::RGB));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    dim3 block_size(16, 16);
    dim3 grid_size(options.windowWidth / block_size.x, options.windowHeight / block_size.y);

    calcMandelbrot <<<grid_size, block_size >>> (gpuBuffer, options, maxIterations);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy((unsigned char*)IO::GetOutputBuffer(), gpuBuffer, options.windowWidth * options.windowHeight * sizeof(IO::RGB), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(gpuBuffer);

    return cudaStatus;
}
