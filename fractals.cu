#include "IO.h"
#include <stdio.h>


__device__ vec2 calcNext(vec2 z, vec2 c) {
    const Float zr = z.x * z.x - z.y * z.y;
    const Float zi = 2.f * z.x * z.y;

    return vec2{ zr, zi } + c;
}

__device__ int calcIterations(vec2 z0, vec2 c, int max_iter) {
    vec2 zn = z0;
    int iter = 0;

    while ((zn.x * zn.x + zn.y * zn.y <= 4.0f) && (iter < max_iter)) {
        zn = calcNext(zn, c);
        iter++;
    }

    return iter;
}

__device__ Float maxf(Float a, Float b) {
    return (a > b) ? a : b;
}

__device__ Float minf(Float a, Float b) {
    return (a < b) ? a : b;
}

#define MAX_ITER 50

static vec2 cameraPos{ 0, 0 };
static Float cameraZoom = 0.2;

__global__ void calcMandelbrot(IO::RGB* buffer, int width, int height, vec2 z0, int max_iterations, Float zoom, vec2 center) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * width + col;

    Float wph = (Float)width / (Float)height;
    Float x0 = ((Float)col / (Float)width) / zoom - (0.5 / zoom) - center.x;
    Float y0 = (((Float)row / (Float)height) / zoom - (0.5 / zoom) - center.y) / wph;

    vec2 z = { x0, y0 };
    vec2 c = { z0.x, z0.y };
    int iter = 0;
    Float xtemp = 0;
    while ((z.x * z.x + z.y * z.y <= 4.0f) && (iter < max_iterations)) {
        xtemp = z.x * z.x - z.y * z.y + c.x;
        z.y = 2.0f * z.x * z.y + c.y;
        z.x = xtemp;
        iter++;
    }

    Float color = 
        //5.0 * 
        ((Float)iter 
         * 255.l / (Float)max_iterations
         -  log2f(maxf(1.f, log2f(length(z))))
        );


    buffer[i].r = minf(255.f, color);
    buffer[i].g = minf(255.f, color);
    buffer[i].b = minf(255.f, color);
}

cudaError_t mandelbrotCuda(int maxIter, vec2 z0 = { 0, 0 }) {
    IO::RGB* gpuBuffer = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&gpuBuffer, g_windowWidth * g_windowHeight * sizeof(IO::RGB));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    dim3 block_size(16, 16);
    dim3 grid_size(g_windowWidth / block_size.x, g_windowHeight / block_size.y);

    calcMandelbrot << < grid_size, block_size >> > (gpuBuffer, g_windowWidth, g_windowHeight, z0, maxIter, cameraZoom, cameraPos);

    //// Check for any errors launching the kernel
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
    cudaStatus = cudaMemcpy((unsigned char*)IO::GetOutputBuffer(), gpuBuffer, g_windowWidth * g_windowHeight * sizeof(IO::RGB), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(gpuBuffer);

    return cudaStatus;
}

int main() {
    IO::OpenWindow(800, 800);
    
    vec2 z0;
    vec2 dragStart; // normalized
    while (!SDL_QuitRequested()) {
        IO::HandleEvents();
        vec2 normalizedMousePos = IO::NormalizePixel(IO::GetMousePos().x, IO::GetMousePos().y);

        const Uint8* state = SDL_GetKeyboardState(nullptr);
        if (state[SDL_SCANCODE_SPACE]) {
            z0 = { 0, 0 };
            cameraPos = { 0, 0 };
            cameraZoom = 0.2;
        }

        static vec2 z0Start;
        if (IO::MouseClicked(SDL_BUTTON_LEFT)) {
            z0Start = z0;
            dragStart = normalizedMousePos;
        }
        if (IO::IsButtonDown(SDL_BUTTON_LEFT)) {
            z0 = z0Start + 0.25l * (normalizedMousePos - dragStart) / cameraZoom;
        }

        static Float zoom = 1.l;
        static Float zoomDP = 1.02;
        static Float zoomDN = 1.035;
        if (IO::GetMouseWheel() > 0) 
            zoom = zoomDP;
        else if (IO::GetMouseWheel() < 0)
            zoom = 1.l / zoomDN;
        if (abs(zoom - 1.l) > 0.0001f)
            cameraPos = cameraPos - 0.5l * normalizedMousePos / cameraZoom + 0.5l * normalizedMousePos / (cameraZoom * zoom);
        cameraZoom *= zoom;
        zoom = 1.l + (zoom - 1.l) * 0.975l;

        
        mandelbrotCuda(((float)MAX_ITER * powl(cameraZoom, 1.0l / 12.l)), z0);
        
    	IO::Render();
    }
    
    IO::Quit();
    
    return 0;
}
