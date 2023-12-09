#include "utils/Json.h"
#include "utils/ExternalResource.h"
#include "IO.h"

#include <stdio.h>

JSON_C(vec2, JSON_M(x), JSON_M(y))

struct Camera {
    vec2 position;
    Float zoom;
}; JSON_C(Camera, JSON_M(position), JSON_M(zoom))

enum class FractalType { Mandelbrot = 0x00, Julia, _COUNT };

struct Options {
    int windowWidth, windowHeight;
    Camera camera{ vec2(), 0.2 };

    FractalType type = FractalType::Julia;
    int baseIterations = 50;
    Float iterationIncreaseFallOff = 12.l;
    vec2 z0;
    vec2 c;

    void SetProperty(vec2 value) {
        switch (type)
        {
        case FractalType::Mandelbrot:
            c = value; break;
        case FractalType::Julia:
            z0 = value; break;
        default:
            break;
        }
    }

    __device__ __host__ vec2 GetProperty() {
        if (type == FractalType::Mandelbrot) return c;
        return z0;
    }

}; JSON_C(Options, JSON_M(windowWidth), JSON_M(windowHeight), JSON_M(camera), JSON_M(type), JSON_M(z0), JSON_M(c))

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

__global__ void calcMandelbrot(IO::RGB* buffer, Options options, int maxIterations) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * options.windowWidth + col;

    Float wph = (Float)options.windowWidth / (Float)options.windowHeight;
    Float x0 = ((Float)col / (Float)options.windowWidth) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.x;
    Float y0 = (((Float)row / (Float)options.windowHeight) / options.camera.zoom - (0.5 / options.camera.zoom) - options.camera.position.y) / wph;

    vec2 z, c;
    if (options.type == FractalType::Mandelbrot) {
        z = options.GetProperty();
        c = { x0, y0 };
    }
    else {
        z = { x0, y0 };
        c = options.GetProperty();
    }

    int iter = 0;
    Float xtemp = 0;
    while ((z.x * z.x + z.y * z.y <= 4.0f) && (iter < maxIterations)) {
        xtemp = z.x * z.x - z.y * z.y + c.x;
        z.y = 2.0f * z.x * z.y + c.y;
        z.x = xtemp;
        iter++;
    }

    Float color = 
        //5.0 * 
        ((Float)iter 
         * 255.l / (Float)maxIterations
         -  log2f(maxf(1.f, log2f(length(z))))
        );


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
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&gpuBuffer, options.windowWidth * options.windowHeight * sizeof(IO::RGB));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    dim3 block_size(16, 16);
    dim3 grid_size(options.windowWidth / block_size.x, options.windowHeight / block_size.y);

    calcMandelbrot << < grid_size, block_size >> > (gpuBuffer, options, maxIterations);

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
    cudaStatus = cudaMemcpy((unsigned char*)IO::GetOutputBuffer(), gpuBuffer, options.windowWidth * options.windowHeight * sizeof(IO::RGB), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(gpuBuffer);

    return cudaStatus;
}

auto& g_options = external_resource<"options.json", Json::wrap<Options>>::value;

int main() {
    IO::OpenWindow(g_options.windowWidth, g_options.windowHeight);
    
    vec2 dragStart; // normalized
    while (!SDL_QuitRequested()) {
        IO::HandleEvents();
        g_options.windowWidth = IO::GetWindowWidth();
        g_options.windowHeight = IO::GetWindowHeight();

        vec2 normalizedMousePos = IO::NormalizePixel(IO::GetMousePos().x, IO::GetMousePos().y);

        const Uint8* state = SDL_GetKeyboardState(nullptr);
        if (state[SDL_SCANCODE_SPACE]) {
            g_options.SetProperty({ 0, 0 });
            g_options.camera.position = { 0, 0 };
            g_options.camera.zoom = 0.2;
        }

        static vec2 z0Start;
        if (IO::MouseClicked(SDL_BUTTON_LEFT)) {
            z0Start = g_options.GetProperty();
            dragStart = normalizedMousePos;
        }
        if (IO::IsButtonDown(SDL_BUTTON_LEFT)) {
            g_options.SetProperty(z0Start + 0.25l * (normalizedMousePos - dragStart) / g_options.camera.zoom);
        }

        static Float zoom = 1.l;
        static Float zoomDP = 1.02;
        static Float zoomDN = 1.035;
        if (IO::GetMouseWheel() > 0) 
            zoom = zoomDP;
        else if (IO::GetMouseWheel() < 0)
            zoom = 1.l / zoomDN;
        if (abs(zoom - 1.l) > 0.0001f)
            g_options.camera.position = g_options.camera.position - 0.5l * normalizedMousePos / g_options.camera.zoom + 0.5l * normalizedMousePos / (g_options.camera.zoom * zoom);
        g_options.camera.zoom *= zoom;
        zoom = 1.l + (zoom - 1.l) * 0.975l;

        mandelbrotCuda(g_options, (float)g_options.baseIterations * powl(g_options.camera.zoom, 1.0l / g_options.iterationIncreaseFallOff));
        
    	IO::Render();
    }
    
    IO::Quit();

    return 0;
}
