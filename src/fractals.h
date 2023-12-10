#pragma once
#include "vec2.h"
#include "utils/Json.h"

JSON_C(vec2, JSON_M(x), JSON_M(y))

// Structure representing a 2D camera with a position and zoom factor
struct Camera {
    vec2 position;
    Float zoom;
}; JSON_C(Camera, JSON_M(position), JSON_M(zoom))

// Enumeration defining different types of fractals, including Mandelbrot and Julia
enum class FractalType { Mandelbrot = 0x00, Julia, _COUNT };

// Structure representing options for rendering fractals, including window dimensions,
// camera settings, fractal type, and iteration parameters
struct Options {
    int windowWidth = 800;                  // Width of the rendering window
    int windowHeight = 800;                 // Height of the rendering window
    Camera camera{ vec2(), 0.2 };           // Camera configuration with default values

    FractalType type = FractalType::Julia;  // Default fractal type is Julia
    int baseIterations = 50;                // Initial number of iterations for fractal computation
    float iterationIncreaseFallOff = 12.L;  // Fall-off factor for increasing iterations
    vec2 z0;                                // Complex number property for Julia set
    vec2 c;                                 // Complex number property for Mandelbrot set

    // Method for setting a property based on the current fractal type
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

    // Method for getting the current property based on the fractal type
    __device__ __host__ vec2 GetProperty() {
        if (type == FractalType::Mandelbrot) return c;
        return z0;
    }

}; JSON_C(Options, JSON_M(windowWidth), JSON_M(windowHeight), JSON_M(camera),
        JSON_M(type), JSON_M(baseIterations), JSON_M(iterationIncreaseFallOff),
        JSON_M(z0), JSON_M(c))

/**
 * Calculates the Mandelbrot set on the GPU using CUDA.
 *
 * @param options - The options specifying rendering parameters.
 * @param maxIterations - The maximum number of iterations for fractal computation.
 *
 * @return cudaError_t - Returns cudaSuccess on successful execution, or an error code otherwise.
 */
cudaError_t mandelbrotCuda(Options options, int maxIterations);
