#pragma once
#include "vec2.h"
#include "utils/Json.h"

JSON_C(vec2, JSON_M(x), JSON_M(y))

struct Camera {
    vec2 position;
    Float zoom;
}; JSON_C(Camera, JSON_M(position), JSON_M(zoom))

enum class FractalType { Mandelbrot = 0x00, Julia, _COUNT };

struct Options {
    int windowWidth = 800, windowHeight = 800;
    Camera camera{ vec2(), 0.2 };

    FractalType type = FractalType::Julia;
    int baseIterations = 50;
    float iterationIncreaseFallOff = 12.l;
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

}; JSON_C(Options, JSON_M(windowWidth), JSON_M(windowHeight), JSON_M(camera), JSON_M(type), JSON_M(baseIterations), JSON_M(iterationIncreaseFallOff), JSON_M(z0), JSON_M(c))

cudaError_t mandelbrotCuda(Options options, int maxIterations);
