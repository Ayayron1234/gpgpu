#pragma once
#include "vec2.h"

#include <SDL.h>
#undef main

#include <iostream>

static int g_windowWidth;
static int g_windowHeight;

namespace IO {

struct RGB {
	char r = 0;
	char g = 0;
	char b = 0;
	char a = 0;
};

struct Camera {
	vec2 position;
	Float zoom;
};

struct OutputBuffer {
	RGB* buffer = new RGB[800 * 800];
	SDL_Texture* texture;
};

struct SDL_Instance {
	SDL_Renderer* renderer = nullptr;
	SDL_Window* window = nullptr;

	Uint8 mouseButtons = 0;
	Uint8 prevMouseButtons = 0;
	float mouseScrollAmount = 0;

	OutputBuffer output;
	
	static SDL_Instance& instance() {
		static SDL_Instance c_instance;
		return c_instance;
	}
};

void OpenWindow();

void Render();

void Quit() {
	SDL_DestroyRenderer(SDL_Instance::instance().renderer);
	SDL_DestroyWindow(SDL_Instance::instance().window);
	SDL_Quit();
}

void OpenWindow(int width, int height) {
	g_windowWidth = width;
	g_windowHeight = height;

	SDL_Init(SDL_INIT_EVERYTHING);

	SDL_Instance::instance().window = SDL_CreateWindow("GPGPU", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, g_windowWidth, g_windowHeight, SDL_WINDOW_RESIZABLE);
	SDL_Instance::instance().renderer = SDL_CreateRenderer(SDL_Instance::instance().window, -1, SDL_RENDERER_ACCELERATED);

	SDL_Instance::instance().output.texture = SDL_CreateTexture(SDL_Instance::instance().renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, g_windowWidth, g_windowHeight);
}

void Render() {
	SDL_UpdateTexture(SDL_Instance::instance().output.texture, nullptr, SDL_Instance::instance().output.buffer, sizeof(IO::RGB) * g_windowWidth);
	SDL_RenderCopy(SDL_Instance::instance().renderer, SDL_Instance::instance().output.texture, nullptr, nullptr);

	SDL_RenderPresent(SDL_Instance::instance().renderer);

	SDL_SetRenderDrawColor(SDL_Instance::instance().renderer, 0, 0, 0, 255);
	SDL_RenderClear(SDL_Instance::instance().renderer);
}

void HandleEvents() {
	SDL_Instance::instance().prevMouseButtons = SDL_Instance::instance().mouseButtons;
	SDL_Instance::instance().mouseScrollAmount = 0;
	
	int newWidth, newHeight;
	SDL_GetWindowSize(SDL_Instance::instance().window, &newWidth, &newHeight);
	if (newWidth != g_windowWidth || newHeight != g_windowHeight) {
		delete[] SDL_Instance::instance().output.buffer;
		SDL_Instance::instance().output.buffer = new RGB[newWidth * newHeight];
		
		g_windowWidth = newWidth;
		g_windowHeight = newHeight;

		SDL_DestroyTexture(SDL_Instance::instance().output.texture);
		SDL_Instance::instance().output.texture = SDL_CreateTexture(SDL_Instance::instance().renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, g_windowWidth, g_windowHeight);
	}

	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		switch (event.type)
		{
		case SDL_MOUSEBUTTONDOWN: {
			SDL_Instance::instance().mouseButtons |= (1u << event.button.button);
		} break;
		case SDL_MOUSEBUTTONUP: {
			SDL_Instance::instance().mouseButtons &= ~(1u << event.button.button);
		} break;
		case SDL_MOUSEWHEEL:
			SDL_Instance::instance().mouseScrollAmount = event.wheel.preciseY;
			break;
		default:
			break;
		}
	}
}

bool IsButtonDown(Uint8 button) {
	button = (1u << button);
	return (SDL_Instance::instance().mouseButtons & button) == button;
}

bool MouseClicked(Uint8 button) {
	button = (1u << button);
	return ((SDL_Instance::instance().mouseButtons & button) == button) && !((SDL_Instance::instance().prevMouseButtons & button) == button);
}

bool MouseReleased(Uint8 button) {
	button = (1u << button);
	return !((SDL_Instance::instance().mouseButtons & button) == button) && ((SDL_Instance::instance().prevMouseButtons & button) == button);
}

vec2 NormalizePixel(int x, int y) {
	return { (2.f * x) / (Float)g_windowWidth - 1.f, ((2.f * y) / (Float)g_windowHeight - 1.f)};
}

vec2 GetMousePos() {
	int x, y;
	SDL_GetMouseState(&x, &y);
	return vec2(x, y);
}

float GetMouseWheel() {
	return SDL_Instance::instance().mouseScrollAmount;
}

RGB* GetOutputBuffer() {
	return SDL_Instance::instance().output.buffer;
}

}
