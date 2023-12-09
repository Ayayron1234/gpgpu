#pragma once
#include "vec2.h"

#include <SDL.h>
#undef main

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_sdl2.h"
#include "imgui/backends/imgui_impl_sdlrenderer2.h"
#include "imgui/imgui_stdlib.h"

#include <iostream>

namespace IO {

struct RGB {
	char r = 0;
	char g = 0;
	char b = 0;
	char a = 0;
}; 

struct OutputBuffer {
	RGB* buffer = nullptr;
	SDL_Texture* texture;

	void resize(int width, int height);
};

struct SDL_Instance {
	SDL_Renderer* renderer = nullptr;
	SDL_Window* window = nullptr;

	Uint8 mouseButtons = 0;
	Uint8 prevMouseButtons = 0;
	float mouseScrollAmount = 0;

	int windowWidth, windowHeight;

	OutputBuffer output;
	
	static SDL_Instance& instance() {
		static SDL_Instance c_instance;
		return c_instance;
	}
};

void OpenWindow();

void Render();

void HandleEvents();

void Quit();

inline void OpenWindow(int width, int height) {
	SDL_Instance::instance().windowWidth = width;
	SDL_Instance::instance().windowHeight = height;

	// Init SDL and open a window
	SDL_Init(SDL_INIT_EVERYTHING);

	SDL_Instance::instance().window = SDL_CreateWindow("GPGPU", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SDL_Instance::instance().windowWidth, SDL_Instance::instance().windowHeight, SDL_WINDOW_RESIZABLE);
	SDL_Instance::instance().renderer = SDL_CreateRenderer(SDL_Instance::instance().window, -1, SDL_RENDERER_ACCELERATED);

	// Init ImGUI
	ImGui::CreateContext();

	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	//ImGui::StyleColorsLight();
	//ImGui::StyleColorsDark();
	ImGui::StyleColorsClassic();

	ImGui_ImplSDL2_InitForSDLRenderer(SDL_Instance::instance().window, SDL_Instance::instance().renderer);
	ImGui_ImplSDLRenderer2_Init(SDL_Instance::instance().renderer);

	// Init output buffer
	SDL_Instance::instance().output.texture = SDL_CreateTexture(SDL_Instance::instance().renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, SDL_Instance::instance().windowWidth, SDL_Instance::instance().windowHeight);
	SDL_Instance::instance().output.buffer = new RGB[width * height];

	// Start first frame
	ImGui_ImplSDLRenderer2_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();
	ImGui::DockSpaceOverViewport((const ImGuiViewport*)0, ImGuiDockNodeFlags_PassthruCentralNode);
}

inline bool IsButtonDown(Uint8 button) {
	button = (1u << button);
	return (SDL_Instance::instance().mouseButtons & button) == button;
}

inline bool MouseClicked(Uint8 button) {
	button = (1u << button);
	return ((SDL_Instance::instance().mouseButtons & button) == button) && !((SDL_Instance::instance().prevMouseButtons & button) == button);
}

inline bool MouseReleased(Uint8 button) {
	button = (1u << button);
	return !((SDL_Instance::instance().mouseButtons & button) == button) && ((SDL_Instance::instance().prevMouseButtons & button) == button);
}

inline vec2 NormalizePixel(int x, int y) {
	return { (2.f * x) / (Float)SDL_Instance::instance().windowWidth - 1.f, ((2.f * y) / (Float)SDL_Instance::instance().windowHeight - 1.f)};
}

inline vec2 GetMousePos() {
	int x, y;
	SDL_GetMouseState(&x, &y);
	return vec2(x, y);
}

inline float GetMouseWheel() {
	return SDL_Instance::instance().mouseScrollAmount;
}

inline RGB* GetOutputBuffer() {
	return SDL_Instance::instance().output.buffer;
}

inline int GetWindowWidth() {
	return SDL_Instance::instance().windowWidth;
}

inline int GetWindowHeight() {
	return SDL_Instance::instance().windowHeight;
}

}
