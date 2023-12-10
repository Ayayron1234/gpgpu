#include "IO.h"
#include <fstream>

#include <SDL2/SDL.h>
#undef main

const wchar_t* GetWC(const char* c) {
	const size_t cSize = strlen(c) + 1;
	wchar_t* wc = new wchar_t[cSize];
	mbstowcs(wc, c, cSize);

	return wc;
}

namespace IO {

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
	std::wstring droppedFilePath;

	int windowWidth, windowHeight;

	OutputBuffer output;

	static SDL_Instance& instance();
};

void Render() {
	// Update SDL texture with the output buffer data and 
	// render the texture to the entire window
	SDL_UpdateTexture(SDL_Instance::instance().output.texture, nullptr, SDL_Instance::instance().output.buffer, sizeof(IO::RGB) * SDL_Instance::instance().windowWidth);
	SDL_RenderCopy(SDL_Instance::instance().renderer, SDL_Instance::instance().output.texture, nullptr, nullptr);

	// Render ImGui draw data
	ImGui::Render();
	ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData());

	// Present the rendered frame
	SDL_RenderPresent(SDL_Instance::instance().renderer);

	// Start a new frame for ImGui rendering
	ImGui_ImplSDLRenderer2_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();
	ImGui::DockSpaceOverViewport((const ImGuiViewport*)0, ImGuiDockNodeFlags_PassthruCentralNode);

	// Clear the renderer
	SDL_SetRenderDrawColor(SDL_Instance::instance().renderer, 0, 0, 0, 255);
	SDL_RenderClear(SDL_Instance::instance().renderer);
}

void OutputBuffer::resize(int width, int height) {
	delete[] buffer;
	SDL_DestroyTexture(texture);

	buffer = new RGB[width * height];
	texture = SDL_CreateTexture(SDL_Instance::instance().renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, width, height);
}

void IO::HandleEvents() {
	// Store previous mouse button state, reset scroll amount, and clear dropped file path
	SDL_Instance::instance().prevMouseButtons = SDL_Instance::instance().mouseButtons;
	SDL_Instance::instance().mouseScrollAmount = 0;
	SDL_Instance::instance().droppedFilePath.clear();

	// Check for window resize and update output buffer accordingly
	int newWidth, newHeight;
	SDL_GetWindowSize(SDL_Instance::instance().window, &newWidth, &newHeight);
	if (newWidth != SDL_Instance::instance().windowWidth || newHeight != SDL_Instance::instance().windowHeight) {
		SDL_Instance::instance().output.resize(newWidth, newHeight);

		SDL_Instance::instance().windowWidth = newWidth;
		SDL_Instance::instance().windowHeight = newHeight;
	}

	// Process SDL events
	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		ImGui_ImplSDL2_ProcessEvent(&event);

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
		case (SDL_DROPFILE): {
			const wchar_t* path = GetWC(event.drop.file);
			SDL_Instance::instance().droppedFilePath = path;
			delete[] path;
			break;
		}
		default:
			break;
		}
	}
}

void Quit() {
	SDL_DestroyRenderer(SDL_Instance::instance().renderer);
	SDL_DestroyWindow(SDL_Instance::instance().window);
	SDL_Quit();
}

void OpenWindow(int width, int height) {
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

bool IsButtonDown(uint8_t button) {
	button = (1u << button);
	return (SDL_Instance::instance().mouseButtons & button) == button;
}

bool MouseClicked(uint8_t button) {
	button = (1u << button);
	return ((SDL_Instance::instance().mouseButtons & button) == button) && !((SDL_Instance::instance().prevMouseButtons & button) == button);
}

bool MouseReleased(uint8_t button) {
	button = (1u << button);
	return !((SDL_Instance::instance().mouseButtons & button) == button) && ((SDL_Instance::instance().prevMouseButtons & button) == button);
}

vec2 NormalizePixel(int x, int y) {
	return { (2.f * x) / (Float)SDL_Instance::instance().windowWidth - 1.f, ((2.f * y) / (Float)SDL_Instance::instance().windowHeight - 1.f) };
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

int GetWindowWidth() {
	return SDL_Instance::instance().windowWidth;
}

int GetWindowHeight() {
	return SDL_Instance::instance().windowHeight;
}

bool FileDropped() {
	return SDL_Instance::instance().droppedFilePath.length() > 0;
}

const std::wstring& GetDroppedFilePath() {
	return SDL_Instance::instance().droppedFilePath;
}

void ResizeWindow(int width, int height) {
	SDL_Instance::instance().output.resize(width, height);
	SDL_SetWindowSize(SDL_Instance::instance().window, width, height);
}

inline SDL_Instance& SDL_Instance::instance() {
	static SDL_Instance c_instance;
	return c_instance;
}

}
