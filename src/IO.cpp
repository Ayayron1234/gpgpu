#include "IO.h"
#include <fstream>

const wchar_t* GetWC(const char* c)
{
	const size_t cSize = strlen(c) + 1;
	wchar_t* wc = new wchar_t[cSize];
	mbstowcs(wc, c, cSize);

	return wc;
}

namespace IO {

void Render() {
	SDL_UpdateTexture(SDL_Instance::instance().output.texture, nullptr, SDL_Instance::instance().output.buffer, sizeof(IO::RGB) * SDL_Instance::instance().windowWidth);
	SDL_RenderCopy(SDL_Instance::instance().renderer, SDL_Instance::instance().output.texture, nullptr, nullptr);

	ImGui::Render();
	ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData());

	SDL_RenderPresent(SDL_Instance::instance().renderer);

	ImGui_ImplSDLRenderer2_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();
	ImGui::DockSpaceOverViewport((const ImGuiViewport*)0, ImGuiDockNodeFlags_PassthruCentralNode);

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
	SDL_Instance::instance().prevMouseButtons = SDL_Instance::instance().mouseButtons;
	SDL_Instance::instance().mouseScrollAmount = 0;
	SDL_Instance::instance().droppedFilePath.clear();

	int newWidth, newHeight;
	SDL_GetWindowSize(SDL_Instance::instance().window, &newWidth, &newHeight);
	if (newWidth != SDL_Instance::instance().windowWidth || newHeight != SDL_Instance::instance().windowHeight) {
		SDL_Instance::instance().output.resize(newWidth, newHeight);

		SDL_Instance::instance().windowWidth = newWidth;
		SDL_Instance::instance().windowHeight = newHeight;
	}

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

}
