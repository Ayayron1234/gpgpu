#include "fractals.h"

#include "utils/ExternalResource.h"
#include "IO.h"

auto& g_options = external_resource<"options.json", Json::wrap<Options>>::value;

bool g_doShowPropertiesWindow = true;
bool g_propertiesWindowHovered = false;
void ShowPropertiesWindow() {
    if (!g_doShowPropertiesWindow) return;

    bool isWindowOpen = true;
    ImGui::Begin("Properties", &isWindowOpen, ImGuiWindowFlags_NoResize);

    g_propertiesWindowHovered = ImGui::IsWindowHovered();
    ImGui::SetWindowSize({ 256 , 216 });

    ImGui::SeparatorText("Type:");
    ImGui::RadioButton("Mandelbrot", (int*)&g_options.type, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Julia", (int*)&g_options.type, 1);

    ImGui::SeparatorText("Max iterations:");
    ImGui::Text("Maximum iterations at zoom: 1.0");
    ImGui::DragInt("##maxIterations", &g_options.baseIterations, 1, 25, 400);

    ImGui::SeparatorText("Iteration increase falloff:");
    ImGui::DragFloat("##iterationIncreaseFalloff", &g_options.iterationIncreaseFallOff, 0.02f, 2.f, 20.f);

    ImGui::End();
}

int main() {
    IO::OpenWindow(g_options.windowWidth, g_options.windowHeight);

    std::cout << "f" << std::endl;

    vec2 dragStart; // normalized
    while (!SDL_QuitRequested()) {
        // Handle events
        IO::HandleEvents();
        g_options.windowWidth = IO::GetWindowWidth();
        g_options.windowHeight = IO::GetWindowHeight();
        vec2 normalizedMousePos = IO::NormalizePixel(IO::GetMousePos().x, IO::GetMousePos().y);

        // Draw GUI
        ShowPropertiesWindow();

        // Reset camera when space is pressed
        const Uint8* state = SDL_GetKeyboardState(nullptr);
        if (state[SDL_SCANCODE_SPACE]) {
            g_options.SetProperty({ 0, 0 });
            g_options.camera.position = { 0, 0 };
            g_options.camera.zoom = 0.2;
        }

        static vec2 z0Start;
        static bool dragDisabled = false;
        if (IO::MouseClicked(SDL_BUTTON_LEFT)) {
            dragDisabled = g_propertiesWindowHovered;
            z0Start = g_options.GetProperty();
            dragStart = normalizedMousePos;
        }
        if (IO::IsButtonDown(SDL_BUTTON_LEFT) && !dragDisabled) {
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


