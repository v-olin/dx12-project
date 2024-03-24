project(PathTracerX, VERSION 0.01)

# Create executable
add_executable(PathTracerX)
target_sources(PathTracerX PRIVATE 
	App.h
	App.cpp
	GUI.h
	GUI.cpp
	Mouse.h
	Model.h
	Mouse.cpp
	Window.h
	Window.cpp
	WinMain.cpp
	PathWin.h
	StringUtil.h
	Keyboard.cpp
	Keyboard.h
	Helper.h
	Renderer.h
	Renderer.cpp
	RendererSettings.h
	Scene.h
	Material.h
	Texture.h
	Camera.h
	Transform.h
	Transform.cpp
	Light.h
	Selectable.h
	GraphicsAPI.h
	DXRenderer.h
	DXRenderer.cpp
	Exceptions.h
	Exceptions.cpp
	Logger.h
	Logger.cpp
)

source_group("DX" FILES
	DXRenderer.h
	DXRenderer.cpp
)

source_group("Renderer" FILES
	GraphicsAPI.h
	Renderer.h
	Renderer.cpp
	RendererSettings.h
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS")

target_include_directories(PathTracerX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(PathTracerX PRIVATE
	vendor
	imgui
	imguilib
	d3d12.lib
	dxcompiler.lib
	dxgi.lib
	d3dcompiler.lib
)

target_link_libraries(imguilib PRIVATE 
	imgui
)
