project(PathTracerX, VERSION 0.01)

# Create executable
add_executable(PathTracerX)
target_sources(PathTracerX PRIVATE 
	App.h
	App.cpp
	ImguiManager.h
	ImguiManager.cpp
	Mouse.h
	Mouse.cpp
	Renderer.h
	Renderer.cpp
	Window.h
	Window.cpp
	WinMain.cpp
	PathWin.h
	StringUtil.h
	Keyboard.cpp
	Keyboard.h
	Helper.h
)

source_group("GUI" FILES

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
