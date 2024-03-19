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

find_library(D3D12_LIBRARY d3d12.lib     PATHS     "$ENV{DXSDK_DIR}/Lib/x86" "C:/Program Files (x86)/Microsoft DirectX SDK/Lib/x86"     DOC "The directory where d3d12.lib resides")  
find_library(D3D12_LIBRARY2 dxcompiler.lib     PATHS     "$ENV{DXSDK_DIR}/Lib/x86" "C:/Program Files (x86)/Microsoft DirectX SDK/Lib/x86"     DOC "The directory where d3dx12.lib resides") 
find_library(D3D12_LIBRARY3 dxgi.lib     PATHS     "$ENV{DXSDK_DIR}/Lib/x86" "C:/Program Files (x86)/Microsoft DirectX SDK/Lib/x86"     DOC "The directory where d3dx12.lib resides") 
find_library(D3D12_LIBRARY4 d3dcompiler.lib     PATHS     "$ENV{DXSDK_DIR}/Lib/x86" "C:/Program Files (x86)/Microsoft DirectX SDK/Lib/x86"     DOC "The directory where d3dx12.lib resides") 

target_include_directories(PathTracerX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(PathTracerX PRIVATE
	vendor
	imgui
	imguilib
	${D3D12_LIBRARY}
	${D3D12_LIBRARY2}
	${D3D12_LIBRARY3}
	${D3D12_LIBRARY4}
)

target_link_libraries(imguilib PRIVATE 
	imgui
)
