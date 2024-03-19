add_subdirectory ( ${CMAKE_SOURCE_DIR}/vendor/imgui-cmake ${CMAKE_CURRENT_BINARY_DIR}/imgui )
set_target_properties(imgui PROPERTIES FOLDER external)

set(imgui_src 
	imgui-cmake/backends/imgui_impl_dx12.cpp
	imgui-cmake/backends/imgui_impl_dx12.h
	imgui-cmake/backends/imgui_impl_win32.cpp
	imgui-cmake/backends/imgui_impl_win32.h
)
add_library(imguilib ${imgui_src})
target_include_directories(imguilib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/imgui-cmake/backends)


add_library(vendor INTERFACE)
