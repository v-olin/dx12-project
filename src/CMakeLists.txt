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
	Model.cpp
	Vertex.h
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
	RenderSettings.h
	Scene.h
	Scene.cpp
	Model.h
	Model.cpp
	Material.h
	Texture.h
	Texture.cpp
	Camera.h
	Camera.cpp
	Transform.h
	Transform.cpp
	Light.h
	Selectable.h
	GraphicsAPI.h
	DXRenderer.h
	DXRenderer.cpp
	DXVertexBuffer.h
	DXVertexBuffer.cpp
	DXIndexBuffer.h
	DXIndexBuffer.cpp
	DXConstantBuffer.h
	DXConstantBuffer.cpp
	NVBLASGenerator.h
	NVBLASGenerator.cpp
	NVRaytracingPipelineGenerator.h
	NVRaytracingPipelineGenerator.cpp
	NVRootSignatureGenerator.h
	NVRootSignatureGenerator.cpp
	NVShaderBindingTableGenerator.h
	NVShaderBindingTableGenerator.cpp
	NVTLASGenerator.h
	NVTLASGenerator.cpp
	Exceptions.h
	Exceptions.cpp
	Serializer.h
	Serializer.cpp
	Serializable.h
	ResourceManager.h
	ResourceManager.cpp
	../vendor/d3dx12/d3dx12.h
	../vendor/ImGuizmo/ImGuizmo.h
	../vendor/ImGuizmo/ImGuizmo.cpp
	Logger.h
	Logger.cpp
	Event.h
	KeyEvent.h
	MouseEvent.h
	WindowEvent.h
	TimedEvent.h
	ProcedualWorldManager.h
	ProcedualWorldManager.cpp
	Noise.h
	Noise.cpp
	Culling.cpp
	Culling.h
	Pong.h
	Pong.cpp
)

source_group("DX" FILES
	DXRenderer.h
	DXRenderer.cpp
	DXVertexBuffer.h
	DXVertexBuffer.cpp
	DXIndexBuffer.h
	DXIndexBuffer.cpp
	DXConstantBuffer.h
	DXConstantBuffer.cpp
)

source_group("NV" FILES
	NVBLASGenerator.h
	NVBLASGenerator.cpp
	NVRaytracingPipelineGenerator.h
	NVRaytracingPipelineGenerator.cpp
	NVRootSignatureGenerator.h
	NVRootSignatureGenerator.cpp
	NVShaderBindingTableGenerator.h
	NVShaderBindingTableGenerator.cpp
	NVTLASGenerator.h
	NVTLASGenerator.cpp
)

source_group("Serializer" FILES
	Serializer.h
	Serializer.cpp
	Serializable.h
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
