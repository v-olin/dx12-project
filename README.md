# Project in DirectX 12 for the Chalmers course DAT205 Advanced Computer Graphics

## Build
To build the project, generate solution files in the `/build` folder with CMake and build it with Visual Studio.
```
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
```

## Post-build steps
**Note:** This project only compiles in debug mode for unknown reasons.

### Raytracing setup
- Nvidia RTX cards
  - You will need precompiled copies of `dxcompiler.dll` and `dxil.dll` ([can be downloaded from here](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.8.2405)) in the same folder as the project executable. This project uses the DXR for raytracing and the shader libraries are compiled with these compilers. These compilers probably already exist in a Windows 10+ machine but the location or version is unknown. If you wish not to download the compilers or try the DXR features, you can comment out the lines `renderer->InitRaytracingPipeline(...)` in `src/App.cpp`.
- Nvidia GTX cards
    - Raytracing is not supported so the pipeline will never be created. You should still be able to run the project without downloading extra shader compilers but only test rasterization-based rendering. This is also untested behavior.
- Any AMD card
  - This territory is unknown as none of us developers used an AMD card during development.

### Scene configuration
This project uses yaml-files for scene loading where you can manually add, edit, or remove objects in the scene. Scene files lie in the `scenes/` directory.

To run the project you also need a configuration file for the project. Duplicate `example_config.yaml` as `config.yaml` to configure app settings like resolution and the scene to load.

If you want to try the DXR pong game, set the scene name in `config.yaml` to `pong`. Setting it to `Pong` will load the game scene and not the game itself.

## Project limitations
These are some of the limitations of the project that we did not have the time or motivation to fix
- Unable to resize window after initial size (buffers are not resized)
- Cannot add models to the scene through the menu when rendering with DXR (acceleration structures are not rebuilt)
- App freezes for a short while when scrolling the mouse wheel down (bad event handling)
- Cannot have more than 3 lights in scene when using DXR (buffer too small)

## Features

### Rasterization features
- Procedural terrain generation
- Chunked terrain loading
- Texture blending of procedural terrain
- Normal, shininess, and texture maps

### Raytracing features
- Ambient occlusion[^buggy]
- Randomness compute shader for DXR sampling[^buggy]
- Transparent and reflective material and objects[^prim][^npong]
- TAA compute shader with depth projection[^npong]
  - Custom DXR depth buffer
- Motion blur compute shader[^buggy][^npong]
- Accurately rendered material using raytracing[^pong][^buggy]
- Bloom compute shader[^buggy]

[^buggy]: A bit buggy implementation
[^prim]: Only works on primitive models
[^npong]: Not applicable in the Pong game
[^pong]: Only applicable in the Pong game
