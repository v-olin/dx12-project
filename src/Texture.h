#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <memory>
#include "PathWin.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include "Helper.h"

#include <string>
#include <vector>
#include <wrl.h>

#include <wincodec.h>
#include "Helper.h"
#include "../vendor/d3dx12/d3dx12.h"
namespace pathtracex {
	enum TextureType
		{
			COLTEX,
			NORMALTEX,
			NUMTEXTURETYPES
		};
	class Texture {
	public:
	
		std::string filename;
		std::string directory;
		bool valid{ false };
		ID3D12Resource* textureBuffer; // the resource heap containing our texture

		bool load(const std::string& directory, const std::string& filename, int n, ID3D12DescriptorHeap** descriptorHeap, TextureType texType);
		void free();
	private:
		DXGI_FORMAT GetDXGIFormatFromWICFormat(WICPixelFormatGUID& wicFormatGUID);
		WICPixelFormatGUID GetConvertToWICFormat(WICPixelFormatGUID& wicFormatGUID);
		int GetDXGIFormatBitsPerPixel(DXGI_FORMAT& dxgiFormat);
		int LoadImageDataFromFile(BYTE** imageData, D3D12_RESOURCE_DESC& resourceDescription, LPCWSTR filename, int& bytesPerRow);



	};
}