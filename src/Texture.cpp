#include "Texture.h"
#include "Model.h"
#include "PathWin.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include "DXRenderer.h"


namespace pathtracex {
	void Texture::free()
	{
	}

	bool Texture::load(const std::string& _directory, const std::string& _filename, int n)
	{
		filename = file_util::normalise(_filename);
		directory = file_util::normalise(_directory);
		//TODO figure out how to load textures with directX

		/*
		in order to make all the calls using the commandList and the device 
		maybe there should be a function in the renderer that can create textureBuffer
		so that we can with every mesh change to the correct texture.
		This can alos be usefull when loading objects in runtime, since then the model
		itself can make sure that the trextures are created when its loaded!
		Also we need to send a bool saying if the current mesh even has a texture, since the tutorial 
		assumes that every vertexBuffer being drawn has a texture.
		*/

		DXRenderer* renderer = DXRenderer::getInstance();
		ID3D12Resource* textureBufferUploadHeap;

		renderer->resetCommandList();
		HRESULT hr;



		// load the image, create a texture resource and descriptor heap

		   // Load the image from file
		D3D12_RESOURCE_DESC textureDesc;
		int imageBytesPerRow;
		BYTE* imageData;
		// Initializing an object of wstring
		std::string fullFileName = directory + filename;
		std::wstring temp = std::wstring(fullFileName.begin(), fullFileName.end());

		// Applying c_str() method on temp
		LPCWSTR wideString = temp.c_str();
		int imageSize = LoadImageDataFromFile(&imageData, textureDesc, wideString, imageBytesPerRow);

		// make sure we have data
		if (imageSize <= 0)
		{
			return false;
		}

		// create a default heap where the upload heap will copy its contents into (contents being the texture)
		CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);
		hr = renderer->device->CreateCommittedResource(
			&heapProperties, // a default heap
			D3D12_HEAP_FLAG_NONE, // no flags
			&textureDesc, // the description of our texture
			D3D12_RESOURCE_STATE_COPY_DEST, // We will copy the texture from the upload heap to here, so we start it out in a copy dest state
			nullptr, // used for render targets and depth/stencil buffers
			IID_PPV_ARGS(&textureBuffer));
		if (FAILED(hr))
		{
			auto reason = renderer->device->GetDeviceRemovedReason();
			return false;
		}
		textureBuffer->SetName(L"Texture Buffer Resource Heap");
		UINT64 textureUploadBufferSize;
		// this function gets the size an upload buffer needs to be to upload a texture to the gpu.
		// each row must be 256 byte aligned except for the last row, which can just be the size in bytes of the row
		// eg. textureUploadBufferSize = ((((width * numBytesPerPixel) + 255) & ~255) * (height - 1)) + (width * numBytesPerPixel);
		//textureUploadBufferSize = (((imageBytesPerRow + 255) & ~255) * (textureDesc.Height - 1)) + imageBytesPerRow;
		renderer->device->GetCopyableFootprints(&textureDesc, 0, 1, 0, nullptr, nullptr, nullptr, &textureUploadBufferSize);

		// now we create an upload heap to upload our texture to the GPU
		CD3DX12_HEAP_PROPERTIES heapPropertiesUpload(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC bufferResourceDescUpload = CD3DX12_RESOURCE_DESC::Buffer(textureUploadBufferSize);
		hr = renderer->device->CreateCommittedResource(
			&heapPropertiesUpload, // upload heap
			D3D12_HEAP_FLAG_NONE, // no flags
			&bufferResourceDescUpload, // resource description for a buffer (storing the image data in this heap just to copy to the default heap)
			D3D12_RESOURCE_STATE_GENERIC_READ, // We will copy the contents from this heap to the default heap above
			nullptr,
			IID_PPV_ARGS(&textureBufferUploadHeap));
		if (FAILED(hr))
		{
			return false;
		}
		textureBufferUploadHeap->SetName(L"Texture Buffer Upload Resource Heap");

		// store vertex buffer in upload heap
		D3D12_SUBRESOURCE_DATA textureData = {};
		textureData.pData = &imageData[0]; // pointer to our image data
		textureData.RowPitch = imageBytesPerRow; // size of all our triangle vertex data
		textureData.SlicePitch = imageBytesPerRow * textureDesc.Height; // also the size of our triangle vertex data

		// Now we copy the upload buffer contents to the default heap
		UpdateSubresources(renderer->commandList, textureBuffer, textureBufferUploadHeap, 0, 0, 1, &textureData);

		// transition the texture default heap to a pixel shader resource (we will be sampling from this heap in the pixel shader to get the color of pixels)
		CD3DX12_RESOURCE_BARRIER textureBufferResourceBarrier = CD3DX12_RESOURCE_BARRIER::Transition(textureBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		renderer->commandList->ResourceBarrier(1, &textureBufferResourceBarrier);

		// create the descriptor heap that will store our srv
		D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
		heapDesc.NumDescriptors = 1;
		heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		hr = renderer->device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mainDescriptorHeap));
		if (FAILED(hr))
		{
			return false;
		}
		// now we create a shader resource view (descriptor that points to the texture and describes it)
		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Format = textureDesc.Format;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = 1;
		renderer->device->CreateShaderResourceView(textureBuffer, &srvDesc, mainDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

		renderer->finishedRecordingCommandList();

		renderer->executeCommandList();
		// We make sure the index buffer is uploaded to the GPU before the renderer uses it
		renderer->incrementFenceAndSignalCurrentFrame();

		// we are done with image data now that we've uploaded it to the gpu, so free it up
		delete imageData;


		valid = true;
		return true;
	}
	// get the dxgi format equivilent of a wic format
	DXGI_FORMAT Texture::GetDXGIFormatFromWICFormat(WICPixelFormatGUID& wicFormatGUID)
	{
		if (wicFormatGUID == GUID_WICPixelFormat128bppRGBAFloat) return DXGI_FORMAT_R32G32B32A32_FLOAT;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppRGBAHalf) return DXGI_FORMAT_R16G16B16A16_FLOAT;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppRGBA) return DXGI_FORMAT_R16G16B16A16_UNORM;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppRGBA) return DXGI_FORMAT_R8G8B8A8_UNORM;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppBGRA) return DXGI_FORMAT_B8G8R8A8_UNORM;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppBGR) return DXGI_FORMAT_B8G8R8X8_UNORM;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppRGBA1010102XR) return DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM;

		else if (wicFormatGUID == GUID_WICPixelFormat32bppRGBA1010102) return DXGI_FORMAT_R10G10B10A2_UNORM;
		else if (wicFormatGUID == GUID_WICPixelFormat16bppBGRA5551) return DXGI_FORMAT_B5G5R5A1_UNORM;
		else if (wicFormatGUID == GUID_WICPixelFormat16bppBGR565) return DXGI_FORMAT_B5G6R5_UNORM;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppGrayFloat) return DXGI_FORMAT_R32_FLOAT;
		else if (wicFormatGUID == GUID_WICPixelFormat16bppGrayHalf) return DXGI_FORMAT_R16_FLOAT;
		else if (wicFormatGUID == GUID_WICPixelFormat16bppGray) return DXGI_FORMAT_R16_UNORM;
		else if (wicFormatGUID == GUID_WICPixelFormat8bppGray) return DXGI_FORMAT_R8_UNORM;
		else if (wicFormatGUID == GUID_WICPixelFormat8bppAlpha) return DXGI_FORMAT_A8_UNORM;

		else return DXGI_FORMAT_UNKNOWN;
	}
	// get a dxgi compatible wic format from another wic format
	WICPixelFormatGUID Texture::GetConvertToWICFormat(WICPixelFormatGUID& wicFormatGUID)
	{
		if (wicFormatGUID == GUID_WICPixelFormatBlackWhite) return GUID_WICPixelFormat8bppGray;
		else if (wicFormatGUID == GUID_WICPixelFormat1bppIndexed) return GUID_WICPixelFormat32bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat2bppIndexed) return GUID_WICPixelFormat32bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat4bppIndexed) return GUID_WICPixelFormat32bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat8bppIndexed) return GUID_WICPixelFormat32bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat2bppGray) return GUID_WICPixelFormat8bppGray;
		else if (wicFormatGUID == GUID_WICPixelFormat4bppGray) return GUID_WICPixelFormat8bppGray;
		else if (wicFormatGUID == GUID_WICPixelFormat16bppGrayFixedPoint) return GUID_WICPixelFormat16bppGrayHalf;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppGrayFixedPoint) return GUID_WICPixelFormat32bppGrayFloat;
		else if (wicFormatGUID == GUID_WICPixelFormat16bppBGR555) return GUID_WICPixelFormat16bppBGRA5551;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppBGR101010) return GUID_WICPixelFormat32bppRGBA1010102;
		else if (wicFormatGUID == GUID_WICPixelFormat24bppBGR) return GUID_WICPixelFormat32bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat24bppRGB) return GUID_WICPixelFormat32bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppPBGRA) return GUID_WICPixelFormat32bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppPRGBA) return GUID_WICPixelFormat32bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat48bppRGB) return GUID_WICPixelFormat64bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat48bppBGR) return GUID_WICPixelFormat64bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppBGRA) return GUID_WICPixelFormat64bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppPRGBA) return GUID_WICPixelFormat64bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppPBGRA) return GUID_WICPixelFormat64bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat48bppRGBFixedPoint) return GUID_WICPixelFormat64bppRGBAHalf;
		else if (wicFormatGUID == GUID_WICPixelFormat48bppBGRFixedPoint) return GUID_WICPixelFormat64bppRGBAHalf;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppRGBAFixedPoint) return GUID_WICPixelFormat64bppRGBAHalf;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppBGRAFixedPoint) return GUID_WICPixelFormat64bppRGBAHalf;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppRGBFixedPoint) return GUID_WICPixelFormat64bppRGBAHalf;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppRGBHalf) return GUID_WICPixelFormat64bppRGBAHalf;
		else if (wicFormatGUID == GUID_WICPixelFormat48bppRGBHalf) return GUID_WICPixelFormat64bppRGBAHalf;
		else if (wicFormatGUID == GUID_WICPixelFormat128bppPRGBAFloat) return GUID_WICPixelFormat128bppRGBAFloat;
		else if (wicFormatGUID == GUID_WICPixelFormat128bppRGBFloat) return GUID_WICPixelFormat128bppRGBAFloat;
		else if (wicFormatGUID == GUID_WICPixelFormat128bppRGBAFixedPoint) return GUID_WICPixelFormat128bppRGBAFloat;
		else if (wicFormatGUID == GUID_WICPixelFormat128bppRGBFixedPoint) return GUID_WICPixelFormat128bppRGBAFloat;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppRGBE) return GUID_WICPixelFormat128bppRGBAFloat;
		else if (wicFormatGUID == GUID_WICPixelFormat32bppCMYK) return GUID_WICPixelFormat32bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppCMYK) return GUID_WICPixelFormat64bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat40bppCMYKAlpha) return GUID_WICPixelFormat64bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat80bppCMYKAlpha) return GUID_WICPixelFormat64bppRGBA;

#if (_WIN32_WINNT >= _WIN32_WINNT_WIN8) || defined(_WIN7_PLATFORM_UPDATE)
		else if (wicFormatGUID == GUID_WICPixelFormat32bppRGB) return GUID_WICPixelFormat32bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppRGB) return GUID_WICPixelFormat64bppRGBA;
		else if (wicFormatGUID == GUID_WICPixelFormat64bppPRGBAHalf) return GUID_WICPixelFormat64bppRGBAHalf;
#endif

		else return GUID_WICPixelFormatDontCare;
	}
	// get the number of bits per pixel for a dxgi format
	int Texture::GetDXGIFormatBitsPerPixel(DXGI_FORMAT& dxgiFormat)
	{
		if (dxgiFormat == DXGI_FORMAT_R32G32B32A32_FLOAT) return 128;
		else if (dxgiFormat == DXGI_FORMAT_R16G16B16A16_FLOAT) return 64;
		else if (dxgiFormat == DXGI_FORMAT_R16G16B16A16_UNORM) return 64;
		else if (dxgiFormat == DXGI_FORMAT_R8G8B8A8_UNORM) return 32;
		else if (dxgiFormat == DXGI_FORMAT_B8G8R8A8_UNORM) return 32;
		else if (dxgiFormat == DXGI_FORMAT_B8G8R8X8_UNORM) return 32;
		else if (dxgiFormat == DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM) return 32;

		else if (dxgiFormat == DXGI_FORMAT_R10G10B10A2_UNORM) return 32;
		else if (dxgiFormat == DXGI_FORMAT_B5G5R5A1_UNORM) return 16;
		else if (dxgiFormat == DXGI_FORMAT_B5G6R5_UNORM) return 16;
		else if (dxgiFormat == DXGI_FORMAT_R32_FLOAT) return 32;
		else if (dxgiFormat == DXGI_FORMAT_R16_FLOAT) return 16;
		else if (dxgiFormat == DXGI_FORMAT_R16_UNORM) return 16;
		else if (dxgiFormat == DXGI_FORMAT_R8_UNORM) return 8;
		else if (dxgiFormat == DXGI_FORMAT_A8_UNORM) return 8;
	}
	// load and decode image from file
	int Texture::LoadImageDataFromFile(BYTE** imageData, D3D12_RESOURCE_DESC& resourceDescription, LPCWSTR filename, int& bytesPerRow)
	{
		HRESULT hr;

		// we only need one instance of the imaging factory to create decoders and frames
		static IWICImagingFactory* wicFactory;

		// reset decoder, frame and converter since these will be different for each image we load
		IWICBitmapDecoder* wicDecoder = NULL;
		IWICBitmapFrameDecode* wicFrame = NULL;
		IWICFormatConverter* wicConverter = NULL;

		bool imageConverted = false;

		if (wicFactory == NULL)
		{
			// Initialize the COM library
			CoInitialize(NULL);

			// create the WIC factory
			hr = CoCreateInstance(
				CLSID_WICImagingFactory,
				NULL,
				CLSCTX_INPROC_SERVER,
				IID_PPV_ARGS(&wicFactory)
			);
			if (FAILED(hr)) return 0;
		}

		// load a decoder for the image
		hr = wicFactory->CreateDecoderFromFilename(
			filename,                        // Image we want to load in
			NULL,                            // This is a vendor ID, we do not prefer a specific one so set to null
			GENERIC_READ,                    // We want to read from this file
			WICDecodeMetadataCacheOnLoad,    // We will cache the metadata right away, rather than when needed, which might be unknown
			&wicDecoder                      // the wic decoder to be created
		);
		if (FAILED(hr)) return 0;

		// get image from decoder (this will decode the "frame")
		hr = wicDecoder->GetFrame(0, &wicFrame);
		if (FAILED(hr)) return 0;

		// get wic pixel format of image
		WICPixelFormatGUID pixelFormat;
		hr = wicFrame->GetPixelFormat(&pixelFormat);
		if (FAILED(hr)) return 0;

		// get size of image
		UINT textureWidth, textureHeight;
		hr = wicFrame->GetSize(&textureWidth, &textureHeight);
		if (FAILED(hr)) return 0;

		// we are not handling sRGB types in this tutorial, so if you need that support, you'll have to figure
		// out how to implement the support yourself

		// convert wic pixel format to dxgi pixel format
		DXGI_FORMAT dxgiFormat = GetDXGIFormatFromWICFormat(pixelFormat);

		// if the format of the image is not a supported dxgi format, try to convert it
		if (dxgiFormat == DXGI_FORMAT_UNKNOWN)
		{
			// get a dxgi compatible wic format from the current image format
			WICPixelFormatGUID convertToPixelFormat = GetConvertToWICFormat(pixelFormat);

			// return if no dxgi compatible format was found
			if (convertToPixelFormat == GUID_WICPixelFormatDontCare) return 0;

			// set the dxgi format
			dxgiFormat = GetDXGIFormatFromWICFormat(convertToPixelFormat);

			// create the format converter
			hr = wicFactory->CreateFormatConverter(&wicConverter);
			if (FAILED(hr)) return 0;

			// make sure we can convert to the dxgi compatible format
			BOOL canConvert = FALSE;
			hr = wicConverter->CanConvert(pixelFormat, convertToPixelFormat, &canConvert);
			if (FAILED(hr) || !canConvert) return 0;

			// do the conversion (wicConverter will contain the converted image)
			hr = wicConverter->Initialize(wicFrame, convertToPixelFormat, WICBitmapDitherTypeErrorDiffusion, 0, 0, WICBitmapPaletteTypeCustom);
			if (FAILED(hr)) return 0;

			// this is so we know to get the image data from the wicConverter (otherwise we will get from wicFrame)
			imageConverted = true;
		}

		int bitsPerPixel = GetDXGIFormatBitsPerPixel(dxgiFormat); // number of bits per pixel
		bytesPerRow = (textureWidth * bitsPerPixel) / 8; // number of bytes in each row of the image data
		int imageSize = bytesPerRow * textureHeight; // total image size in bytes

		// allocate enough memory for the raw image data, and set imageData to point to that memory
		*imageData = (BYTE*)malloc(imageSize);

		// copy (decoded) raw image data into the newly allocated memory (imageData)
		if (imageConverted)
		{
			// if image format needed to be converted, the wic converter will contain the converted image
			hr = wicConverter->CopyPixels(0, bytesPerRow, imageSize, *imageData);
			if (FAILED(hr)) return 0;
		}
		else
		{
			// no need to convert, just copy data from the wic frame
			hr = wicFrame->CopyPixels(0, bytesPerRow, imageSize, *imageData);
			if (FAILED(hr)) return 0;
		}

		// now describe the texture with the information we have obtained from the image
		resourceDescription = {};
		resourceDescription.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		resourceDescription.Alignment = 0; // may be 0, 4KB, 64KB, or 4MB. 0 will let runtime decide between 64KB and 4MB (4MB for multi-sampled textures)
		resourceDescription.Width = textureWidth; // width of the texture
		resourceDescription.Height = textureHeight; // height of the texture
		resourceDescription.DepthOrArraySize = 1; // if 3d image, depth of 3d image. Otherwise an array of 1D or 2D textures (we only have one image, so we set 1)
		resourceDescription.MipLevels = 1; // Number of mipmaps. We are not generating mipmaps for this texture, so we have only one level
		resourceDescription.Format = dxgiFormat; // This is the dxgi format of the image (format of the pixels)
		resourceDescription.SampleDesc.Count = 1; // This is the number of samples per pixel, we just want 1 sample
		resourceDescription.SampleDesc.Quality = 0; // The quality level of the samples. Higher is better quality, but worse performance
		resourceDescription.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; // The arrangement of the pixels. Setting to unknown lets the driver choose the most efficient one
		resourceDescription.Flags = D3D12_RESOURCE_FLAG_NONE; // no flags

		// return the size of the image. remember to delete the image once your done with it (in this tutorial once its uploaded to the gpu)
		return imageSize;
	}



}