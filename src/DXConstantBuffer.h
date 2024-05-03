#include "PathWin.h"
#include <d3d12.h>

namespace pathtracex {
	class DXConstantBuffer {
	public:
		DXConstantBuffer(size_t size);
		~DXConstantBuffer();

		ID3D12Resource* constantBuffer;
		size_t bufferSize;
	private:
		void allocateBuffer();
	};
}