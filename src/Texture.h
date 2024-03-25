#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <memory>

#include "Helper.h"
namespace pathtracex {
	class Texture {
	public:
		bool valid = false;
		uint32_t gl_id = 0;
		uint32_t gl_id_internal = 0;
		std::string filename;
		std::string directory;
		int width, height;
		uint8_t* data;
		uint8_t n_components = 4;

		bool load(const std::string& directory, const std::string& filename, int nof_component);
		float4 sample(float2 uv) const;
		void free();
	};
}