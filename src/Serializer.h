#pragma once
#include "Scene.h"

namespace pathtracex {
	class Serializer {
	public:
		void serializeScene(Scene& scene);

	private:
		void createYAMLFile(const std::string& fileFolder, const std::string& fileName);
	};
}