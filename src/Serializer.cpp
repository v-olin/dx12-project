#include "Serializer.h"
#include <fstream>

namespace pathtracex {
#define SCENE_PATH "../../scenes/"
#define SCENE_FILE_EXTENSION ".yaml"

	void Serializer::serializeScene(Scene& scene)
	{
		// Create yaml file
		createYAMLFile(SCENE_PATH, scene.sceneName);
	}

	void Serializer::createYAMLFile(const std::string& fileFolder, const std::string& fileName)
	{
		std::string filePath = fileFolder + fileName + SCENE_FILE_EXTENSION;
		std::ofstream outfile(filePath);
		outfile.close();
	}
}
