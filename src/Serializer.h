#pragma once
#include "Scene.h"
#include "yaml-cpp/yaml.h"

namespace pathtracex {
	class Serializer {
	public:
		void serializeScene(Scene& scene);

	private:
		void createYAMLFile(const std::string& fileFolder, const std::string& fileName);

		void serializeModels(Scene& scene, YAML::Emitter& out);

		void serializeSerializable(Serializable* serializable, YAML::Emitter& out);

		void deserializeScene(const std::string& sceneName, Scene& scene);
		void deserializeModels(YAML::Node node, Scene& scene);
	};
}