#pragma once
#include "Scene.h"
#include "yaml-cpp/yaml.h"
#include "App.h"

namespace pathtracex {
	class Serializer {
	public:
		static void serializeScene(Scene& scene);
		static void deserializeScene(const std::string& sceneName, Scene& scene);

		static void serializeConfig(AppConfig& config);
		static AppConfig deserializeConfig();
	private:
		static void createYAMLFile(const std::string fileFolder, const std::string fileName);

		static void serializeModels(Scene& scene, YAML::Emitter& out);
		static void serializeLights(Scene& scene, YAML::Emitter& out);

		static void serializeSerializable(Serializable* serializable, YAML::Emitter& out);

		static void deserializeModels(YAML::Node node, Scene& scene);
		static void deserializeLights(YAML::Node node, Scene& scene);

		static void deserializeSerializable(YAML::Node node, Serializable* serializable);
	};
}