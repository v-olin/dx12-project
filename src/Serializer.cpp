#include "Serializer.h"
#include <fstream>

#include "Logger.h"

namespace pathtracex {
#define SCENE_PATH "../../scenes/"
#define SCENE_FILE_EXTENSION ".yaml"

	void Serializer::serializeScene(Scene& scene)
	{
		// Create yaml file
		createYAMLFile(SCENE_PATH, scene.sceneName);

		YAML::Emitter out;
		out << YAML::BeginMap;

		serializeModels(scene, out);

		out << YAML::EndMap;

		std::ofstream fout(SCENE_PATH + scene.sceneName + SCENE_FILE_EXTENSION);
		fout << out.c_str();
	}

	void Serializer::createYAMLFile(const std::string& fileFolder, const std::string& fileName)
	{
		std::string filePath = fileFolder + fileName + SCENE_FILE_EXTENSION;
		std::ofstream outfile(filePath);
		outfile.close();
	}

	// Serializes a serializable to the given YAML emitter as key value pairs for each serializable variable
	void Serializer::serializeSerializable(Serializable* serializable, YAML::Emitter& out)
	{
		for (const auto serializableVariable : serializable->getSerializableVariables())
		{
			out << YAML::Key << serializableVariable.name;
			if (serializableVariable.type == SerializableType::FLOAT)
			{
				out << YAML::Value << *static_cast<float*>(serializableVariable.data);
			}
			else if (serializableVariable.type == SerializableType::VECTOR3 || serializableVariable.type == SerializableType::COLOR)
			{
				float* vec3 = static_cast<float*>(serializableVariable.data);
				std::vector<float> vec3Vector(vec3, vec3 + 3);
				out << YAML::Flow << YAML::Value << vec3Vector;
			}
			else if (serializableVariable.type == SerializableType::VECTOR4)
			{
				float* vec4 = static_cast<float*>(serializableVariable.data);
				std::vector<float> vec4Vector(vec4, vec4 + 4);
				out << YAML::Flow << YAML::Value << vec4Vector;
			}
			else if (serializableVariable.type == SerializableType::BOOLEAN)
			{
				out << YAML::Value << *static_cast<bool*>(serializableVariable.data);
			}
			else if (serializableVariable.type == SerializableType::STRING)
			{
				out << YAML::Value << *static_cast<std::string*>(serializableVariable.data);
			}
			else if (serializableVariable.type == SerializableType::INT)
			{
				out << YAML::Value << *static_cast<int*>(serializableVariable.data);
			}
			else if (serializableVariable.type == SerializableType::MATRIX4X4)
			{
				DirectX::XMMATRIX matrix = *static_cast<DirectX::XMMATRIX*>(serializableVariable.data);
				float* mat4Pointer = &matrix.r->m128_f32[0];
				std::vector<float> mat4Vector(mat4Pointer, mat4Pointer + 16);
				out << YAML::Value << YAML::Flow << mat4Vector;
			}
			else
			{
				LOG_ERROR("Failed to serialize serializable because of unknown serializable type");
			}
		}
	}

	void Serializer::serializeModels(Scene& scene, YAML::Emitter& out)
	{
		out << YAML::Key << "Models";
		out << YAML::Value << YAML::BeginSeq;

		for (auto model : scene.models) {
			out << YAML::BeginMap;
			serializeSerializable(model.get(), out);
			out << YAML::EndMap;
		}


		out << YAML::EndSeq;
	}

}
