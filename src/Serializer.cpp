#include "Serializer.h"
#include <fstream>

#include "Logger.h"

namespace pathtracex {
#define SCENE_FOLDER_PATH "../../scenes/"
#define SCENE_FILE_EXTENSION ".yaml"

	void Serializer::serializeScene(Scene& scene)
	{
		// Create yaml file
		createYAMLFile(SCENE_FOLDER_PATH, scene.sceneName);

		YAML::Emitter out;
		out << YAML::BeginMap;

		serializeModels(scene, out);

		out << YAML::EndMap;

		std::ofstream fout(SCENE_FOLDER_PATH + scene.sceneName + SCENE_FILE_EXTENSION);
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
			out << YAML::Key << "PrimativeType" << YAML::Value << Model::primitiveModelTypeToString(model->primativeType);
			out << YAML::EndMap;
		}


		out << YAML::EndSeq;
	}

	void Serializer::deserializeScene(const std::string& sceneName, Scene& scene)
	{
		LOG_TRACE("Deserializing scene: {}", sceneName);
		std::string scenePath = SCENE_FOLDER_PATH + sceneName + SCENE_FILE_EXTENSION;
		LOG_TRACE("Loading file: " + scenePath);
		YAML::Node state = YAML::LoadFile(scenePath);
		deserializeModels(state, scene);
	}

	// Deserializes the given serializable from the given YAML node
// The given node should be the components node of a game object
	void Serializer::deserializeSerializable(YAML::Node node, Serializable* serializable)
	{
		// TODO: improve complexity, currently does unnecessary iterations
		for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
		{
			std::string nodeName = it->first.as<std::string>();
			for (auto serializableVariable : serializable->getSerializableVariables())
			{
				if (nodeName == serializableVariable.name)
				{
					if (serializableVariable.type == SerializableType::FLOAT)
					{
						*static_cast<float*>(serializableVariable.data) = it->second.as<float>();
					}
					else if (serializableVariable.type == SerializableType::VECTOR3 || serializableVariable.type == SerializableType::COLOR)
					{
						std::vector<float> vec3 = it->second.as<std::vector<float>>();
						std::copy(vec3.begin(), vec3.end(), static_cast<float*>(serializableVariable.data));
					}
					else if (serializableVariable.type == SerializableType::VECTOR4)
					{
						std::vector<float> vec4 = it->second.as<std::vector<float>>();
						std::copy(vec4.begin(), vec4.end(), static_cast<float*>(serializableVariable.data));
					}
					else if (serializableVariable.type == SerializableType::BOOLEAN)
					{
						*static_cast<bool*>(serializableVariable.data) = it->second.as<bool>();
					}
					else if (serializableVariable.type == SerializableType::STRING)
					{
						*static_cast<std::string*>(serializableVariable.data) = it->second.as<std::string>();
					}
					else if (serializableVariable.type == SerializableType::INT)
					{
						*static_cast<int*>(serializableVariable.data) = it->second.as<int>();
					}
					else if (serializableVariable.type == SerializableType::MATRIX4X4)
					{
						std::vector<float> mat4 = it->second.as<std::vector<float>>();
						DirectX::XMMATRIX matrix;
						std::copy(mat4.begin(), mat4.end(), matrix.r->m128_f32);
						*static_cast<DirectX::XMMATRIX*>(serializableVariable.data) = matrix;
					}
					else
					{
						LOG_ERROR("Failed to deserialize serializable because of unknown serializable type");
					}

				}
			}
		}
	}

	void Serializer::deserializeModels(YAML::Node node, Scene& scene)
	{
		YAML::Node modelsNode;
		try
		{
			modelsNode = node["Models"];
		}
		catch (const std::exception& e)
		{
			LOG_ERROR("Failed to deserialize Models: " + std::string(e.what()));
			return;
		}

		for (YAML::const_iterator it = modelsNode.begin(); it != modelsNode.end(); ++it)
		{
			YAML::Node modelNode = *it;
			std::string primativeType = modelNode["PrimativeType"].as<std::string>();
			PrimitiveModelType type = Model::stringToPrimitiveModelType(primativeType);
			std::string filename = modelNode["Filename"].as<std::string>();
			
			std::shared_ptr<Model> model;

			if (type != PrimitiveModelType::NONE)
			{
				model = Model::createPrimative(type);
			}
			else
			{
				model = std::make_shared<Model>(filename);
			}

			deserializeSerializable(modelNode, model.get());
			scene.models.push_back(model);

			//std::string filename = textureNode["fileName"].as<std::string>();
			//Texture* texture = Texture::create(filename);
			//deserializeTexture(*it, game, texture);
		}
	}

}