// Model.h - Description
#pragma once
#include "Transform.h"
#include <string>
#include "Selectable.h"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "Transform.h"
#include "Helper.h"

namespace pathtracex {

	struct Texture {
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

	struct Material {
		std::string m_name;
		float3 m_color;
		float m_shininess;
		float m_metalness;
		float m_fresnel;
		float3 m_emission;
		float m_transparency;
		float m_ior;
		Texture m_color_texture;
		Texture m_shininess_texture;
		Texture m_metalness_texture;
		Texture m_fresnel_texture;
		Texture m_emission_texture;
	};

	struct Mesh {
		std::string m_name;
		uint32_t m_material_idx;
		// Where this Mesh's vertices start
		uint32_t m_start_index;
		uint32_t m_number_of_vertices;
	};

	class Model {
	public:
		Model(std::string path);
		//Model(std::shared_ptr<Model> src);
		~Model();
		void Draw();
		// Buffers on CPU
		/*
		hopefully not needed
		std::vector<DirectX::SimpleMath::Vector3> m_positions;
		std::vector<DirectX::SimpleMath::Vector3> m_normals;
		std::vector<DirectX::SimpleMath::Vector2> m_texture_coordinates;
		*/


		Transform trans;

		// The name of the whole model
		std::string m_name;
		// The filename of this model
		std::string m_filename;
		// The materials
		std::vector<Material> m_materials;
		// A model will contain one or more "Meshes"
		std::vector<Mesh> m_meshes;
		// Buffers on GPU
		//TODO: everything below this might have to be adapted
		uint32_t m_positions_bo;
		uint32_t m_normals_bo;
		uint32_t m_texture_coordinates_bo;
		// Vertex Array Object
		uint32_t m_vaob;
		bool m_hasDedicatedShader;

		float3 max_cords;
		float3 min_cords;
	};
}