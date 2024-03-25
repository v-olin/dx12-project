#include "Model.h"
#include <iostream>
#include <algorithm>
// DON'T REMOVE DEFINES, AND DON'T DEFINE ANYWHERE ELSE!!!!!!!!!!!!!
#define TINYOBJLOADER_IMPLEMENTATION
#include "../../vendor/tinyobjloader/tiny_obj_loader.h"
namespace pathtracex {
	namespace file_util {

		std::string normalise(const std::string& file_name);
		std::string parent_path(const std::string& file_name);
		std::string file_stem(const std::string& file_name);
		std::string file_extension(const std::string& file_name);
		std::string change_extension(const std::string& file_name, const std::string& ext);

		std::string normalise(const std::string& file_name)
		{
			std::string nname;
			nname.reserve(file_name.size());
			for (const char c : file_name)
			{
				if (c == '\\')
				{
					if (nname.back() != '/')
					{
						nname += '/';
					}
				}
				else
				{
					nname += c;
				}
			}

			return nname;
		}

		std::string file_stem(const std::string& file_name)
		{
			size_t slash = file_name.find_last_of("\\/");
			size_t dot = file_name.find_last_of(".");
			if (slash != std::string::npos)
			{
				return file_name.substr(slash + 1, dot - slash - 1);
			}
			else
			{
				return file_name.substr(0, dot);
			}
		}

		std::string file_extension(const std::string& file_name)
		{
			size_t separator = file_name.find_last_of(".");
			if (separator == std::string::npos)
			{
				return "";
			}
			else
			{
				return file_name.substr(separator);
			}
		}

		std::string change_extension(const std::string& file_name, const std::string& ext)
		{
			size_t separator = file_name.find_last_of(".");
			if (separator == std::string::npos)
			{
				return file_name + ext;
			}
			else
			{
				return file_name.substr(0, separator) + ext;
			}
		}

		std::string parent_path(const std::string& file_name)
		{
			size_t separator = file_name.find_last_of("\\/");
			if (separator != std::string::npos)
			{
				return file_name.substr(0, separator + 1);
			}
			else
			{
				return "./";
			}
		}

	}	
	void Texture::free()
	{
	/*	if (data)
		{
			stbi_image_free(data);
			data = nullptr;
		}
		if (gl_id_internal)
		{
			glDeleteTextures(1, &gl_id_internal);
			gl_id_internal = 0;
		}*/
	}

	bool Texture::load(const std::string& _directory, const std::string& _filename, int _components)
	{
		filename = file_util::normalise(_filename);
		directory = file_util::normalise(_directory);
		valid = true;
		int components;
		//TODO figure out how to load textures with directX
		/*stbi_set_flip_vertically_on_load(true);
		data = stbi_load((directory + filename).c_str(), &width, &height, &components, _components);
		if (data == nullptr)
		{
			Logger::log("ERROR: loadModelFromOBJ(): Failed to load texture: " + filename + " in " + directory, LOG_LEVEL_FATAL);
		}
		glGenTextures(1, &gl_id_internal);
		gl_id = gl_id_internal;
		glBindTexture(GL_TEXTURE_2D, gl_id_internal);
		GLenum format, internal_format;
		n_components = _components;
		if (_components == 1)
		{
			format = GL_R;
			internal_format = GL_R8;
		}
		else if (_components == 3)
		{
			format = GL_RGB;
			internal_format = GL_RGB;
		}
		else if (_components == 4)
		{
			format = GL_RGBA;
			internal_format = GL_RGBA;
		}
		else
		{
			Logger::log("Texture loading not implemented for this number of compenents.\n", LOG_LEVEL_FATAL);
		}
		glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

		stbi_image_free(data);
		glBindTexture(GL_TEXTURE_2D, 0);*/
		return true;
	}

	float4 Texture::sample(float2 uv) const
	{
		int x = int(uv.x * width + 0.5) % width;
		int y = int(uv.y * height + 0.5) % height;
		if (n_components == 4)
		{
			return float4(data[(y * width + x) * 4 + 0], data[(y * width + x) * 4 + 1],
				data[(y * width + x) * 4 + 2], data[(y * width + x) * 4 + 3])
				/ 255.f;
		}
		else
		{
			// Just return one channel
			return float4(data[(y * width + x) * n_components + 0], data[(y * width + x) * n_components + 0],
				data[(y * width + x) * n_components + 0], data[(y * width + x) * n_components + 0])
				/ 255.f;
		}
	}
	Model::~Model()
	{
		for (auto& material : m_materials)
		{
			if (material.m_color_texture.valid)
				material.m_color_texture.free();
			if (material.m_shininess_texture.valid)
				material.m_shininess_texture.free();
			if (material.m_metalness_texture.valid)
				material.m_metalness_texture.free();
			if (material.m_fresnel_texture.valid)
				material.m_fresnel_texture.free();
			if (material.m_emission_texture.valid)
				material.m_emission_texture.free();
		}
	/*	glDeleteBuffers(1, &m_positions_bo);
		glDeleteBuffers(1, &m_normals_bo);
		glDeleteBuffers(1, &m_texture_coordinates_bo);*/
	}
	Model::Model(std::string name
			, std::vector<Material> materials
			, std::vector<Mesh> meshes
			, DXVertexBuffer* vertex_buffer
			, DXIndexBuffer* index_buffer
			, bool hasDedicatedShader
			, float3 max_cords
			, float3 min_cords
		, std::vector<Vertex> vertices = {}
		, std::vector<uint32_t> indices = {})
		: m_name(name)
		, m_materials(materials)
		, m_meshes(meshes)
		, m_vertex_buffer(vertex_buffer)
		, m_index_buffer(index_buffer)
		, m_hasDedicatedShader(hasDedicatedShader)
		, m_max_cords(max_cords)
		, m_min_cords(min_cords)
		, vertices(vertices)
		, indices(indices)
	{
		// Create the vertex buffer and index buffer

	}


	Model::Model(std::string path)
		: m_hasDedicatedShader(false)
		//TODO: This could be fucked
		, m_max_cords(-INFINITE)
		, m_min_cords(INFINITE)
		, m_vertex_buffer(nullptr)
		, m_index_buffer(nullptr)
	{
		std::string filename, extension, directory;

		filename = file_util::normalise(path);
		directory = file_util::parent_path(path);
		filename = file_util::file_stem(path);
		extension = file_util::file_extension(path);


		if (extension != ".obj")
		{
			std::cout << "Fatal: loadModelFromOBJ(): Expecting filename ending in '.obj'\n";
			exit(1);
		}

		std::cout << "Loading " << path << "..." << std::flush;
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn;
		std::string err;

		// Expect '.mtl' file in the same directory and triangulate meshes
		bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
			(directory + filename + extension).c_str(), directory.c_str(), true);

		// `err` may contain warning message.
		if (!err.empty())
			std::cerr << err << std::endl;

		if (!ret)
			exit(1);

		m_name = filename;
		m_filename = path;

		for (const auto& m : materials)
		{
			Material material;
			material.m_name = m.name;
			material.m_color = float3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
			if (m.diffuse_texname != "")
			{
				material.m_color_texture.load(directory, m.diffuse_texname, 4);
			}
			material.m_metalness = m.metallic;
			if (m.metallic_texname != "")
			{
				material.m_metalness_texture.load(directory, m.metallic_texname, 1);
			}
			material.m_fresnel = m.specular[0];
			if (m.specular_texname != "")
			{
				material.m_fresnel_texture.load(directory, m.specular_texname, 1);
			}
			material.m_shininess = m.roughness;
			if (m.roughness_texname != "")
			{
				material.m_shininess_texture.load(directory, m.roughness_texname, 1);
			}
			material.m_emission = float3(m.emission[0], m.emission[1], m.emission[2]);
			if (m.emissive_texname != "")
			{
				material.m_emission_texture.load(directory, m.emissive_texname, 4);
			}
			material.m_transparency = m.transmittance[0];
			material.m_ior = m.ior;
			m_materials.push_back(material);
		}

		uint64_t number_of_vertices = 0;
		for (const auto& shape : shapes)
		{
			number_of_vertices += shape.mesh.indices.size();
		}
		std::vector<float3> m_positions;
		std::vector<float3> m_normals;
		std::vector<float2> m_texture_coordinates;

		m_positions.resize(number_of_vertices);
		m_normals.resize(number_of_vertices);
		m_texture_coordinates.resize(number_of_vertices);

		std::vector<DirectX::SimpleMath::Vector4> auto_normals(attrib.vertices.size() / 3);
		for (const auto& shape : shapes)
		{
			for (int face = 0; face < int(shape.mesh.indices.size()) / 3; face++)
			{
				float3 v0 = float3(attrib.vertices[shape.mesh.indices[face * 3 + 0].vertex_index * 3 + 0],
					attrib.vertices[shape.mesh.indices[face * 3 + 0].vertex_index * 3 + 1],
					attrib.vertices[shape.mesh.indices[face * 3 + 0].vertex_index * 3 + 2]);
				float3 v1 = float3(attrib.vertices[shape.mesh.indices[face * 3 + 1].vertex_index * 3 + 0],
					attrib.vertices[shape.mesh.indices[face * 3 + 1].vertex_index * 3 + 1],
					attrib.vertices[shape.mesh.indices[face * 3 + 1].vertex_index * 3 + 2]);
				float3 v2 = float3(attrib.vertices[shape.mesh.indices[face * 3 + 2].vertex_index * 3 + 0],
					attrib.vertices[shape.mesh.indices[face * 3 + 2].vertex_index * 3 + 1],
					attrib.vertices[shape.mesh.indices[face * 3 + 2].vertex_index * 3 + 2]);

				float3 e0 = float3(v1 - v0);
				float3 e1 = v2 - v0;
				e0.Normalize();
				e1.Normalize();
				float3 face_normal = e0.Cross(e1);

				auto_normals[shape.mesh.indices[face * 3 + 0].vertex_index] += float4(face_normal.x, face_normal.y, face_normal.z, 1.0f);
				auto_normals[shape.mesh.indices[face * 3 + 1].vertex_index] += float4(face_normal.x, face_normal.y, face_normal.z, 1.0f);
				auto_normals[shape.mesh.indices[face * 3 + 2].vertex_index] += float4(face_normal.x, face_normal.y, face_normal.z, 1.0f);
			}
		}
		for (auto& normal : auto_normals)
		{
			normal = (1.0f / normal.w) * normal;
		}

		int vertices_so_far = 0;
		for (int s = 0; s < shapes.size(); ++s)
		{
			const auto& shape = shapes[s];
			int next_material_index = shape.mesh.material_ids[0];
			int next_material_starting_face = 0;
			std::vector<bool> finished_materials(materials.size(), false);
			int number_of_materials_in_shape = 0;
			while (next_material_index != -1)
			{
				int current_material_index = next_material_index;
				int current_material_starting_face = next_material_starting_face;
				next_material_index = -1;
				next_material_starting_face = -1;
				// Process a new Mesh with a unique material
				Mesh mesh;
				mesh.m_name = shape.name + "_" + materials[current_material_index].name;
				mesh.m_material_idx = current_material_index;
				mesh.m_start_index = vertices_so_far;
				number_of_materials_in_shape += 1;

				uint64_t number_of_faces = shape.mesh.indices.size() / 3;
				for (int i = current_material_starting_face; i < number_of_faces; i++)
				{
					if (shape.mesh.material_ids[i] != current_material_index)
					{
						if (next_material_index >= 0)
							continue;
						else if (finished_materials[shape.mesh.material_ids[i]])
							continue;
						else
						{ // Found a new material that we have not processed.
							next_material_index = shape.mesh.material_ids[i];
							next_material_starting_face = i;
						}
					}
					else
					{
						for (int j = 0; j < 3; j++)
						{
							int v = shape.mesh.indices[i * 3 + j].vertex_index;
							m_positions[vertices_so_far + j] =
								float3(attrib.vertices[shape.mesh.indices[i * 3 + j].vertex_index * 3 + 0],
									attrib.vertices[shape.mesh.indices[i * 3 + j].vertex_index * 3 + 1],
									attrib.vertices[shape.mesh.indices[i * 3 + j].vertex_index * 3 + 2]);


							/*	
							auto elementMax = [](float3 v1, float3 v2) {
								return float3(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z));
								};
							auto elementMin = [](float3 v1, float3 v2) {
								return float3(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z));
								};
							max_cords = elementMax(m_positions[vertices_so_far + j], max_cords);
							min_cords = elementMin(m_positions[vertices_so_far + j], min_cords);
							*/
							m_max_cords = m_max_cords.Max(m_positions[vertices_so_far + j], m_max_cords);
							m_min_cords = m_min_cords.Min(m_min_cords, m_positions[vertices_so_far + j]);

							if (shape.mesh.indices[i * 3 + j].normal_index == -1)
							{
								// No normal, use the autogenerated
								m_normals[vertices_so_far + j] = float3(
									auto_normals[shape.mesh.indices[i * 3 + j].vertex_index]);
							}
							else
							{
								m_normals[vertices_so_far + j] =
									float3(attrib.normals[shape.mesh.indices[i * 3 + j].normal_index * 3 + 0],
										attrib.normals[shape.mesh.indices[i * 3 + j].normal_index * 3 + 1],
										attrib.normals[shape.mesh.indices[i * 3 + j].normal_index * 3 + 2]);
							}
							if (shape.mesh.indices[i * 3 + j].texcoord_index == -1)
							{
								// No UV coordinates. Use null.
								m_texture_coordinates[vertices_so_far + j] = float2(0.0f);
							}
							else
							{
								m_texture_coordinates[vertices_so_far + j] = float2(
									attrib.texcoords[shape.mesh.indices[i * 3 + j].texcoord_index * 2 + 0],
									attrib.texcoords[shape.mesh.indices[i * 3 + j].texcoord_index * 2 + 1]);
							}
						}
						vertices_so_far += 3;
					}
				}

				mesh.m_number_of_vertices = vertices_so_far - mesh.m_start_index;
				m_meshes.push_back(mesh);
				finished_materials[current_material_index] = true;
			}
			if (number_of_materials_in_shape == 1)
			{
				m_meshes.back().m_name = shape.name;
			}
		}

		std::sort(m_meshes.begin(), m_meshes.end(),
			[](const Mesh& a, const Mesh& b) { return a.m_name < b.m_name; });

		//TODO: the hard shit, aka translate to DirectX
		/*
		glGenVertexArrays(1, &m_vaob);
		glBindVertexArray(m_vaob);
		glGenBuffers(1, &m_positions_bo);
		glBindBuffer(GL_ARRAY_BUFFER, m_positions_bo);
		glBufferData(GL_ARRAY_BUFFER, m_positions.size() * sizeof(DirectX::float3), &m_positions[0].x,
			GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);
		glEnableVertexAttribArray(0);
		glGenBuffers(1, &m_normals_bo);
		glBindBuffer(GL_ARRAY_BUFFER, m_normals_bo);
		glBufferData(GL_ARRAY_BUFFER, m_normals.size() * sizeof(DirectX::float3), &m_normals[0].x,
			GL_STATIC_DRAW);
		glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, 0);
		glEnableVertexAttribArray(1);
		glGenBuffers(1, &m_texture_coordinates_bo);
		glBindBuffer(GL_ARRAY_BUFFER, m_texture_coordinates_bo);
		glBufferData(GL_ARRAY_BUFFER, m_texture_coordinates.size() * sizeof(float2),
			&m_texture_coordinates[0].x, GL_STATIC_DRAW);
		glVertexAttribPointer(2, 2, GL_FLOAT, false, 0, 0);
		glEnableVertexAttribArray(2);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		*/

		std::cout << "done.\n";
	}
	void Model::Draw() {
		/* TODO: implement draw?
		glBindVertexArray(m_vaob);
		static ShaderManager& sm = ShaderManager::getInstance();
		auto shader = sm.getDefaultShader();
		for (auto& mesh : m_meshes) {
			const Material& mat = m_materials[mesh.m_material_idx];
			bool hasColor = mat.m_color_texture.valid;
			sm.SetInteger1(shader, hasColor ? 1 : 0, "has_color_texture");
			if (hasColor) {
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, mat.m_color_texture.gl_id);
			}
			bool hasEmission = mat.m_emission_texture.valid;
			sm.SetInteger1(shader, hasEmission ? 1 : 0, "has_emission_texture");

			if (hasEmission) {
				glActiveTexture(GL_TEXTURE5);
				glBindTexture(GL_TEXTURE_2D, mat.m_emission_texture.gl_id);
			}
			glActiveTexture(0);
			sm.SetVec3(shader, mat.m_color, "material_color");

			sm.SetFloat1(shader, mat.m_metalness, "material_metalness");
			sm.SetFloat1(shader, mat.m_fresnel, "material_fresnel");
			sm.SetFloat1(shader, mat.m_shininess, "material_shininess");
			sm.SetVec3(shader, mat.m_emission, "material_emission");
			glDrawArrays(GL_TRIANGLES, mesh.m_start_index, (GLsizei)mesh.m_number_of_vertices);
		}
		glBindVertexArray(0);
		*/
	}

	std::shared_ptr<Model> Model::createPrimative(PrimitiveModelType type) {
		switch (type)
		{
		case pathtracex::CUBE:
			return createCube();
		case pathtracex::SPHERE:
			return createSphere(10, 10);
		case pathtracex::CYLINDER:
			break;
		case pathtracex::PLANE:
			return createPlane();
		case pathtracex::NONE:
			break;
		default:
			return nullptr;
		}
	}

	std::shared_ptr<Model> Model::createCube() {
		std::vector<Vertex> tmp_vertices{};
		std::vector<Vertex> vertecies;
		std::vector<uint32_t> indices{};
		std::vector<float3> m_positions;
		std::vector<float3> m_normals;
		std::vector<float2> m_texture_coordinates;
		std::vector<Mesh> meshes;
		std::vector<Material> materials;
		Vertex vertex;
		size_t number_of_vertices = 36;

		m_positions.resize(number_of_vertices);
		m_normals.resize(number_of_vertices);
		m_texture_coordinates.resize(number_of_vertices);


		// Front
		tmp_vertices.push_back({ float3(-0.5f, -0.5f, 0.5f), float4(1, 0, 0, 1), float3(0, 0, 1), float2(0, 0) });
		tmp_vertices.push_back({ float3(0.5f, -0.5f, 0.5f), float4(0, 1, 0, 1), float3(0, 0, 1), float2(1, 0) });
		tmp_vertices.push_back({ float3(0.5f, 0.5f, 0.5f), float4(0, 0, 1, 1), float3(0, 0, 1), float2(1, 1) });
		tmp_vertices.push_back({ float3(-0.5f, 0.5f, 0.5f), float4(1, 1, 0, 1), float3(0, 0, 1), float2(0, 1) });

		indices.push_back(0); indices.push_back(1); indices.push_back(2);
		indices.push_back(2); indices.push_back(3); indices.push_back(0);
		Mesh front_mesh;
		front_mesh.m_name = "front_mesh";
		front_mesh.m_material_idx = 0;

		front_mesh.m_start_index = 0;
		front_mesh.m_number_of_vertices = 6;
		meshes.push_back(front_mesh);

		for (int i = 0; i < 6; i++) {
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}

		// Back
		tmp_vertices.push_back({ float3(-0.5f, -0.5f, -0.5f), float4(1, 0, 0, 1), float3(0, 0, -1), float2(0, 0) });
		tmp_vertices.push_back({ float3(0.5f, -0.5f, -0.5f), float4(0, 1, 0, 1), float3(0, 0, -1), float2(1, 0) });
		tmp_vertices.push_back({ float3(0.5f, 0.5f, -0.5f), float4(0, 0, 1, 1), float3(0, 0, -1), float2(1, 1) });
		tmp_vertices.push_back({ float3(-0.5f, 0.5f, -0.5f), float4(1, 1, 0, 1), float3(0, 0, -1), float2(0, 1) });

		indices.push_back(6); indices.push_back(5); indices.push_back(4);
		indices.push_back(4); indices.push_back(7); indices.push_back(6);
		Mesh back_mesh;
		back_mesh.m_name = "back_mesh";
		back_mesh.m_material_idx = 1;

		back_mesh.m_start_index = 6;
		back_mesh.m_number_of_vertices = 6;
		meshes.push_back(back_mesh);

		for (int i = 6; i < 12; i++) {
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}

		// Left
		tmp_vertices.push_back({ float3(-0.5f, -0.5f, -0.5f), float4(1, 0, 0, 1), float3(-1, 0, 0), float2(0, 0) });
		tmp_vertices.push_back({ float3(-0.5f, -0.5f, 0.5f), float4(0, 1, 0, 1), float3(-1, 0, 0), float2(1, 0) });
		tmp_vertices.push_back({ float3(-0.5f, 0.5f, 0.5f), float4(0, 0, 1, 1), float3(-1, 0, 0), float2(1, 1) });
		tmp_vertices.push_back({ float3(-0.5f, 0.5f, -0.5f), float4(1, 1, 0, 1), float3(-1, 0, 0), float2(0, 1) });

		indices.push_back(8); indices.push_back(9); indices.push_back(10);
		indices.push_back(10); indices.push_back(11); indices.push_back(8);

		Mesh left_mesh;
		left_mesh.m_name = "left_mesh";
		left_mesh.m_material_idx = 2;

		left_mesh.m_start_index = 12;
		left_mesh.m_number_of_vertices = 6;
		meshes.push_back(left_mesh);

		for (int i = 12; i < 18; i++) {
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}

		// Right
		tmp_vertices.push_back({ float3(0.5f, -0.5f, -0.5f), float4(1, 0, 0, 1), float3(1, 0, 0), float2(0, 0) });
		tmp_vertices.push_back({ float3(0.5f, -0.5f, 0.5f), float4(0, 1, 0, 1), float3(1, 0, 0), float2(1, 0) });
		tmp_vertices.push_back({ float3(0.5f, 0.5f, 0.5f), float4(0, 0, 1, 1), float3(1, 0, 0), float2(1, 1) });
		tmp_vertices.push_back({ float3(0.5f, 0.5f, -0.5f), float4(1, 1, 0, 1), float3(1, 0, 0), float2(0, 1) });

		indices.push_back(14); indices.push_back(13); indices.push_back(12);
		indices.push_back(12); indices.push_back(15); indices.push_back(14);
		Mesh right_mesh;
		right_mesh.m_name = "right_mesh";
		right_mesh.m_material_idx = 3;

		right_mesh.m_start_index = 18;
		right_mesh.m_number_of_vertices = 6;
		meshes.push_back(right_mesh);

		for (int i = 18; i < 24; i++) {
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}



		// Top
		tmp_vertices.push_back({ float3(-0.5f, 0.5f, -0.5f), float4(1, 0, 0, 1), float3(0, 1, 0), float2(0, 0) });
		tmp_vertices.push_back({ float3(0.5f, 0.5f, -0.5f), float4(0, 1, 0, 1), float3(0, 1, 0), float2(1, 0) });
		tmp_vertices.push_back({ float3(0.5f, 0.5f, 0.5f), float4(0, 0, 1, 1), float3(0, 1, 0), float2(1, 1) });
		tmp_vertices.push_back({ float3(-0.5f, 0.5f, 0.5f), float4(1, 1, 0, 1), float3(0, 1, 0), float2(0, 1) });

		indices.push_back(18); indices.push_back(17); indices.push_back(16);
		indices.push_back(16); indices.push_back(19); indices.push_back(18);
		Mesh top_mesh;
		top_mesh.m_name = "top_mesh";
		top_mesh.m_material_idx = 4;

		top_mesh.m_start_index = 24;
		top_mesh.m_number_of_vertices = 6;
		meshes.push_back(top_mesh);

		for (int i = 24; i < 30; i++) {
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}



		// Bottom
		tmp_vertices.push_back({ float3(-0.5f, -0.5f, -0.5f), float4(1, 0, 0, 1), float3(0, -1, 0), float2(0, 0) });
		tmp_vertices.push_back({ float3(0.5f, -0.5f, -0.5f), float4(0, 1, 0, 1), float3(0, -1, 0), float2(1, 0) });
		tmp_vertices.push_back({ float3(0.5f, -0.5f, 0.5f), float4(0, 0, 1, 1), float3(0, -1, 0), float2(1, 1) });
		tmp_vertices.push_back({ float3(-0.5f, -0.5f, 0.5f), float4(1, 1, 0, 1), float3(0, -1, 0), float2(0, 1) });

		indices.push_back(20); indices.push_back(21); indices.push_back(22);
		indices.push_back(22); indices.push_back(23); indices.push_back(20);

		Mesh bottom_mesh;
		bottom_mesh.m_name = "bottom_mesh";
		bottom_mesh.m_material_idx = 5;

		bottom_mesh.m_start_index = 30;
		bottom_mesh.m_number_of_vertices = 6;
		meshes.push_back(bottom_mesh);

		for (int i = 30; i < 36; i++) {
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}
	

		float3 max_cords(m_positions.at(0));
		float3 min_cords(m_positions.at(0));
		for (size_t i = 1; i < m_positions.size(); i++) {
			max_cords = max_cords.Max(max_cords, m_positions.at(i));
			min_cords = max_cords.Min(min_cords, m_positions.at(i));
		}

		//TODO: create buffers on gpu so that bo:s actually are something...
		//TODO creatye materials for cube faces, 
		// posibly 6 diffwerent ones loaded from a file so that you can tweek and save changes
		DXVertexBuffer*	vertex_buffer = new DXVertexBuffer(vertecies);
		std::vector<DWORD> indecies;
		for (size_t i = 0; i < vertecies.size(); i++)
			indecies.push_back(i);
		DXIndexBuffer* index_buffer = new DXIndexBuffer(indecies);


		return std::make_shared<Model>("Primative Cube"
			, materials
			, meshes
			, vertex_buffer
			, index_buffer
			, false
			, max_cords
			, min_cords);
	}
	std::shared_ptr<Model> Model::createSphere(int stacks, int slices) {
		std::vector<Vertex> tmp_vertices{};
		std::vector<uint32_t> indices{};

		float radius = 1.0f;
		float sectorStep = 2 * 3.1415 / slices;
		float stackStep = 3.1415 / stacks;
		float sectorAngle, stackAngle;
		float4 cols[] = {
			{1, 0, 0, 1},
			{0, 1, 0, 1},
			{0, 0, 1, 1}
		};
		for (int i = 0; i <= stacks; ++i)
		{
			stackAngle = 3.1415 / 2 - i * stackStep;
			float xy = radius * cosf(stackAngle);
			float z = radius * sinf(stackAngle);

			for (int j = 0; j <= slices; ++j)
			{
				sectorAngle = j * sectorStep;

				float x = xy * cosf(sectorAngle);
				float y = xy * sinf(sectorAngle);

				float3 normal = (float3(x, y, z));
				normal.Normalize();
				float2 uv = float2((float)j / slices, (float)i / stacks);

				tmp_vertices.push_back({ float3(x, y, z), cols[i%3], normal, uv});
			}
		}

		int k1, k2;

		for (int i = 0; i < stacks; ++i)
		{
			k1 = i * (slices + 1);
			k2 = k1 + slices + 1;

			for (int j = 0; j < slices; ++j, ++k1, ++k2)
			{
				if (i != 0)
				{
					indices.push_back(k1);
					indices.push_back(k2);
					indices.push_back(k1 + 1);
				}

				if (i != (stacks - 1))
				{
					indices.push_back(k1 + 1);
					indices.push_back(k2);
					indices.push_back(k2 + 1);
				}
			}
		}
		Mesh mesh;
		mesh.m_material_idx = 0;
		mesh.m_name = "sphere_mesh";
		mesh.m_start_index = 0;
		mesh.m_number_of_vertices = indices.size();
		
		std::vector<Vertex> vertecies;
		size_t idx;
		for (size_t i = 0; i < indices.size(); i++) {
			idx = indices.at(i);
			vertecies.push_back(tmp_vertices.at(idx));
		}
		std::vector<Mesh> meshes;
		std::vector<Material> materials;

		float3 max_cords(vertecies.at(0).pos);
		float3 min_cords(vertecies.at(0).pos);
		for (size_t i = 1; i < vertecies.size(); i++) {
			max_cords = max_cords.Max(max_cords, vertecies.at(0).pos);
			min_cords = max_cords.Min(min_cords, vertecies.at(0).pos);
		}


		DXVertexBuffer*	vertex_buffer = new DXVertexBuffer(vertecies);
		std::vector<DWORD> indecies;
		for (size_t i = 0; i < vertecies.size(); i++)
			indecies.push_back(i);
		DXIndexBuffer* index_buffer = new DXIndexBuffer(indecies);
		return std::make_shared<Model>("Primative Sphere"
			, materials
			, meshes
			, vertex_buffer
			, index_buffer
			, false
			, max_cords
			, min_cords);
	}
	std::shared_ptr<Model> Model::createPlane() {
		std::vector<Vertex> tmp_vertices{};
		std::vector<uint32_t> indices{};

		tmp_vertices.push_back({ float3(-5.f, 0.0f, -5.f), float4(1, 0, 0, 1), float3(0, 1, 0), float2(0, 0) });
		tmp_vertices.push_back({ float3(5.f, 0.0f, -5.f), float4(0, 1, 0, 1), float3(0, 1, 0), float2(1, 0) });
		tmp_vertices.push_back({ float3(5.f, 0.0f, 5.f), float4(0, 0, 1, 1), float3(0, 1, 0), float2(1, 1) });
		tmp_vertices.push_back({ float3(-5.f, 0.0f, 5.f), float4(1, 1, 0, 1), float3(0, 1, 0), float2(0, 1) });

		indices.push_back(0); indices.push_back(1); indices.push_back(2);
		indices.push_back(2); indices.push_back(3); indices.push_back(0);
		indices.push_back(2); indices.push_back(1); indices.push_back(0);
		indices.push_back(0); indices.push_back(3); indices.push_back(2);

		Mesh mesh;
		mesh.m_material_idx = 0;
		mesh.m_name = "sphere_mesh";
		mesh.m_start_index = 0;
		mesh.m_number_of_vertices = indices.size();
		
		std::vector<Vertex> vertecies;
		size_t idx;
		for (size_t i = 0; i < indices.size(); i++) {
			idx = indices.at(i);
			vertecies.push_back(tmp_vertices.at(idx));
		}
		std::vector<Mesh> meshes;
		std::vector<Material> materials;

		float3 max_cords(vertecies.at(0).pos);
		float3 min_cords(vertecies.at(0).pos);
		for (size_t i = 1; i < vertecies.size(); i++) {
			max_cords = max_cords.Max(max_cords, vertecies.at(0).pos);
			min_cords = max_cords.Min(min_cords, vertecies.at(0).pos);
		}

		DXVertexBuffer*	vertex_buffer = new DXVertexBuffer(vertecies);
		std::vector<DWORD> indecies;
		for (size_t i = 0; i < vertecies.size(); i++)
			indecies.push_back(i);
		DXIndexBuffer* index_buffer = new DXIndexBuffer(indecies);

		return std::make_shared<Model>("Primative plane"
			, materials
			, meshes
			, vertex_buffer
			, index_buffer
			, false
			, max_cords
			, min_cords);
	}



}