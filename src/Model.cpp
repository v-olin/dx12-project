#include "Model.h"
#include <iostream>
#include <algorithm>
#include "Logger.h"
#include "Noise.h"
#include "ProcedualWorldManager.h"

// DON'T REMOVE DEFINES, AND DON'T DEFINE ANYWHERE ELSE!!!!!!!!!!!!!
#define TINYOBJLOADER_IMPLEMENTATION
#include "../../vendor/tinyobjloader/tiny_obj_loader.h"
#include "DXRenderer.h"

namespace pathtracex
{

#define PATH_TO_ASSETS "../../assets/"

	namespace file_util
	{

		std::string normalise(const std::string &file_name)
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

		std::string file_stem(const std::string &file_name)
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

		std::string file_extension(const std::string &file_name)
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

		std::string change_extension(const std::string &file_name, const std::string &ext)
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

		std::string parent_path(const std::string &file_name)
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
	Model::~Model()
	{
		for (auto &material : materials)
		{
			if (material.colorTexture.valid)
				material.colorTexture.free();
			if (material.shininessTexture.valid)
				material.shininessTexture.free();
			if (material.metalnessTexture.valid)
				material.metalnessTexture.free();
			if (material.fresnelTexture.valid)
				material.fresnelTexture.free();
			if (material.emissionTexture.valid)
				material.emissionTexture.free();
		}
		/*	glDeleteBuffers(1, &m_positions_bo);
			glDeleteBuffers(1, &m_normals_bo);
			glDeleteBuffers(1, &m_texture_coordinates_bo);*/
	}
	Model::Model(std::string name, std::vector<Material> materials, std::vector<Mesh> meshes, bool hasDedicatedShader, float3 max_cords, float3 min_cords, std::vector<Vertex> vertices = {}, std::vector<uint32_t> indices = {})
		: name(name), materials(materials), meshes(meshes), maxCords(max_cords), minCords(min_cords), vertices(vertices), indices(indices)
	{
		// Create the vertex buffer and index buffer
		vertexBuffer = std::make_unique<DXVertexBuffer>(vertices);
		indexBuffer = std::make_unique<DXIndexBuffer>(indices);
	}

	Model::Model(std::string filenameWithExtension)
		// TODO: This could be fucked
		: maxCords(-INFINITE), minCords(INFINITE)
	{
		std::string filename, extension, directory;

		std::string path = PATH_TO_ASSETS + filenameWithExtension;

		filename = file_util::normalise(path);
		directory = file_util::parent_path(path);
		filename = file_util::file_stem(path);
		extension = file_util::file_extension(path);

		this->filename = filename + extension;

		if (extension != ".obj")
		{
			std::cout << "Fatal: loadModelFromOBJ(): Expecting filename ending in '.obj'\n";
			exit(1);
		}

		std::cout << "Loading " << path << "..." << std::flush;
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> objMaterials;
		std::string warn;
		std::string err;

		// Expect '.mtl' file in the same directory and triangulate meshes
		bool ret = tinyobj::LoadObj(&attrib, &shapes, &objMaterials, &warn, &err,
									(directory + filename + extension).c_str(), directory.c_str(), true);

		// `err` may contain warning message.
		if (!err.empty())
			std::cerr << err << std::endl;

		if (!ret)
		{
			LOG_FATAL("loadModelFromOBJ(): Failed to load model: {0}", path);
			abort();
		}

		name = filename;
		filename = path;


		D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
		ZeroMemory(&heapDesc, sizeof(heapDesc));
		heapDesc.NumDescriptors = NUMTEXTURETYPES;
		heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

		DXRenderer* renderer = DXRenderer::getInstance();


		for (const auto &m : objMaterials)
		{
			Material material;
			material.name = m.name;
			material.color = float3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);

			renderer->createTextureDescriptorHeap(heapDesc, &material.mainDescriptorHeap);

			if (m.diffuse_texname != "")
			{
				material.colorTexture.load(directory, m.diffuse_texname, 4, &material.mainDescriptorHeap, COLTEX);
			}
			material.metalness = m.metallic;
			if (m.metallic_texname != "")
			{
				material.metalnessTexture.load(directory, m.metallic_texname, 1, &material.mainDescriptorHeap, METALNESSTEX);
			}
			material.fresnel = m.specular[0];
			if (m.specular_texname != "")
			{
				material.fresnelTexture.load(directory, m.specular_texname, 1, &material.mainDescriptorHeap, FRESNELTEX);
			}
			material.shininess = m.roughness;
			if (m.roughness_texname != "")
			{
				material.shininessTexture.load(directory, m.roughness_texname, 1, &material.mainDescriptorHeap, SHININESSTEX);
			}
			material.emission = float3(m.emission[0], m.emission[1], m.emission[2]);
			if (m.emissive_texname != "")
			{
				material.emissionTexture.load(directory, m.emissive_texname, 4, &material.mainDescriptorHeap, EMISIONTEX);
			}
			if (m.bump_texname != "")
			{
				material.normalTexture.load(directory, m.bump_texname, 3, &material.mainDescriptorHeap, NORMALTEX);
			}
			material.transparency = m.transmittance[0];
			material.ior = m.ior;
			materials.push_back(material);
		}

		uint64_t number_of_vertices = 0;
		for (const auto &shape : shapes)
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
		for (const auto &shape : shapes)
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
		for (auto &normal : auto_normals)
		{
			normal = (1.0f / normal.w) * normal;
		}

		int vertices_so_far = 0;
		for (int s = 0; s < shapes.size(); ++s)
		{
			const auto &shape = shapes[s];
			int next_material_index = shape.mesh.material_ids[0];
			int next_material_starting_face = 0;
			std::vector<bool> finished_materials(materials.size(), 0);
			int number_of_materials_in_shape = 0;
			while (next_material_index != -1)
			{
				int current_material_index = next_material_index;
				int current_material_starting_face = next_material_starting_face;
				next_material_index = -1;
				next_material_starting_face = -1;
				// Process a new Mesh with a unique material
				Mesh mesh;
				mesh.name = shape.name + "_" + materials[current_material_index].name;
				mesh.materialIdx = current_material_index;
				mesh.startIndex = vertices_so_far;
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
							maxCords = maxCords.Max(m_positions[vertices_so_far + j], maxCords);
							minCords = minCords.Min(minCords, m_positions[vertices_so_far + j]);

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
								float u = (attrib.texcoords[shape.mesh.indices[i * 3 + j].texcoord_index * 2 + 0]);
								// This "-1) *-1" hack will fuck us if we load textures that use tex coords outside [0,1]
								float v = (attrib.texcoords[shape.mesh.indices[i * 3 + j].texcoord_index * 2 + 1] - 1) * -1;
								m_texture_coordinates[vertices_so_far + j] = float2(u, v);
							}
						}
						vertices_so_far += 3;
					}
				}

				mesh.numberOfVertices = vertices_so_far - mesh.startIndex;
				meshes.push_back(mesh);
				finished_materials[current_material_index] = true;
			}
			if (number_of_materials_in_shape == 1)
			{
				meshes.back().name = shape.name;
			}
		}

		std::sort(meshes.begin(), meshes.end(),
				  [](const Mesh &a, const Mesh &b)
				  { return a.name < b.name; });

		// trying very simple way tyo draw vertecies
		for (int i = 0; i < number_of_vertices; i++)
		{
			float3 n = m_normals.at(i);
			float4 col = {0.0, 0.0, 0.0, 0.0};

			Vertex vert{m_positions.at(i), col, n, m_texture_coordinates.at(i)};
			vertices.push_back(vert);
		}

		for (auto mesh : meshes)
		{
			for (size_t i = 0; i < mesh.numberOfVertices; i++)
			{
				indices.push_back(i + mesh.startIndex);
				if (materials.size() == 0)
					continue;
				auto mat = materials.at(mesh.materialIdx);
				vertices.at(i + mesh.startIndex).color = DirectX::XMFLOAT4(mat.color.x, mat.color.y, mat.color.z, 1.f);
			}
		}
		vertexBuffer = std::make_unique<DXVertexBuffer>(vertices);
		indexBuffer = std::make_unique<DXIndexBuffer>(indices);
		std::cout << "done.\n";
	}

	std::shared_ptr<Model> Model::createPrimative(PrimitiveModelType type)
	{
		switch (type)
		{
		case pathtracex::CUBE:
			return createCube();
		case pathtracex::SPHERE:
			return createSphere(100, 100);
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

	std::string Model::primitiveModelTypeToString(PrimitiveModelType type)
	{
		switch (type)
		{
		case pathtracex::CUBE:
			return "Cube";
			break;
		case pathtracex::SPHERE:
			return "Sphere";
			break;
		case pathtracex::CYLINDER:
			return "Cylinder";
			break;
		case pathtracex::PLANE:
			return "Plane";
			break;
		case pathtracex::NONE:
			return "None";
			break;
		default:
			break;
		}
	}

	PrimitiveModelType Model::stringToPrimitiveModelType(std::string type)
	{
		if (type == "Plane")
			return PrimitiveModelType::PLANE;
		else if (type == "Cube")
			return PrimitiveModelType::CUBE;
		else if (type == "Cylinder")
			return PrimitiveModelType::CYLINDER;
		else if (type == "Sphere")
			return PrimitiveModelType::SPHERE;
		else
		{
			return PrimitiveModelType::NONE;
		}
	}

	std::shared_ptr<Model> Model::createProcedualWorldMesh(float3 startPos, float sideLength, int seed, int tesselation, int heightScale)
	{
		std::vector<float3> positions{};
		std::vector<uint32_t> indices{};
		std::vector<float3> normals{};
		std::vector<float2> texcoords{};
		std::vector<Vertex> vertices{};

		//		m_mapSize = size;
		float sideLen = sideLength / (tesselation - 1);

		float totalX = 0.0f;
		float totalZ = 0.0f;

		for (int i = 0; i < tesselation; i++)
		{
			float x = -sideLength / 2 + i * sideLen;

			for (int j = 0; j < tesselation; j++)
			{
				float z = -sideLength / 2 + j * sideLen;
				float y = Noise::perlin(startPos.x + x, startPos.z + z, 100, seed, 2) // 2 is octaves
						* heightScale;
				positions.push_back({x, y, z});

				texcoords.push_back({totalX / sideLength, totalZ / sideLength});

				if (i != tesselation - 1 && j != tesselation - 1)
				{
					indices.push_back(i * tesselation + j);
					indices.push_back(i * tesselation + j + 1);
					indices.push_back((i + 1) * tesselation + j + 1);

					indices.push_back(i * tesselation + j);
					indices.push_back((i + 1) * tesselation + j + 1);
					indices.push_back((i + 1) * tesselation + j);
				}
				totalZ += sideLen;
			}
			totalZ = 0;
			totalX += sideLen;
		}
		/*

		// despair, k�l please fix
		for (int i = 0; i < tesselation; i++)
		{
			for (int j = 0; j < tesselation; j++)
			{
				//  	A
				//  B	x	C
				//  	D
				vec3 x, a, b, c, d, n;
				x = vertices.at(i * tesselation + j);

				if (i == 0 && j == 0)
				{
					a = vertices.at((i + 1) * tesselation + j);
					c = vertices.at(i * tesselation + j + 1);

					n = cross(c - x, a - x);
				}
				else if (i == tesselation - 1 && j == tesselation - 1)
				{
					b = vertices.at(i * tesselation + j - 1);
					d = vertices.at((i - 1) * tesselation + j);

					n = cross(b - x, d - x);
				}
				else if (i == 0)
				{
					b = vertices.at(i * tesselation + j - 1);
					c = vertices.at(i * tesselation + j + 1);
					d = vertices.at((i + 1) * tesselation + j);

					n = cross(b - x, d - x) + cross(d - x, c - x);
				}
				else if (i == tesselation - 1)
				{
					a = vertices.at((i - 1) * tesselation + j);
					b = vertices.at(i * tesselation + j - 1);
					c = vertices.at(i * tesselation + j + 1);

					n = cross(a - x, b - x) + cross(c - x, a - x);
				}
				else if (j == 0)
				{
					a = vertices.at((i - 1) * tesselation + j);
					c = vertices.at(i * tesselation + j + 1);
					d = vertices.at((i + 1) * tesselation + j);

					n = cross(c - x, a - x) + cross(d - x, c - x);
				}
				else if (j == tesselation - 1)
				{
					a = vertices.at((i - 1) * tesselation + j);
					b = vertices.at(i * tesselation + j - 1);
					d = vertices.at((i + 1) * tesselation + j);

					n = cross(a - x, b - x) + cross(b - x, d - x);
				}
				else
				{
					a = vertices.at((i - 1) * tesselation + j);
					b = vertices.at(i * tesselation + j - 1);
					c = vertices.at(i * tesselation + j + 1);
					d = vertices.at((i + 1) * tesselation + j);

					n = cross(a - x, b - x)
						+ cross(b - x, d - x)
						+ cross(d - x, c - x)
						+ cross(c - x, a - x);
				}
				normals.push_back(-normalize(n));
				//normals.push_back(vec3(0.0, 1.0, 0.0));
			}
		}
			*/

		float3 max_cords(positions.at(0));
		float3 min_cords(positions.at(0));
		for (size_t i = 1; i < positions.size(); i++)
		{
			max_cords = max_cords.Max(max_cords, positions.at(i));
			min_cords = max_cords.Min(min_cords, positions.at(i));
		}

		// TODO creatye materials for cube faces,
		//  posibly 6 diffwerent ones loaded from a file so that you can tweek and save changes
		std::vector<uint32_t> indecies;

		//		for (auto mesh : meshes) {
		//			for (size_t i = 0; i < mesh.numberOfVertices; i++)
		//				indecies.push_back(i + mesh.startIndex);
		//		}

		for (size_t i = 0; i < positions.size(); i++)
		{
			// Color depends on the height
			float4 color = float4(positions.at(i).y / heightScale, 0, 0, 1);
			if (positions.at(i).y < 0)
				color = float4(0, 0, 1, 1);
			vertices.push_back({positions.at(i), color, float3(0, 1, 0), texcoords.at(i)});
		}

		Mesh mesh = Mesh();
		mesh.name = "Procedual mesh";
		mesh.materialIdx = 0;
		mesh.startIndex = 0;
		mesh.numberOfVertices = indices.size();

		std::vector<Material> materials;
		std::vector<Mesh> meshes;

		meshes.push_back(mesh);

		std::shared_ptr<Model> model = std::make_shared<Model>("Procedual mesh", materials, meshes, false, max_cords, min_cords, vertices, indices);
		// model->primativeType = PrimitiveModelType::CUBE;

		return model;
	}

	std::shared_ptr<Model> Model::createCube()
	{
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
		tmp_vertices.push_back({float3(-0.5f, -0.5f, 0.5f), float4(1, 0, 0, 1), float3(0, 0, 1), float2(0, 0)});
		tmp_vertices.push_back({float3(0.5f, -0.5f, 0.5f), float4(0, 1, 0, 1), float3(0, 0, 1), float2(1, 0)});
		tmp_vertices.push_back({float3(0.5f, 0.5f, 0.5f), float4(0, 0, 1, 1), float3(0, 0, 1), float2(1, 1)});
		tmp_vertices.push_back({float3(-0.5f, 0.5f, 0.5f), float4(1, 1, 0, 1), float3(0, 0, 1), float2(0, 1)});

		indices.push_back(0);
		indices.push_back(1);
		indices.push_back(2);
		indices.push_back(2);
		indices.push_back(3);
		indices.push_back(0);
		Mesh front_mesh;
		front_mesh.name = "front_mesh";
		front_mesh.materialIdx = 0;

		front_mesh.startIndex = 0;
		front_mesh.numberOfVertices = 6;
		meshes.push_back(front_mesh);

		for (int i = 0; i < 6; i++)
		{
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}

		// Back
		tmp_vertices.push_back({float3(-0.5f, -0.5f, -0.5f), float4(1, 0, 0, 1), float3(0, 0, -1), float2(0, 0)});
		tmp_vertices.push_back({float3(0.5f, -0.5f, -0.5f), float4(0, 1, 0, 1), float3(0, 0, -1), float2(1, 0)});
		tmp_vertices.push_back({float3(0.5f, 0.5f, -0.5f), float4(0, 0, 1, 1), float3(0, 0, -1), float2(1, 1)});
		tmp_vertices.push_back({float3(-0.5f, 0.5f, -0.5f), float4(1, 1, 0, 1), float3(0, 0, -1), float2(0, 1)});

		indices.push_back(6);
		indices.push_back(5);
		indices.push_back(4);
		indices.push_back(4);
		indices.push_back(7);
		indices.push_back(6);
		Mesh back_mesh;
		back_mesh.name = "back_mesh";
		back_mesh.materialIdx = 1;

		back_mesh.startIndex = 6;
		back_mesh.numberOfVertices = 6;
		meshes.push_back(back_mesh);

		for (int i = 6; i < 12; i++)
		{
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}

		// Left
		tmp_vertices.push_back({float3(-0.5f, -0.5f, -0.5f), float4(1, 0, 0, 1), float3(-1, 0, 0), float2(0, 0)});
		tmp_vertices.push_back({float3(-0.5f, -0.5f, 0.5f), float4(0, 1, 0, 1), float3(-1, 0, 0), float2(1, 0)});
		tmp_vertices.push_back({float3(-0.5f, 0.5f, 0.5f), float4(0, 0, 1, 1), float3(-1, 0, 0), float2(1, 1)});
		tmp_vertices.push_back({float3(-0.5f, 0.5f, -0.5f), float4(1, 1, 0, 1), float3(-1, 0, 0), float2(0, 1)});

		indices.push_back(8);
		indices.push_back(9);
		indices.push_back(10);
		indices.push_back(10);
		indices.push_back(11);
		indices.push_back(8);

		Mesh left_mesh;
		left_mesh.name = "left_mesh";
		left_mesh.materialIdx = 2;

		left_mesh.startIndex = 12;
		left_mesh.numberOfVertices = 6;
		meshes.push_back(left_mesh);

		for (int i = 12; i < 18; i++)
		{
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}

		// Right
		tmp_vertices.push_back({float3(0.5f, -0.5f, -0.5f), float4(1, 0, 0, 1), float3(1, 0, 0), float2(0, 0)});
		tmp_vertices.push_back({float3(0.5f, -0.5f, 0.5f), float4(0, 1, 0, 1), float3(1, 0, 0), float2(1, 0)});
		tmp_vertices.push_back({float3(0.5f, 0.5f, 0.5f), float4(0, 0, 1, 1), float3(1, 0, 0), float2(1, 1)});
		tmp_vertices.push_back({float3(0.5f, 0.5f, -0.5f), float4(1, 1, 0, 1), float3(1, 0, 0), float2(0, 1)});

		indices.push_back(14);
		indices.push_back(13);
		indices.push_back(12);
		indices.push_back(12);
		indices.push_back(15);
		indices.push_back(14);
		Mesh right_mesh;
		right_mesh.name = "right_mesh";
		right_mesh.materialIdx = 3;

		right_mesh.startIndex = 18;
		right_mesh.numberOfVertices = 6;
		meshes.push_back(right_mesh);

		for (int i = 18; i < 24; i++)
		{
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}

		// Top
		tmp_vertices.push_back({float3(-0.5f, 0.5f, -0.5f), float4(1, 0, 0, 1), float3(0, 1, 0), float2(0, 0)});
		tmp_vertices.push_back({float3(0.5f, 0.5f, -0.5f), float4(0, 1, 0, 1), float3(0, 1, 0), float2(1, 0)});
		tmp_vertices.push_back({float3(0.5f, 0.5f, 0.5f), float4(0, 0, 1, 1), float3(0, 1, 0), float2(1, 1)});
		tmp_vertices.push_back({float3(-0.5f, 0.5f, 0.5f), float4(1, 1, 0, 1), float3(0, 1, 0), float2(0, 1)});

		indices.push_back(18);
		indices.push_back(17);
		indices.push_back(16);
		indices.push_back(16);
		indices.push_back(19);
		indices.push_back(18);
		Mesh top_mesh;
		top_mesh.name = "top_mesh";
		top_mesh.materialIdx = 4;

		top_mesh.startIndex = 24;
		top_mesh.numberOfVertices = 6;
		meshes.push_back(top_mesh);

		for (int i = 24; i < 30; i++)
		{
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}

		// Bottom
		tmp_vertices.push_back({float3(-0.5f, -0.5f, -0.5f), float4(1, 0, 0, 1), float3(0, -1, 0), float2(0, 0)});
		tmp_vertices.push_back({float3(0.5f, -0.5f, -0.5f), float4(0, 1, 0, 1), float3(0, -1, 0), float2(1, 0)});
		tmp_vertices.push_back({float3(0.5f, -0.5f, 0.5f), float4(0, 0, 1, 1), float3(0, -1, 0), float2(1, 1)});
		tmp_vertices.push_back({float3(-0.5f, -0.5f, 0.5f), float4(1, 1, 0, 1), float3(0, -1, 0), float2(0, 1)});

		indices.push_back(20);
		indices.push_back(21);
		indices.push_back(22);
		indices.push_back(22);
		indices.push_back(23);
		indices.push_back(20);

		Mesh bottom_mesh;
		bottom_mesh.name = "bottom_mesh";
		bottom_mesh.materialIdx = 5;

		bottom_mesh.startIndex = 30;
		bottom_mesh.numberOfVertices = 6;
		meshes.push_back(bottom_mesh);

		for (int i = 30; i < 36; i++)
		{
			vertex = tmp_vertices.at(indices.at(i));
			m_positions.push_back(vertex.pos);
			m_normals.push_back(vertex.normal);
			m_texture_coordinates.push_back(vertex.tex);
			vertecies.push_back(vertex);
		}

		float3 max_cords(m_positions.at(0));
		float3 min_cords(m_positions.at(0));
		for (size_t i = 1; i < m_positions.size(); i++)
		{
			max_cords = max_cords.Max(max_cords, m_positions.at(i));
			min_cords = max_cords.Min(min_cords, m_positions.at(i));
		}

		// TODO creatye materials for cube faces,
		//  posibly 6 diffwerent ones loaded from a file so that you can tweek and save changes
		std::vector<uint32_t> indecies;

		for (auto mesh : meshes)
		{
			for (size_t i = 0; i < mesh.numberOfVertices; i++)
				indecies.push_back(i + mesh.startIndex);
		}

		std::shared_ptr<Model> model = std::make_shared<Model>("Primative Cube", materials, meshes, false, max_cords, min_cords, vertecies, indecies);
		model->primativeType = PrimitiveModelType::CUBE;

		return model;
	}

	std::shared_ptr<Model> Model::createSphere(int stacks, int slices)
	{
		std::vector<Vertex> tmp_vertices{};
		std::vector<uint32_t> indices{};

		float radius = 1.0f;
		float sectorStep = 2 * 3.1415 / slices;
		float stackStep = 3.1415 / stacks;
		float sectorAngle, stackAngle;
		float4 cols[] = {
			{1, 0, 0, 1},
			{0, 1, 0, 1},
			{0, 0, 1, 1}};
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

				float4 col{normal.x, normal.y, normal.z, 1};
				tmp_vertices.push_back({float3(x, y, z), col, normal, uv});
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
		mesh.materialIdx = 0;
		mesh.name = "sphere_mesh";
		mesh.startIndex = 0;
		mesh.numberOfVertices = indices.size();
		std::vector<Mesh> meshes{mesh};

		std::vector<Vertex> vertecies;
		size_t idx;
		for (size_t i = 0; i < indices.size(); i++)
		{
			idx = indices.at(i);
			vertecies.push_back(tmp_vertices.at(idx));
		}
		std::vector<Material> materials;

		float3 max_cords(vertecies.at(0).pos);
		float3 min_cords(vertecies.at(0).pos);
		for (size_t i = 1; i < vertecies.size(); i++)
		{
			max_cords = max_cords.Max(max_cords, vertecies.at(0).pos);
			min_cords = max_cords.Min(min_cords, vertecies.at(0).pos);
		}

		std::vector<uint32_t> indecies;
		for (size_t i = 0; i < vertecies.size(); i++)
			indecies.push_back(i);

		std::shared_ptr<Model> model = std::make_shared<Model>("Primative Sphere", materials, meshes, false, max_cords, min_cords, vertecies, indecies);
		model->primativeType = PrimitiveModelType::SPHERE;

		return model;
	}

	std::shared_ptr<Model> Model::createPlane()
	{
		std::vector<Vertex> tmp_vertices{};
		std::vector<uint32_t> indices{};

		tmp_vertices.push_back({float3(-5.f, 0.0f, -5.f), float4(1, 0, 0, 1), float3(0, 1, 0), float2(0, 0)});
		tmp_vertices.push_back({float3(5.f, 0.0f, -5.f), float4(0, 1, 0, 1), float3(0, 1, 0), float2(1, 0)});
		tmp_vertices.push_back({float3(5.f, 0.0f, 5.f), float4(0, 0, 1, 1), float3(0, 1, 0), float2(1, 1)});
		tmp_vertices.push_back({float3(-5.f, 0.0f, 5.f), float4(1, 1, 0, 1), float3(0, 1, 0), float2(0, 1)});

		indices.push_back(0);
		indices.push_back(1);
		indices.push_back(2);
		indices.push_back(2);
		indices.push_back(3);
		indices.push_back(0);
		indices.push_back(2);
		indices.push_back(1);
		indices.push_back(0);
		indices.push_back(0);
		indices.push_back(3);
		indices.push_back(2);

		Mesh mesh;
		mesh.materialIdx = 0;
		mesh.name = "Plane mesh";
		mesh.startIndex = 0;
		mesh.numberOfVertices = indices.size();

		std::vector<Vertex> vertecies;
		size_t idx;
		for (size_t i = 0; i < indices.size(); i++)
		{
			idx = indices.at(i);
			vertecies.push_back(tmp_vertices.at(idx));
		}
		std::vector<Mesh> meshes{mesh};
		std::vector<Material> materials;

		float3 max_cords(vertecies.at(0).pos);
		float3 min_cords(vertecies.at(0).pos);
		for (size_t i = 1; i < vertecies.size(); i++)
		{
			max_cords = max_cords.Max(max_cords, vertecies.at(0).pos);
			min_cords = max_cords.Min(min_cords, vertecies.at(0).pos);
		}

		std::vector<uint32_t> indecies;
		for (size_t i = 0; i < vertecies.size(); i++)
			indecies.push_back(i);

		std::shared_ptr<Model> model = std::make_shared<Model>("Primative Plane", materials, meshes, false, max_cords, min_cords, vertecies, indecies);
		model->primativeType = PrimitiveModelType::PLANE;

		return model;
	}

}