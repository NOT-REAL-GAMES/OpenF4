#include "assets/nif_model.hpp"

#include <NifFile.hpp>
#include <Shaders.hpp>

#include <algorithm>
#include <cctype>
#include <istream>
#include <stdexcept>

namespace empire::assets
{
    namespace
    {
        class SpanBuffer final : public std::streambuf
        {
        public:
            explicit SpanBuffer(std::span<const std::byte> bytes)
            {
                char* begin = reinterpret_cast<char*>(const_cast<std::byte*>(bytes.data()));
                setg(begin, begin, begin + bytes.size());
            }
        };

        nifly::MatTransform globalTransform(nifly::NifFile& file, nifly::NiShape* shape)
        {
            nifly::MatTransform result = shape->GetTransformToParent();
            for (nifly::NiNode* parent = file.GetParentNode(shape); parent;
                 parent = file.GetParentNode(parent))
                result = parent->GetTransformToParent().ComposeTransforms(result);
            return result;
        }

        glm::vec3 convert(const nifly::Vector3& value)
        {
            return {value.x, value.z, value.y};
        }

        bool renderableShape(const nifly::NiShape& shape)
        {
            std::string name = shape.name.get();
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            return name.find("editor") == std::string::npos
                && name.find("marker") == std::string::npos
                && name.find("collision") == std::string::npos
                && name.find("bhk") == std::string::npos
                && name.find("shadow") == std::string::npos;
        }
    }

    ModelGeometry loadNifModel(std::span<const std::byte> bytes)
    {
        SpanBuffer buffer(bytes);
        std::istream stream(&buffer);
        nifly::NifFile file(stream);
        if (!file.IsValid())
            throw std::runtime_error("NIF parser rejected the model");

        ModelGeometry result;
        for (nifly::NiShape* shape : file.GetShapes())
        {
            if (!renderableShape(*shape))
                continue;
            const auto* positions = file.GetVertsForShape(shape);
            if (!positions || positions->empty())
                continue;
            std::vector<nifly::Triangle> triangles;
            if (!shape->GetTriangles(triangles) || triangles.empty())
                continue;
            const auto* normals = file.GetNormalsForShape(shape);
            const auto* uvs = file.GetUvsForShape(shape);
            const nifly::MatTransform transform = globalTransform(file, shape);
            ModelSurface surface;
            surface.vertices.reserve(positions->size());
            if (nifly::NiShader* shader = file.GetShader(shape))
            {
                surface.materialPath = shader->name.get();
                if (const auto* lighting = dynamic_cast<nifly::BSLightingShaderProperty*>(shader))
                {
                    const std::string rootMaterial = lighting->GetWetMaterialName();
                    if (!rootMaterial.empty())
                        surface.materialPath = rootMaterial;
                }
            }
            file.GetTextureSlot(shape, surface.diffuseTexture, 0);
            file.GetTextureSlot(shape, surface.normalTexture, 1);
            if (const auto* alpha = file.GetAlphaProperty(shape); alpha && (alpha->flags & 0x0200) != 0)
                surface.alphaCutoff = alpha->threshold / 255.0f;
            for (std::size_t index = 0; index < positions->size(); ++index)
            {
                const auto position = transform.ApplyTransform((*positions)[index]);
                glm::vec3 normal{};
                if (normals && index < normals->size())
                    normal = glm::normalize(convert(transform.ApplyTransformToDir((*normals)[index])));
                glm::vec2 uv{};
                if (uvs && index < uvs->size())
                    uv = {(*uvs)[index].u, (*uvs)[index].v};
                surface.vertices.push_back({convert(position), normal, uv});
            }
            surface.indices.reserve(triangles.size() * 3);
            for (const auto& triangle : triangles)
            {
                if (triangle.p1 >= positions->size() || triangle.p2 >= positions->size()
                    || triangle.p3 >= positions->size())
                    continue;
                surface.indices.insert(surface.indices.end(), {
                    triangle.p1, triangle.p2, triangle.p3});
            }

            if (!normals || normals->size() < positions->size())
            {
                for (std::size_t index = 0; index + 2 < surface.indices.size(); index += 3)
                {
                    const std::uint32_t a = surface.indices[index];
                    const std::uint32_t b = surface.indices[index + 1];
                    const std::uint32_t c = surface.indices[index + 2];
                    const glm::vec3 normal = glm::cross(
                        surface.vertices[b].position - surface.vertices[a].position,
                        surface.vertices[c].position - surface.vertices[a].position);
                    surface.vertices[a].normal += normal;
                    surface.vertices[b].normal += normal;
                    surface.vertices[c].normal += normal;
                }
                for (auto& vertex : surface.vertices)
                {
                    const float length = glm::length(vertex.normal);
                    vertex.normal = length > 0.0f
                        ? vertex.normal / length : glm::vec3(0.0f, 1.0f, 0.0f);
                }
            }
            if (!surface.indices.empty())
                result.surfaces.push_back(std::move(surface));
        }
        if (result.surfaces.empty())
            throw std::runtime_error("NIF contains no renderable triangle geometry");
        return result;
    }
}
