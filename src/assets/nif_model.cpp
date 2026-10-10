#include "assets/nif_model.hpp"

#include <NifFile.hpp>
#include <ExtraData.hpp>
#include <Shaders.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <istream>
#include <stdexcept>
#include <unordered_map>

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace openf4::assets
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

        glm::mat4 convertTransform(const glm::mat4& bethesda)
        {
            glm::mat4 swap(1.0f);
            swap[1] = {0.0f, 0.0f, 1.0f, 0.0f};
            swap[2] = {0.0f, 1.0f, 0.0f, 0.0f};
            return swap * bethesda * swap;
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
        auto& header = file.GetHeader();
        for (std::uint32_t block = 0; block < header.GetNumBlocks(); ++block)
        {
            if (auto* parents = header.GetBlock<nifly::BSConnectPointParents>(block))
            {
                for (const auto& point : parents->connectPoints)
                {
                    glm::mat4 root(1.0f);
                    nifly::MatTransform rootTransform;
                    const std::string rootName = point.root.get();
                    if (!rootName.empty()
                        && file.GetNodeTransformToGlobal(rootName, rootTransform))
                        root = rootTransform.ToGLMMatrix<glm::mat4>();
                    glm::quat rotation(point.rotation.w, point.rotation.x,
                        point.rotation.y, point.rotation.z);
                    if (glm::length(rotation) <= 0.0f)
                        rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
                    else
                        rotation = glm::normalize(rotation);
                    glm::mat4 local = glm::translate(glm::mat4(1.0f),
                        glm::vec3(point.translation.x, point.translation.y,
                            point.translation.z))
                        * glm::mat4_cast(rotation);
                    local = glm::scale(local, glm::vec3(point.scale));
                    result.attachmentPoints.push_back(ModelGeometry::ConnectionPoint{
                        point.variableName.get(), convertTransform(root * local)});
                }
            }
            if (auto* children = header.GetBlock<nifly::BSConnectPointChildren>(block))
                for (const auto& target : children->targets)
                    result.attachmentTargets.push_back(target.get());
        }
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
            const auto* colors = file.GetColorsForShape(shape);
            const bool skinned = shape->IsSkinned();
            const nifly::MatTransform transform = skinned
                ? nifly::MatTransform{} : globalTransform(file, shape);
            std::vector<std::uint32_t> boneMap;
            std::vector<glm::uvec4> vertexBones(positions->size());
            std::vector<glm::vec4> vertexWeights(positions->size());
            if (skinned)
            {
                std::vector<std::string> boneNames;
                file.GetShapeBoneList(shape, boneNames);
                if (boneNames.empty())
                    throw std::runtime_error("Skinned NIF shape has no bone list");
                boneMap.reserve(boneNames.size());
                for (std::uint32_t localBone = 0; localBone < boneNames.size(); ++localBone)
                {
                    nifly::MatTransform skinToBone;
                    if (!file.GetShapeTransformSkinToBone(shape, localBone, skinToBone))
                        throw std::runtime_error("Skinned NIF shape has no bind transform");
                    const std::string& name = boneNames[localBone];
                    const auto existing = std::ranges::find_if(result.skinBones,
                        [&](const ModelSkinBone& bone) { return bone.name == name; });
                    std::uint32_t globalBone = 0;
                    if (existing == result.skinBones.end())
                    {
                        if (result.skinBones.size() >= 256)
                            throw std::runtime_error("Skinned NIF exceeds 256 bones");
                        globalBone = static_cast<std::uint32_t>(result.skinBones.size());
                        result.skinBones.push_back({name,
                            convertTransform(skinToBone.ToGLMMatrix<glm::mat4>())});
                    }
                    else
                        globalBone = static_cast<std::uint32_t>(existing - result.skinBones.begin());
                    boneMap.push_back(globalBone);

                    std::unordered_map<std::uint16_t, float> weights;
                    file.GetShapeBoneWeights(shape, localBone, weights);
                    for (const auto& [vertexIndex, weight] : weights)
                    {
                        if (vertexIndex >= positions->size() || !std::isfinite(weight) || weight <= 0.0f)
                            continue;
                        auto& values = vertexWeights[vertexIndex];
                        glm::length_t slot = 0;
                        for (glm::length_t candidate = 1; candidate < values.length(); ++candidate)
                            if (values[candidate] < values[slot])
                                slot = candidate;
                        if (weight > values[slot])
                        {
                            values[slot] = weight;
                            vertexBones[vertexIndex][slot] = globalBone;
                        }
                    }
                }
                for (std::size_t vertex = 0; vertex < vertexWeights.size(); ++vertex)
                {
                    const float sum = vertexWeights[vertex].x + vertexWeights[vertex].y
                        + vertexWeights[vertex].z + vertexWeights[vertex].w;
                    if (sum > 0.0f)
                        vertexWeights[vertex] /= sum;
                    else
                    {
                        vertexBones[vertex].x = boneMap.front();
                        vertexWeights[vertex].x = 1.0f;
                    }
                }
            }
            ModelSurface surface;
            surface.vertices.reserve(positions->size());
            if (nifly::NiShader* shader = file.GetShader(shape))
            {
                surface.materialPath = shader->name.get();
                surface.doubleSided = shader->IsDoubleSided();
                if (const auto* lighting = dynamic_cast<nifly::BSLightingShaderProperty*>(shader))
                {
                    const std::string rootMaterial = lighting->GetWetMaterialName();
                    if (!rootMaterial.empty())
                        surface.materialPath = rootMaterial;
                }
                if (const auto* property = dynamic_cast<const nifly::BSShaderProperty*>(shader))
                    surface.decal = (property->shaderFlags1
                        & (nifly::SLSF1_DECAL | nifly::SLSF1_DYNAMIC_DECAL)) != 0;
            }
            file.GetTextureSlot(shape, surface.diffuseTexture, 0);
            file.GetTextureSlot(shape, surface.normalTexture, 1);
            if (const auto* alpha = file.GetAlphaProperty(shape))
            {
                surface.alphaBlend = (alpha->flags & 1) != 0;
                if ((alpha->flags & 0x0200) != 0)
                    surface.alphaCutoff = alpha->threshold / 255.0f;
            }
            for (std::size_t index = 0; index < positions->size(); ++index)
            {
                const auto position = transform.ApplyTransform((*positions)[index]);
                glm::vec3 normal{};
                if (normals && index < normals->size())
                    normal = glm::normalize(convert(transform.ApplyTransformToDir((*normals)[index])));
                glm::vec2 uv{};
                if (uvs && index < uvs->size())
                    uv = {(*uvs)[index].u, (*uvs)[index].v};
                render::SourceVertex vertex;
                vertex.position = convert(position);
                vertex.normal = normal;
                vertex.uv = uv;
                if (skinned)
                {
                    vertex.boneIndices = vertexBones[index];
                    vertex.boneWeights = vertexWeights[index];
                }
                if (colors && index < colors->size())
                {
                    const auto& color = (*colors)[index];
                    vertex.color = {color.r, color.g, color.b, color.a};
                }
                surface.vertices.push_back(vertex);
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

    std::vector<glm::mat4> sampleAnimation(
        const ModelGeometry& geometry, const ModelAnimationClip& clip, float time)
    {
        // Normalize time to the clip duration
        float normalizedTime = time;
        if (clip.duration > 0.0f)
            normalizedTime = std::fmod(time, clip.duration);
        if (normalizedTime < 0.0f)
            normalizedTime += clip.duration;

        // Start with bind pose transforms
        std::vector<glm::mat4> transforms = bindPoseTransforms(geometry);

        // For each bone that has animation data, compute the animated transform
        for (const auto& channel : clip.channels)
        {
            auto boneIt = std::ranges::find_if(geometry.skinBones,
                [&](const ModelSkinBone& bone) { return bone.name == channel.boneName; });
            
            if (boneIt == geometry.skinBones.end())
                continue;

            std::size_t boneIndex = static_cast<std::size_t>(boneIt - geometry.skinBones.begin());

            // Find the two keyframes that bracket this time
            std::size_t idx1 = 0;
            std::size_t idx2 = 0;
            
            if (!channel.times.empty())
            {
                // Binary search or linear search for the bracketing keyframes
                for (std::size_t i = 0; i < channel.times.size(); ++i)
                {
                    if (channel.times[i] <= normalizedTime)
                        idx1 = i;
                }
                
                idx2 = (idx1 + 1) % channel.times.size();
                
                // Compute interpolation factor
                float t = 0.0f;
                if (idx2 != idx1)
                {
                    float t1 = channel.times[idx1];
                    float t2 = channel.times[idx2];
                    if (t2 > t1)
                        t = (normalizedTime - t1) / (t2 - t1);
                }
                
                // Interpolate position
                glm::vec3 position = channel.positions[idx1];
                if (idx2 != idx1)
                    position = glm::mix(channel.positions[idx1], channel.positions[idx2], t);
                
                // Interpolate rotation (SLERP)
                glm::quat rotation = channel.rotations[idx1];
                if (idx2 != idx1)
                    rotation = glm::slerp(channel.rotations[idx1], channel.rotations[idx2], t);
                
                // Interpolate scale
                glm::vec3 scale = channel.scales[idx1];
                if (idx2 != idx1)
                    scale = glm::mix(channel.scales[idx1], channel.scales[idx2], t);
                
                // Build the local transform
                glm::mat4 localTransform = glm::translate(glm::mat4(1.0f), position)
                    * glm::mat4_cast(rotation)
                    * glm::scale(glm::mat4(1.0f), scale);
                
                // The skinToBone transform converts from skin space to bone space
                // We need the inverse: bone space to skin space
                transforms[boneIndex] = localTransform * geometry.skinBones[boneIndex].skinToBone;
            }
        }

        return transforms;
    }

    std::vector<glm::mat4> bindPoseTransforms(const ModelGeometry& geometry)
    {
        std::vector<glm::mat4> transforms;
        transforms.reserve(geometry.skinBones.size());
        
        for (const auto& bone : geometry.skinBones)
        {
            // In bind pose, the bone transform is the inverse of the skinToBone transform
            transforms.push_back(glm::inverse(bone.skinToBone));
        }
        
        return transforms;
    }

}
