#pragma once

#include "render/mesh_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>

namespace openf4::assets
{
    // The renderer mirrors X while drawing world geometry.  Imported glTF is
    // mirrored here so the two reflections preserve its right-handed winding.
    inline constexpr float gltfBethesdaUnitsPerMeter = 69.99125f;

    enum class ModelImageEncoding
    {
        rgba8,
        dds
    };

    // Renderer-neutral image payload.  rgba8 pixels are tightly packed RGBA8;
    // DDS payloads use the existing DdsTexture pixel/mip layout and dxgiFormat.
    struct ModelImage
    {
        ModelImageEncoding encoding = ModelImageEncoding::rgba8;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t mipLevels = 1;
        std::uint32_t dxgiFormat = 0;
        bool srgb = false;
        std::vector<std::byte> pixels;
        std::vector<std::uint32_t> mipOffsets;
    };

    struct ModelSurface
    {
        std::vector<render::SourceVertex> vertices;
        std::vector<std::uint32_t> indices;
        std::string materialPath;
        std::string diffuseTexture;
        std::string normalTexture;
        glm::vec4 baseColor{1.0f};
        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
        float alphaCutoff = 0.0f;
        bool alphaBlend = false;
        bool doubleSided = false;
        bool decal = false;
        // Tangents stay in source precision until a tangent-space renderer path
        // consumes them.  xyz is direction and w is the bitangent sign.
        std::vector<glm::vec4> tangents;
        std::optional<ModelImage> diffuseImage;
        std::optional<ModelImage> normalImage;
        std::optional<ModelImage> metallicRoughnessImage;
    };

    struct ModelSkinBone
    {
        std::string name;
        glm::mat4 skinToBone{1.0f};
    };

    // A single animated channel for one bone.
    struct ModelAnimationChannel
    {
        std::string boneName;
        // Keyframe times in seconds.
        std::vector<float> times;
        // Corresponding positions, rotations, and scales.
        std::vector<glm::vec3> positions;
        std::vector<glm::quat> rotations;
        std::vector<glm::vec3> scales;
    };

    // An animation clip containing channels for all bones.
    struct ModelAnimationClip
    {
        std::string name;
        float duration = 0.0f;
        float ticksPerSecond = 24.0f;
        std::vector<ModelAnimationChannel> channels;
        // Maps bone name to channel index for fast lookup.
        std::unordered_map<std::string, std::size_t> boneIndex;
    };

    struct ModelGeometry
    {
        std::vector<ModelSurface> surfaces;
        std::vector<ModelSkinBone> skinBones;
        std::vector<ModelAnimationClip> animations;
        struct ConnectionPoint
        {
            std::string name;
            glm::mat4 transform{1.0f};
        };
        std::vector<ConnectionPoint> attachmentPoints;
        std::vector<std::string> attachmentTargets;
    };

    [[nodiscard]] ModelGeometry loadNifModel(std::span<const std::byte> bytes);

    // Sample an animation clip at a given time and return bone transforms.
    // The returned vector is indexed by the bone's position in skinBones.
    [[nodiscard]] std::vector<glm::mat4> sampleAnimation(
        const ModelGeometry& geometry, const ModelAnimationClip& clip, float time);

    // Compute the bind-pose transforms for a geometry with skin bones.
    [[nodiscard]] std::vector<glm::mat4> bindPoseTransforms(
        const ModelGeometry& geometry);
}
