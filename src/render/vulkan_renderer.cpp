#include "render/vulkan_renderer.hpp"

#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace openf4::render
{
    namespace
    {
        constexpr std::uint32_t frameCount = 2;

        const char* computeShader = R"(
#version 460
layout(local_size_x = 64) in;

struct Vertex { vec4 position; vec3 normal; uint blendWeights; vec2 texCoord; uint blendWeightsExtra; uint color; uint boneIndices; uint boneWeights; uvec2 padding; };
struct Meshlet { uint vertexOffset; uint indexOffset; uint vertexCount; uint triangleCount; vec4 sphere; uint materialIndex; uint padding0; uint padding1; uint padding2; };
struct Instance { mat4 transform; vec4 bounds; vec4 lodRange; uvec4 meshletRange; uvec4 skinRange; };

layout(std430, binding = 0) readonly buffer Vertices { Vertex vertices[]; };
layout(std430, binding = 1) readonly buffer Indices { uint indices[]; };
layout(std430, binding = 2) readonly buffer Meshlets { Meshlet meshlets[]; };
layout(std430, binding = 3) readonly buffer Instances { Instance instances[]; };
layout(std430, binding = 4) readonly buffer Sources { uvec2 sources[]; };
layout(std430, binding = 5) writeonly buffer Visible { uvec2 visible[]; };
layout(std430, binding = 6) buffer Indirect { coherent uint command[]; };
layout(std140, binding = 7) uniform Camera {
    mat4 viewProjection;
    mat4 lightViewProjection;
    vec4 planes[6];
    vec4 cameraPosition;
    vec4 worldOrigin;
    uvec4 counts;
} camera;

void main()
{
    uint sourceIndex = gl_GlobalInvocationID.x;
    if (sourceIndex >= camera.counts.x)
        return;

    uvec2 source = sources[sourceIndex];
    Instance instance = instances[source.x];
    if (instance.meshletRange.z == 0)
        return;
    Meshlet meshlet = meshlets[source.y];
    if (instance.meshletRange.w == 0) {
        vec3 center;
        float radius;
        if (instance.skinRange.y != 0) {
            center = instance.bounds.xyz - camera.worldOrigin.xyz;
            radius = instance.bounds.w;
        } else {
            center = (instance.transform * vec4(meshlet.sphere.xyz, 1.0)).xyz;
            center.x = -center.x;
            center -= camera.worldOrigin.xyz;
            float scale = max(length(instance.transform[0].xyz),
                max(length(instance.transform[1].xyz), length(instance.transform[2].xyz)));
            radius = meshlet.sphere.w * scale;
        }
        for (uint plane = 0; plane < 6; ++plane) {
            if (dot(camera.planes[plane].xyz, center) + camera.planes[plane].w < -radius)
                return;
        }

        float distanceToCamera = distance(camera.cameraPosition.xz, center.xz);
        if (distanceToCamera + radius < instance.lodRange.x
            || distanceToCamera - radius > instance.lodRange.w)
            return;
    }

    uint destination = atomicAdd(command[0], 1);
    if (destination < camera.counts.y)
        visible[destination] = source;
}
)";

        const char* meshShader = R"(
#version 460
#extension GL_EXT_mesh_shader : require

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

struct Vertex { vec4 position; vec3 normal; uint blendWeights; vec2 texCoord; uint blendWeightsExtra; uint color; uint boneIndices; uint boneWeights; uvec2 padding; };
struct Meshlet { uint vertexOffset; uint indexOffset; uint vertexCount; uint triangleCount; vec4 sphere; uint materialIndex; uint padding0; uint padding1; uint padding2; };
struct Instance { mat4 transform; vec4 bounds; vec4 lodRange; uvec4 meshletRange; uvec4 skinRange; };
struct Material { vec4 baseColor; uvec4 textureIndices; vec4 parameters; uvec4 extraTextureIndices; };

layout(constant_id = 0) const uint decalPass = 0;

layout(std430, binding = 0) readonly buffer Vertices { Vertex vertices[]; };
layout(std430, binding = 1) readonly buffer Indices { uint indices[]; };
layout(std430, binding = 2) readonly buffer Meshlets { Meshlet meshlets[]; };
layout(std430, binding = 3) readonly buffer Instances { Instance instances[]; };
layout(std430, binding = 4) readonly buffer Sources { uvec2 sources[]; };
layout(std430, binding = 5) readonly buffer Visible { uvec2 visible[]; };
layout(std430, binding = 6) readonly buffer Indirect { uint command[]; };
layout(std140, binding = 7) uniform Camera {
    mat4 viewProjection;
    mat4 lightViewProjection;
    vec4 planes[6];
    vec4 cameraPosition;
    vec4 worldOrigin;
    uvec4 counts;
} camera;
layout(std430, binding = 9) readonly buffer Materials { Material materials[]; };
layout(std430, binding = 10) readonly buffer SkinMatrices { mat4 skinMatrices[]; };

layout(location = 0) out vec3 fragmentNormal[];
layout(location = 1) out flat uint fragmentInstance[];
layout(location = 2) out vec3 fragmentWorldPosition[];
layout(location = 3) out vec2 fragmentTexCoord[];
layout(location = 4) out flat uint fragmentMaterial[];
layout(location = 5) out vec4 fragmentBlendWeights[];
layout(location = 6) out vec4 fragmentBlendWeightsExtra[];
layout(location = 7) out vec4 fragmentColor[];

void main()
{
    uvec2 candidate = visible[gl_WorkGroupID.x];
    Instance instance = instances[candidate.x];
    Meshlet meshlet = meshlets[candidate.y];
    bool decal = materials[meshlet.materialIndex].parameters.w > 0.5;
    if (decalPass == 2 ? (instance.skinRange.z == 0 || instance.meshletRange.w != 0)
                      : decal != (decalPass != 0)) {
        SetMeshOutputsEXT(0, 0);
        return;
    }
    SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);

    for (uint vertexIndex = gl_LocalInvocationIndex;
         vertexIndex < meshlet.vertexCount; vertexIndex += gl_WorkGroupSize.x) {
        Vertex vertex = vertices[meshlet.vertexOffset + vertexIndex];
        vec4 localPosition = vertex.position;
        vec3 localNormal = vertex.normal;
        vec4 skinWeights = unpackUnorm4x8(vertex.boneWeights);
        float skinWeightSum = dot(skinWeights, vec4(1.0));
        if (instance.skinRange.y != 0 && skinWeightSum > 0.0) {
            skinWeights /= skinWeightSum;
            uvec4 joints = uvec4(vertex.boneIndices & 255u,
                (vertex.boneIndices >> 8) & 255u,
                (vertex.boneIndices >> 16) & 255u,
                (vertex.boneIndices >> 24) & 255u);
            joints = min(joints, uvec4(instance.skinRange.y - 1u));
            uint first = instance.skinRange.x;
            mat4 skin = skinMatrices[first + joints.x] * skinWeights.x
                + skinMatrices[first + joints.y] * skinWeights.y
                + skinMatrices[first + joints.z] * skinWeights.z
                + skinMatrices[first + joints.w] * skinWeights.w;
            localPosition = skin * vertex.position;
            localNormal = mat3(skin) * vertex.normal;
        }
        vec4 world = instance.transform * localPosition;
        vec3 worldNormal = localNormal;
        if (instance.meshletRange.w != 0) {
            vec2 viewportSize = max(vec2(camera.counts.zw), vec2(1.0));
            vec2 clip = vec2(vertex.position.x / viewportSize.x * 2.0 - 1.0,
                1.0 - vertex.position.y / viewportSize.y * 2.0);
            gl_MeshVerticesEXT[vertexIndex].gl_Position =
                vec4(clip, vertex.position.z, 1.0);
        } else {
            world.x = -world.x;
            world.xyz -= camera.worldOrigin.xyz;
            gl_MeshVerticesEXT[vertexIndex].gl_Position = camera.viewProjection * world;
            worldNormal = mat3(instance.transform) * localNormal;
            worldNormal.x = -worldNormal.x;
        }
        fragmentNormal[vertexIndex] = normalize(worldNormal);
        fragmentInstance[vertexIndex] = candidate.x;
        fragmentWorldPosition[vertexIndex] = world.xyz;
        fragmentTexCoord[vertexIndex] = vertex.texCoord;
        fragmentMaterial[vertexIndex] = meshlet.materialIndex;
        fragmentBlendWeights[vertexIndex] = unpackUnorm4x8(vertex.blendWeights);
        fragmentBlendWeightsExtra[vertexIndex] = unpackUnorm4x8(vertex.blendWeightsExtra);
        fragmentColor[vertexIndex] = unpackUnorm4x8(vertex.color);
    }
    for (uint primitiveIndex = gl_LocalInvocationIndex;
         primitiveIndex < meshlet.triangleCount; primitiveIndex += gl_WorkGroupSize.x) {
        uint first = meshlet.indexOffset + primitiveIndex * 3;
        gl_PrimitiveTriangleIndicesEXT[primitiveIndex] =
            uvec3(indices[first], indices[first + 1], indices[first + 2]);
    }
}
)";

        const char* fragmentShader = R"(
#version 460
#extension GL_EXT_nonuniform_qualifier : require

struct Instance { mat4 transform; vec4 bounds; vec4 lodRange; uvec4 meshletRange; uvec4 skinRange; };
struct Material { vec4 baseColor; uvec4 textureIndices; vec4 parameters; uvec4 extraTextureIndices; };
layout(std430, binding = 3) readonly buffer Instances { Instance instances[]; };
layout(std430, binding = 9) readonly buffer Materials { Material materials[]; };
layout(std140, binding = 7) uniform Camera {
    mat4 viewProjection;
    mat4 lightViewProjection;
    vec4 planes[6];
    vec4 cameraPosition;
    vec4 worldOrigin;
    uvec4 counts;
} camera;
layout(binding = 8) uniform sampler2DShadow shadowMap;
layout(binding = 11) uniform sampler2D diffuseTextures[];
layout(constant_id = 1) const uint selectionPass = 0;

layout(location = 0) in vec3 fragmentNormal;
layout(location = 1) in flat uint fragmentInstance;
layout(location = 2) in vec3 fragmentWorldPosition;
layout(location = 3) in vec2 fragmentTexCoord;
layout(location = 4) in flat uint fragmentMaterial;
layout(location = 5) in vec4 fragmentBlendWeights;
layout(location = 6) in vec4 fragmentBlendWeightsExtra;
layout(location = 7) in vec4 fragmentColor;
layout(location = 0) out vec4 outputColor;

float interleavedGradientNoise(vec2 pixel)
{
    return fract(52.9829189 * fract(
        0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

void main()
{
    Instance instance = instances[fragmentInstance];
    Material material = materials[fragmentMaterial];
    if (instance.meshletRange.w != 0) {
        outputColor = material.baseColor;
        return;
    }
    float distanceToCamera = distance(
        camera.cameraPosition.xz, fragmentWorldPosition.xz);
    float fadeIn = instance.lodRange.y <= instance.lodRange.x
        ? step(instance.lodRange.x, distanceToCamera)
        : smoothstep(instance.lodRange.x, instance.lodRange.y, distanceToCamera);
    float fadeOut = instance.lodRange.w <= instance.lodRange.z
        ? 1.0 - step(instance.lodRange.z, distanceToCamera)
        : 1.0 - smoothstep(instance.lodRange.z, instance.lodRange.w, distanceToCamera);
    float noise = interleavedGradientNoise(gl_FragCoord.xy);
    if ((fadeIn < 1.0 && noise < 1.0 - fadeIn)
        || (fadeOut < 1.0 && noise > fadeOut))
        discard;

    vec3 normal = normalize(fragmentNormal);
    vec3 lightDirection = normalize(vec3(0.35, 0.8, 0.25));
    vec4 lightClip = camera.lightViewProjection * vec4(fragmentWorldPosition, 1.0);
    vec3 lightUv = lightClip.xyz / lightClip.w;
    lightUv.xy = lightUv.xy * 0.5 + 0.5;
    float shadow = 1.0;
    if (all(greaterThanEqual(lightUv, vec3(0.0))) && all(lessThanEqual(lightUv, vec3(1.0)))) {
        vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
        shadow = 0.0;
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
                shadow += texture(shadowMap,
                    vec3(lightUv.xy + vec2(x, y) * texel, lightUv.z - 0.0015));
        shadow /= 9.0;
    }
    float diffuse = max(dot(normal, lightDirection), 0.0) * mix(0.3, 1.0, shadow);
    bool terrain = material.parameters.z > 0.5 && material.parameters.z < 1.5;
    bool terrainLod = material.parameters.z > 1.5 && material.parameters.z < 2.5;
    bool water = material.parameters.z > 2.5;
    vec4 textureSample;
    if (water) {
        textureSample = vec4(1.0);
    } else if (terrain) {
        vec4 weights = max(fragmentBlendWeights, vec4(0.0));
        vec4 extraWeights = max(fragmentBlendWeightsExtra, vec4(0.0));
        float weightSum = dot(weights, vec4(1.0)) + dot(extraWeights, vec4(1.0));
        if (weightSum > 0.0001) {
            weights /= weightSum;
            extraWeights /= weightSum;
        } else {
            weights = vec4(1.0, 0.0, 0.0, 0.0);
        }
        textureSample =
            texture(diffuseTextures[nonuniformEXT(material.textureIndices.x)], fragmentTexCoord) * weights.x
            + texture(diffuseTextures[nonuniformEXT(material.textureIndices.y)], fragmentTexCoord) * weights.y
            + texture(diffuseTextures[nonuniformEXT(material.textureIndices.z)], fragmentTexCoord) * weights.z
            + texture(diffuseTextures[nonuniformEXT(material.textureIndices.w)], fragmentTexCoord) * weights.w
            + texture(diffuseTextures[nonuniformEXT(material.extraTextureIndices.x)], fragmentTexCoord) * extraWeights.x
            + texture(diffuseTextures[nonuniformEXT(material.extraTextureIndices.y)], fragmentTexCoord) * extraWeights.y
            + texture(diffuseTextures[nonuniformEXT(material.extraTextureIndices.z)], fragmentTexCoord) * extraWeights.z
            + texture(diffuseTextures[nonuniformEXT(material.extraTextureIndices.w)], fragmentTexCoord) * extraWeights.w;
    } else {
        textureSample = texture(
            diffuseTextures[nonuniformEXT(material.textureIndices.x)], fragmentTexCoord);
    }
    float opacity = (terrainLod || water)
        ? 1.0 : textureSample.a * material.baseColor.a * fragmentColor.a;
    if (material.parameters.y > 0.0 && opacity < material.parameters.y)
        discard;
    if (selectionPass != 0) {
        if (opacity <= 0.001)
            discard;
        outputColor = vec4(float(instance.skinRange.z) * 0.5, 0.0, 0.0, 1.0);
        return;
    }
    vec3 sampled = textureSample.rgb;
    if (!terrain && material.textureIndices.w != 0u && material.textureIndices.z != 0u) {
        float gray = dot(sampled, vec3(0.333333));
        sampled = textureLod(diffuseTextures[nonuniformEXT(material.textureIndices.z)],
            vec2(gray, 1.0 - material.parameters.x), 0.0).rgb;
    }
    vec3 base = sampled * material.baseColor.rgb * fragmentColor.rgb;
    vec3 color;
    if (water) {
        vec3 viewDirection = normalize(camera.cameraPosition.xyz - fragmentWorldPosition);
        vec3 absolutePosition = fragmentWorldPosition + camera.worldOrigin.xyz;
        vec3 waterNormal = normalize(vec3(
            sin(absolutePosition.x * 0.0021) * 0.06,
            1.0,
            cos(absolutePosition.z * 0.0017) * 0.06));
        float fresnel = pow(1.0 - clamp(dot(waterNormal, viewDirection), 0.0, 1.0), 5.0);
        float highlight = pow(max(dot(reflect(-lightDirection, waterNormal),
            viewDirection), 0.0), 96.0);
        color = mix(vec3(0.025, 0.10, 0.14), vec3(0.28, 0.46, 0.58),
            0.12 + fresnel * 0.78) + highlight * 0.45;
    } else {
        color = base * (terrainLod
            ? 0.1 + max(dot(normal, lightDirection), 0.0)
            : 0.18 + diffuse * 0.82);
    }
    if (terrainLod)
        color *= color * color + 1.0;
    outputColor = vec4(color, opacity);
}
)";

        const char* outlineVertexShader = R"(
#version 460
void main()
{
    vec2 corner = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
)";

        const char* outlineFragmentShader = R"(
#version 460
layout(binding = 0) uniform sampler2D selectionMask;
layout(location = 0) out vec4 outputColor;
void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float selected = texelFetch(selectionMask, pixel, 0).r;
    if (selected == 0.0)
        discard;
    ivec2 size = textureSize(selectionMask, 0);
    bool edge = false;
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x) {
            if (x*x + y*y > 4)
                continue;
            ivec2 neighbor = pixel + ivec2(x, y);
            // The outline is two pixels inward, so it cannot bleed onto an occluder
            // or outside the editor viewport, even at a partially hidden silhouette.
            if (all(greaterThanEqual(neighbor, ivec2(0))) && all(lessThan(neighbor, size)))
                edge = edge || texelFetch(selectionMask, neighbor, 0).r != selected;
        }
    if (!edge)
        discard;
    outputColor = selected > 0.75 ? vec4(1.0, 0.55, 0.05, 1.0)
                                  : vec4(0.0, 0.85, 1.0, 1.0);
}
)";

        const char* shadowMeshShader = R"(
#version 460
#extension GL_EXT_mesh_shader : require

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

struct Vertex { vec4 position; vec3 normal; uint blendWeights; vec2 texCoord; uint blendWeightsExtra; uint color; uint boneIndices; uint boneWeights; uvec2 padding; };
struct Meshlet { uint vertexOffset; uint indexOffset; uint vertexCount; uint triangleCount; vec4 sphere; uint materialIndex; uint padding0; uint padding1; uint padding2; };
struct Instance { mat4 transform; vec4 bounds; vec4 lodRange; uvec4 meshletRange; uvec4 skinRange; };
layout(std430, binding = 0) readonly buffer Vertices { Vertex vertices[]; };
layout(std430, binding = 1) readonly buffer Indices { uint indices[]; };
layout(std430, binding = 2) readonly buffer Meshlets { Meshlet meshlets[]; };
layout(std430, binding = 3) readonly buffer Instances { Instance instances[]; };
layout(std430, binding = 5) readonly buffer Visible { uvec2 visible[]; };
layout(std140, binding = 7) uniform Camera {
    mat4 viewProjection;
    mat4 lightViewProjection;
    vec4 planes[6];
    vec4 cameraPosition;
    vec4 worldOrigin;
    uvec4 counts;
} camera;
layout(std430, binding = 10) readonly buffer SkinMatrices { mat4 skinMatrices[]; };

void main()
{
    uvec2 candidate = visible[gl_WorkGroupID.x];
    Instance instance = instances[candidate.x];
    Meshlet meshlet = meshlets[candidate.y];
    if (instance.meshletRange.w != 0) {
        SetMeshOutputsEXT(0, 0);
        return;
    }
    SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);
    for (uint vertexIndex = gl_LocalInvocationIndex;
         vertexIndex < meshlet.vertexCount; vertexIndex += gl_WorkGroupSize.x) {
        Vertex vertex = vertices[meshlet.vertexOffset + vertexIndex];
        vec4 localPosition = vertex.position;
        vec4 skinWeights = unpackUnorm4x8(vertex.boneWeights);
        float skinWeightSum = dot(skinWeights, vec4(1.0));
        if (instance.skinRange.y != 0 && skinWeightSum > 0.0) {
            skinWeights /= skinWeightSum;
            uvec4 joints = uvec4(vertex.boneIndices & 255u,
                (vertex.boneIndices >> 8) & 255u,
                (vertex.boneIndices >> 16) & 255u,
                (vertex.boneIndices >> 24) & 255u);
            joints = min(joints, uvec4(instance.skinRange.y - 1u));
            uint first = instance.skinRange.x;
            mat4 skin = skinMatrices[first + joints.x] * skinWeights.x
                + skinMatrices[first + joints.y] * skinWeights.y
                + skinMatrices[first + joints.z] * skinWeights.z
                + skinMatrices[first + joints.w] * skinWeights.w;
            localPosition = skin * vertex.position;
        }
        vec4 world = instance.transform * localPosition;
        world.x = -world.x;
        world.xyz -= camera.worldOrigin.xyz;
        gl_MeshVerticesEXT[vertexIndex].gl_Position = camera.lightViewProjection * world;
    }
    for (uint primitiveIndex = gl_LocalInvocationIndex;
         primitiveIndex < meshlet.triangleCount; primitiveIndex += gl_WorkGroupSize.x) {
        uint first = meshlet.indexOffset + primitiveIndex * 3;
        gl_PrimitiveTriangleIndicesEXT[primitiveIndex] =
            uvec3(indices[first], indices[first + 1], indices[first + 2]);
    }
}
)";

        void check(VkResult result, std::string_view operation)
        {
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::string(operation) + " failed with Vulkan error "
                    + std::to_string(result));
        }

        VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment)
        {
            return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
        }

        struct Buffer
        {
            VkBuffer handle = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkDeviceSize size = 0;
            void* mapped = nullptr;
        };

        struct DepthTarget
        {
            VkImage image = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            bool initialized = false;
        };

        struct TextureTarget
        {
            VkImage image = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkDeviceMemory sparseMemory = VK_NULL_HANDLE;
            std::vector<VkSparseImageMemoryBind> imageBinds;
            std::vector<VkSparseMemoryBind> opaqueBinds;
            VkDeviceSize virtualBytes = 0;
            VkDeviceSize residentBytes = 0;
            std::uint32_t firstResidentMip = 0;
            bool sparse = false;
            bool sparseBound = false;
        };

        struct alignas(16) GpuInstance
        {
            glm::mat4 transform{1.0f};
            glm::vec4 bounds{};
            glm::vec4 lodRange{};
            glm::uvec4 meshletRange{};
            glm::uvec4 skinRange{};
        };

        struct alignas(16) GpuMaterial
        {
            glm::vec4 baseColor{1.0f};
            glm::uvec4 textureIndices{};
            glm::vec4 parameters{};
            glm::uvec4 extraTextureIndices{};
        };

        struct alignas(16) CameraData
        {
            glm::mat4 viewProjection{1.0f};
            glm::mat4 lightViewProjection{1.0f};
            std::array<glm::vec4, 6> planes{};
            glm::vec4 cameraPosition{};
            glm::vec4 worldOrigin{};
            glm::uvec4 counts{};
        };

        static_assert(sizeof(GpuVertex) == 64);
        static_assert(sizeof(GpuMeshlet) == 48);
        static_assert(sizeof(GpuInstance) == 128);
        static_assert(sizeof(GpuMaterial) == 64);
        static_assert(sizeof(CameraData) == 272);

        static GpuMaterial makeGpuMaterial(const SceneMaterial& material)
        {
            if (material.terrain)
                return {material.baseColor, material.terrainTextures,
                    glm::vec4(0.0f, 0.0f, 1.0f, 0.0f), material.terrainTexturesExtra};
            return {material.baseColor,
                glm::uvec4(material.diffuseTexture, material.normalTexture,
                    material.grayscaleTexture, material.grayscaleToPalette ? 1u : 0u),
                glm::vec4(material.grayscaleToPaletteScale, material.alphaCutoff,
                    material.water ? 3.0f : (material.terrainLod ? 2.0f : 0.0f),
                    material.decal ? 1.0f : 0.0f), {}};
        }

        std::vector<std::uint32_t> compileShader(
            std::string_view source, shaderc_shader_kind kind, const char* name)
        {
            shaderc::Compiler compiler;
            shaderc::CompileOptions options;
            options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
            options.SetOptimizationLevel(shaderc_optimization_level_performance);
            const auto result = compiler.CompileGlslToSpv(
                source.data(), source.size(), kind, name, options);
            if (result.GetCompilationStatus() != shaderc_compilation_status_success)
                throw std::runtime_error(std::string(name) + ": " + result.GetErrorMessage());
            return {result.cbegin(), result.cend()};
        }

        glm::vec4 normalizedPlane(glm::vec4 plane)
        {
            const float length = glm::length(glm::vec3(plane));
            return length > 0.0f ? plane / length : plane;
        }

        std::array<glm::vec4, 6> extractFrustum(const glm::mat4& matrix)
        {
            const glm::vec4 row0(matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]);
            const glm::vec4 row1(matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]);
            const glm::vec4 row2(matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]);
            const glm::vec4 row3(matrix[0][3], matrix[1][3], matrix[2][3], matrix[3][3]);
            return {
                normalizedPlane(row3 + row0), normalizedPlane(row3 - row0),
                normalizedPlane(row3 + row1), normalizedPlane(row3 - row1),
                normalizedPlane(row2), normalizedPlane(row3 - row2)
            };
        }
    }

    struct VulkanRenderer::Impl
    {
        struct Frame
        {
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
            VkSemaphore acquired = VK_NULL_HANDLE;
            VkSemaphore sparseReady = VK_NULL_HANDLE;
            VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
            VkDescriptorSet selectionDescriptorSet = VK_NULL_HANDLE;
            DepthTarget selectionMask;
            Buffer uniform;
            Buffer visible;
            Buffer indirect;
            std::vector<Buffer> retiredUploads;
            bool submitted = false;
            std::uint64_t descriptorGeneration = 0;
        };

        SDL_Window* window = nullptr;
        VkInstance instance = VK_NULL_HANDLE;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        std::uint32_t queueFamily = 0;
        bool sparseTexturesSupported = false;
        VmaAllocator allocator = VK_NULL_HANDLE;

        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{};
        std::vector<VkImage> swapchainImages;
        std::vector<VkImageView> swapchainViews;
        std::vector<VkImageLayout> swapchainLayouts;
        std::vector<DepthTarget> depthTargets;
        std::vector<VkSemaphore> presentSemaphores;
        std::uint32_t minSwapchainImageCount = frameCount;
        std::uint64_t swapchainGeneration = 0;

        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        std::uint32_t descriptorCapacity = 0;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline computePipeline = VK_NULL_HANDLE;
        VkPipeline graphicsPipeline = VK_NULL_HANDLE;
        VkPipeline decalPipeline = VK_NULL_HANDLE;
        VkPipeline shadowPipeline = VK_NULL_HANDLE;
        VkPipeline selectionPipeline = VK_NULL_HANDLE;
        VkPipeline outlinePipeline = VK_NULL_HANDLE;
        VkDescriptorSetLayout selectionDescriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool selectionDescriptorPool = VK_NULL_HANDLE;
        VkPipelineLayout outlinePipelineLayout = VK_NULL_HANDLE;
        VkSampler selectionSampler = VK_NULL_HANDLE;
        std::vector<std::uint32_t> selectedInstances;
        std::optional<std::uint32_t> activeSelection;
        VkQueryPool queryPool = VK_NULL_HANDLE;
        PFN_vkCmdDrawMeshTasksIndirectEXT drawMeshTasksIndirect = nullptr;

        Buffer vertices;
        Buffer indices;
        Buffer meshlets;
        Buffer instances;
        Buffer sources;
        Buffer materials;
        Buffer skinMatrices;
        DepthTarget shadowTarget;
        VkSampler shadowSampler = VK_NULL_HANDLE;
        VkSampler textureSampler = VK_NULL_HANDLE;
        std::vector<TextureTarget> textureTargets;
        VkExtent2D shadowExtent{2048, 2048};
        glm::vec3 sceneCenter{};
        float sceneRadius = 1.0f;
        std::array<Frame, frameCount> frames;
        std::uint32_t currentFrame = 0;
        std::uint32_t candidateCount = 0;
        std::uint32_t vertexCount = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t meshletCount = 0;
        std::uint32_t instanceCount = 0;
        std::uint32_t materialCount = 0;
        std::vector<GpuInstance> instanceData;
        std::vector<GpuMeshlet> meshletData;
        std::vector<GpuVertex> pendingVertices;
        std::vector<std::uint32_t> pendingIndices;
        std::vector<GpuMeshlet> pendingMeshlets;
        std::vector<GpuInstance> pendingInstances;
        std::vector<std::uint32_t> pendingInstanceUpdates;
        std::vector<glm::uvec2> pendingSources;
        std::vector<GpuMaterial> pendingMaterials;
        std::vector<SceneTexture> pendingTextures;
        struct MeshPatch
        {
            MeshRange range;
            PackedMesh mesh;
        };
        std::vector<MeshPatch> pendingMeshPatches;
        std::vector<glm::mat4> skinData{glm::mat4(1.0f)};
        bool skinDataDirty = false;
        std::uint64_t descriptorGeneration = 1;
        float timestampPeriod = 1.0f;
        std::string gpuName;
        RendererStats rendererStats;

        Impl(SDL_Window* targetWindow, const PackedMesh& mesh,
            std::span<const SceneInstance> sceneInstances,
            std::span<const SceneMaterial> sceneMaterials,
            std::span<const SceneTexture> sceneTextures)
            : window(targetWindow)
        {
            createInstance();
            check(SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)
                    ? VK_SUCCESS : VK_ERROR_SURFACE_LOST_KHR,
                "SDL_Vulkan_CreateSurface");
            selectPhysicalDevice();
            createDevice();
            createAllocator();
            createSwapchain();
            createCommandResources();
            createDescriptorResources(static_cast<std::uint32_t>(sceneTextures.size()));
            uploadScene(mesh, sceneInstances, sceneMaterials);
            createFrameResources();
            createShadowResources();
            uploadTextures(sceneTextures);
            createPipelines();
            updateDescriptorSets();

            VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
            queryInfo.queryCount = frameCount * 2;
            check(vkCreateQueryPool(device, &queryInfo, nullptr, &queryPool), "vkCreateQueryPool");
        }

        ~Impl()
        {
            if (device)
                vkDeviceWaitIdle(device);
            if (queryPool)
                vkDestroyQueryPool(device, queryPool, nullptr);
            if (selectionPipeline)
                vkDestroyPipeline(device, selectionPipeline, nullptr);
            if (outlinePipeline)
                vkDestroyPipeline(device, outlinePipeline, nullptr);
            if (outlinePipelineLayout)
                vkDestroyPipelineLayout(device, outlinePipelineLayout, nullptr);
            if (selectionDescriptorSetLayout)
                vkDestroyDescriptorSetLayout(device, selectionDescriptorSetLayout, nullptr);
            if (selectionSampler)
                vkDestroySampler(device, selectionSampler, nullptr);
            if (graphicsPipeline)
                vkDestroyPipeline(device, graphicsPipeline, nullptr);
            if (decalPipeline)
                vkDestroyPipeline(device, decalPipeline, nullptr);
            if (shadowPipeline)
                vkDestroyPipeline(device, shadowPipeline, nullptr);
            if (computePipeline)
                vkDestroyPipeline(device, computePipeline, nullptr);
            if (pipelineLayout)
                vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            if (descriptorPool)
                vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            if (descriptorSetLayout)
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            for (auto& frame : frames)
            {
                for (auto& upload : frame.retiredUploads)
                    destroyBuffer(upload);
                destroyBuffer(frame.indirect);
                destroyBuffer(frame.visible);
                destroyBuffer(frame.uniform);
                if (frame.acquired)
                    vkDestroySemaphore(device, frame.acquired, nullptr);
                if (frame.sparseReady)
                    vkDestroySemaphore(device, frame.sparseReady, nullptr);
                if (frame.fence)
                    vkDestroyFence(device, frame.fence, nullptr);
            }
            destroyBuffer(sources);
            destroyBuffer(instances);
            destroyBuffer(materials);
            destroyBuffer(skinMatrices);
            destroyBuffer(meshlets);
            destroyBuffer(indices);
            destroyBuffer(vertices);
            if (textureSampler)
                vkDestroySampler(device, textureSampler, nullptr);
            for (auto& texture : textureTargets)
            {
                if (texture.view)
                    vkDestroyImageView(device, texture.view, nullptr);
                if (texture.sparse)
                {
                    if (texture.image)
                        vkDestroyImage(device, texture.image, nullptr);
                    if (texture.sparseMemory)
                        vkFreeMemory(device, texture.sparseMemory, nullptr);
                }
                else if (texture.image)
                    vmaDestroyImage(allocator, texture.image, texture.allocation);
            }
            if (shadowSampler)
                vkDestroySampler(device, shadowSampler, nullptr);
            if (shadowTarget.view)
                vkDestroyImageView(device, shadowTarget.view, nullptr);
            if (shadowTarget.image)
                vmaDestroyImage(allocator, shadowTarget.image, shadowTarget.allocation);
            destroySwapchain();
            if (commandPool)
                vkDestroyCommandPool(device, commandPool, nullptr);
            if (allocator)
                vmaDestroyAllocator(allocator);
            if (device)
                vkDestroyDevice(device, nullptr);
            if (surface)
                vkDestroySurfaceKHR(instance, surface, nullptr);
            if (instance)
                vkDestroyInstance(instance, nullptr);
        }

        void createInstance()
        {
            Uint32 extensionCount = 0;
            const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
            if (!extensions)
                throw std::runtime_error(SDL_GetError());
            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.pApplicationName = "OpenF4";
            application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
            application.pEngineName = "OpenF4";
            application.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
            application.apiVersion = VK_API_VERSION_1_3;
            VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            createInfo.pApplicationInfo = &application;
            createInfo.enabledExtensionCount = extensionCount;
            createInfo.ppEnabledExtensionNames = extensions;
            check(vkCreateInstance(&createInfo, nullptr, &instance), "vkCreateInstance");
        }

        void selectPhysicalDevice()
        {
            std::uint32_t count = 0;
            check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
            std::vector<VkPhysicalDevice> devices(count);
            check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
            for (VkPhysicalDevice candidate : devices)
            {
                VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
                VkPhysicalDeviceVulkan12Features features12{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &meshFeatures};
                VkPhysicalDeviceVulkan13Features features13{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &features12};
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features13};
                vkGetPhysicalDeviceFeatures2(candidate, &features);
                VkPhysicalDeviceProperties candidateProperties{};
                vkGetPhysicalDeviceProperties(candidate, &candidateProperties);
                const auto& limits = candidateProperties.limits;
                constexpr std::uint32_t requiredSampledImages = maxSceneTextures + 1;
                if (!meshFeatures.meshShader || !features13.dynamicRendering || !features13.synchronization2
                    || !features13.maintenance4 || !features13.shaderDemoteToHelperInvocation
                    || !features12.runtimeDescriptorArray
                    || !features12.shaderSampledImageArrayNonUniformIndexing
                    || !features12.descriptorBindingPartiallyBound
                    || !features12.descriptorBindingVariableDescriptorCount
                    || limits.maxPerStageDescriptorSamplers < requiredSampledImages
                    || limits.maxPerStageDescriptorSampledImages < requiredSampledImages
                    || limits.maxDescriptorSetSamplers < requiredSampledImages
                    || limits.maxDescriptorSetSampledImages < requiredSampledImages)
                    continue;

                std::uint32_t familyCount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
                std::vector<VkQueueFamilyProperties> families(familyCount);
                vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
                for (std::uint32_t family = 0; family < familyCount; ++family)
                {
                    VkBool32 present = VK_FALSE;
                    vkGetPhysicalDeviceSurfaceSupportKHR(candidate, family, surface, &present);
                    if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present)
                    {
                        physicalDevice = candidate;
                        queueFamily = family;
                        sparseTexturesSupported = features.features.sparseBinding
                            && features.features.sparseResidencyImage2D
                            && (families[family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT);
                        bool sparseDisabled = false;
#ifdef _WIN32
                        char* sparseSetting = nullptr;
                        std::size_t sparseSettingSize = 0;
                        _dupenv_s(&sparseSetting, &sparseSettingSize, "OPENF4_DISABLE_SPARSE");
                        if (!sparseSetting)
                            _dupenv_s(&sparseSetting, &sparseSettingSize, "EMPIRE_DISABLE_SPARSE");
                        sparseDisabled = sparseSetting != nullptr;
                        std::free(sparseSetting);
#else
                        sparseDisabled = std::getenv("OPENF4_DISABLE_SPARSE") != nullptr
                            || std::getenv("EMPIRE_DISABLE_SPARSE") != nullptr;
#endif
                        if (sparseDisabled)
                            sparseTexturesSupported = false;
                        break;
                    }
                }
                if (physicalDevice)
                    break;
            }
            if (!physicalDevice)
                throw std::runtime_error("No Vulkan 1.3 GPU with VK_EXT_mesh_shader was found");

            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physicalDevice, &properties);
            gpuName = properties.deviceName;
            timestampPeriod = properties.limits.timestampPeriod;
        }

        void createDevice()
        {
            constexpr float priority = 1.0f;
            VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queueInfo.queueFamilyIndex = queueFamily;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
            meshFeatures.meshShader = VK_TRUE;
            VkPhysicalDeviceVulkan12Features features12{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &meshFeatures};
            features12.runtimeDescriptorArray = VK_TRUE;
            features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
            features12.descriptorBindingPartiallyBound = VK_TRUE;
            features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
            VkPhysicalDeviceVulkan13Features features13{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &features12};
            features13.dynamicRendering = VK_TRUE;
            features13.synchronization2 = VK_TRUE;
            features13.maintenance4 = VK_TRUE;
            features13.shaderDemoteToHelperInvocation = VK_TRUE;
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features13};
            features.features.sparseBinding = sparseTexturesSupported;
            features.features.sparseResidencyImage2D = sparseTexturesSupported;
            // Dear ImGui's Vulkan backend uses the KHR entry points for its
            // dynamic-rendering path, including on Vulkan 1.3 implementations.
            constexpr std::array extensions{
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
                VK_EXT_MESH_SHADER_EXTENSION_NAME};
            VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
            createInfo.queueCreateInfoCount = 1;
            createInfo.pQueueCreateInfos = &queueInfo;
            createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
            createInfo.ppEnabledExtensionNames = extensions.data();
            check(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device), "vkCreateDevice");
            vkGetDeviceQueue(device, queueFamily, 0, &queue);
            drawMeshTasksIndirect = reinterpret_cast<PFN_vkCmdDrawMeshTasksIndirectEXT>(
                vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksIndirectEXT"));
            if (!drawMeshTasksIndirect)
                throw std::runtime_error("vkCmdDrawMeshTasksIndirectEXT is unavailable");
        }

        void createAllocator()
        {
            VmaAllocatorCreateInfo createInfo{};
            createInfo.flags = VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
            createInfo.physicalDevice = physicalDevice;
            createInfo.device = device;
            createInfo.instance = instance;
            createInfo.vulkanApiVersion = VK_API_VERSION_1_3;
            check(vmaCreateAllocator(&createInfo, &allocator), "vmaCreateAllocator");
        }

        Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool hostVisible = false)
        {
            Buffer buffer;
            buffer.size = std::max<VkDeviceSize>(size, 4);
            VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bufferInfo.size = buffer.size;
            bufferInfo.usage = usage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo allocationInfo{};
            allocationInfo.usage = hostVisible ? VMA_MEMORY_USAGE_AUTO : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (hostVisible)
                allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo resultInfo{};
            check(vmaCreateBuffer(
                allocator, &bufferInfo, &allocationInfo, &buffer.handle, &buffer.allocation, &resultInfo),
                "vmaCreateBuffer");
            buffer.mapped = resultInfo.pMappedData;
            return buffer;
        }

        void destroyBuffer(Buffer& buffer)
        {
            if (buffer.handle)
                vmaDestroyBuffer(allocator, buffer.handle, buffer.allocation);
            buffer = {};
        }

        void waitForFrames()
        {
            std::array<VkFence, frameCount> fences{};
            for (std::size_t index = 0; index < frames.size(); ++index)
                fences[index] = frames[index].fence;
            check(vkWaitForFences(device, static_cast<std::uint32_t>(fences.size()),
                fences.data(), VK_TRUE, UINT64_MAX), "stream frame fences");
        }

        void submitBufferCopy(VkBuffer source, VkBuffer destination,
            VkDeviceSize sourceOffset, VkDeviceSize destinationOffset, VkDeviceSize size)
        {
            VkCommandBuffer command = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocateInfo.commandPool = commandPool;
            allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocateInfo.commandBufferCount = 1;
            check(vkAllocateCommandBuffers(device, &allocateInfo, &command),
                "vkAllocateCommandBuffers stream");
            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(command, &beginInfo), "vkBeginCommandBuffer stream");
            const VkBufferCopy copy{sourceOffset, destinationOffset, size};
            vkCmdCopyBuffer(command, source, destination, 1, &copy);
            check(vkEndCommandBuffer(command), "vkEndCommandBuffer stream");
            VkFence fence = VK_NULL_HANDLE;
            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check(vkCreateFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence stream");
            VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
            commandInfo.commandBuffer = command;
            VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            submitInfo.commandBufferInfoCount = 1;
            submitInfo.pCommandBufferInfos = &commandInfo;
            check(vkQueueSubmit2(queue, 1, &submitInfo, fence), "vkQueueSubmit2 stream");
            check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "stream upload fence");
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, commandPool, 1, &command);
        }

        bool ensureBuffer(Buffer& buffer, VkDeviceSize required, VkBufferUsageFlags usage,
            bool preserve = true)
        {
            if (buffer.size >= required)
                return false;
            waitForFrames();
            const VkDeviceSize capacity = std::max(required, std::max<VkDeviceSize>(buffer.size * 2, 256));
            Buffer replacement = createBuffer(capacity,
                usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            if (preserve && buffer.handle)
                submitBufferCopy(buffer.handle, replacement.handle, 0, 0, buffer.size);
            destroyBuffer(buffer);
            buffer = replacement;
            return true;
        }

        static VkDeviceSize streamingBufferCapacity(VkDeviceSize populatedSize)
        {
            return std::max<VkDeviceSize>(populatedSize * 2, 256);
        }

        void uploadBytes(Buffer& destination, VkDeviceSize offset, const void* data, VkDeviceSize size)
        {
            if (size == 0)
                return;
            Buffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            std::memcpy(staging.mapped, data, static_cast<std::size_t>(size));
            check(vmaFlushAllocation(allocator, staging.allocation, 0, size),
                "vmaFlushAllocation stream");
            submitBufferCopy(staging.handle, destination.handle, 0, offset, size);
            destroyBuffer(staging);
        }

        void createSwapchain()
        {
            VkSurfaceCapabilitiesKHR capabilities{};
            check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities),
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
            std::uint32_t formatCount = 0;
            check(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr),
                "vkGetPhysicalDeviceSurfaceFormatsKHR");
            std::vector<VkSurfaceFormatKHR> formats(formatCount);
            check(vkGetPhysicalDeviceSurfaceFormatsKHR(
                physicalDevice, surface, &formatCount, formats.data()), "vkGetPhysicalDeviceSurfaceFormatsKHR");
            const auto selected = std::find_if(formats.begin(), formats.end(), [](const auto& format) {
                return format.format == VK_FORMAT_B8G8R8A8_SRGB
                    && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
            });
            const VkSurfaceFormatKHR surfaceFormat = selected == formats.end() ? formats.front() : *selected;
            swapchainFormat = surfaceFormat.format;

            int pixelWidth = 0;
            int pixelHeight = 0;
            SDL_GetWindowSizeInPixels(window, &pixelWidth, &pixelHeight);
            if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
                extent = capabilities.currentExtent;
            else
            {
                extent.width = std::clamp(
                    static_cast<std::uint32_t>(std::max(pixelWidth, 1)),
                    capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
                extent.height = std::clamp(
                    static_cast<std::uint32_t>(std::max(pixelHeight, 1)),
                    capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
            }

            std::uint32_t imageCount = std::max(capabilities.minImageCount + 1, frameCount);
            if (capabilities.maxImageCount > 0)
                imageCount = std::min(imageCount, capabilities.maxImageCount);
            minSwapchainImageCount = std::min(imageCount,
                std::max(capabilities.minImageCount, frameCount));
            VkSwapchainCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
            createInfo.surface = surface;
            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = swapchainFormat;
            createInfo.imageColorSpace = surfaceFormat.colorSpace;
            createInfo.imageExtent = extent;
            createInfo.imageArrayLayers = 1;
            createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.preTransform = capabilities.currentTransform;
            createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
            createInfo.clipped = VK_TRUE;
            check(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain), "vkCreateSwapchainKHR");

            check(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr), "vkGetSwapchainImagesKHR");
            swapchainImages.resize(imageCount);
            check(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data()),
                "vkGetSwapchainImagesKHR");
            minSwapchainImageCount = std::min(minSwapchainImageCount, imageCount);
            swapchainLayouts.assign(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);
            swapchainViews.resize(imageCount);
            depthTargets.resize(imageCount);
            presentSemaphores.resize(imageCount);
            VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            for (std::uint32_t index = 0; index < imageCount; ++index)
            {
                check(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &presentSemaphores[index]),
                    "vkCreateSemaphore");
                VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                viewInfo.image = swapchainImages[index];
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = swapchainFormat;
                viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                viewInfo.subresourceRange.levelCount = 1;
                viewInfo.subresourceRange.layerCount = 1;
                check(vkCreateImageView(device, &viewInfo, nullptr, &swapchainViews[index]),
                    "vkCreateImageView");

                VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                imageInfo.imageType = VK_IMAGE_TYPE_2D;
                imageInfo.format = VK_FORMAT_D32_SFLOAT;
                imageInfo.extent = {extent.width, extent.height, 1};
                imageInfo.mipLevels = 1;
                imageInfo.arrayLayers = 1;
                imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
                imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
                imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
                VmaAllocationCreateInfo allocationInfo{};
                allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                check(vmaCreateImage(allocator, &imageInfo, &allocationInfo,
                    &depthTargets[index].image, &depthTargets[index].allocation, nullptr), "vmaCreateImage");
                viewInfo.image = depthTargets[index].image;
                viewInfo.format = VK_FORMAT_D32_SFLOAT;
                viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                check(vkCreateImageView(device, &viewInfo, nullptr, &depthTargets[index].view),
                    "vkCreateImageView");
            }
            ++swapchainGeneration;
        }

        void destroySwapchain()
        {
            destroySelectionTargets();
            for (VkSemaphore semaphore : presentSemaphores)
                vkDestroySemaphore(device, semaphore, nullptr);
            presentSemaphores.clear();
            for (auto& depth : depthTargets)
            {
                if (depth.view)
                    vkDestroyImageView(device, depth.view, nullptr);
                if (depth.image)
                    vmaDestroyImage(allocator, depth.image, depth.allocation);
            }
            depthTargets.clear();
            for (VkImageView view : swapchainViews)
                vkDestroyImageView(device, view, nullptr);
            swapchainViews.clear();
            swapchainImages.clear();
            swapchainLayouts.clear();
            if (swapchain)
                vkDestroySwapchainKHR(device, swapchain, nullptr);
            swapchain = VK_NULL_HANDLE;
        }

        void recreateSwapchain()
        {
            int width = 0;
            int height = 0;
            SDL_GetWindowSizeInPixels(window, &width, &height);
            if (width <= 0 || height <= 0)
                return;
            check(vkDeviceWaitIdle(device), "vkDeviceWaitIdle during resize");
            destroySwapchain();
            createSwapchain();
        }

        void createCommandResources()
        {
            VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = queueFamily;
            check(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "vkCreateCommandPool");
            std::array<VkCommandBuffer, frameCount> commandBuffers{};
            VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocateInfo.commandPool = commandPool;
            allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocateInfo.commandBufferCount = frameCount;
            check(vkAllocateCommandBuffers(device, &allocateInfo, commandBuffers.data()),
                "vkAllocateCommandBuffers");
            for (std::uint32_t index = 0; index < frameCount; ++index)
                frames[index].commandBuffer = commandBuffers[index];
        }

        static std::uint32_t grownDescriptorCapacity(std::uint32_t required)
        {
            std::uint32_t capacity = 16;
            while (capacity < required && capacity < maxSceneTextures)
                capacity *= 2;
            return std::min(capacity, maxSceneTextures);
        }

        void replaceDescriptorPool(std::uint32_t capacity)
        {
            const std::array sizes{
                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frameCount * 9},
                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frameCount},
                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    frameCount * (capacity + 1)}
            };
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets = frameCount;
            poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
            poolInfo.pPoolSizes = sizes.data();
            VkDescriptorPool newPool = VK_NULL_HANDLE;
            check(vkCreateDescriptorPool(device, &poolInfo, nullptr, &newPool),
                "vkCreateDescriptorPool");

            std::array<VkDescriptorSetLayout, frameCount> layouts{};
            layouts.fill(descriptorSetLayout);
            std::array<std::uint32_t, frameCount> descriptorCounts{};
            descriptorCounts.fill(capacity);
            VkDescriptorSetVariableDescriptorCountAllocateInfo variableInfo{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO};
            variableInfo.descriptorSetCount = frameCount;
            variableInfo.pDescriptorCounts = descriptorCounts.data();
            std::array<VkDescriptorSet, frameCount> sets{};
            VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                &variableInfo};
            allocateInfo.descriptorPool = newPool;
            allocateInfo.descriptorSetCount = frameCount;
            allocateInfo.pSetLayouts = layouts.data();
            const VkResult allocated = vkAllocateDescriptorSets(device, &allocateInfo, sets.data());
            if (allocated != VK_SUCCESS)
            {
                vkDestroyDescriptorPool(device, newPool, nullptr);
                check(allocated, "vkAllocateDescriptorSets");
            }

            const VkDescriptorPool oldPool = descriptorPool;
            descriptorPool = newPool;
            descriptorCapacity = capacity;
            for (std::uint32_t index = 0; index < frameCount; ++index)
                frames[index].descriptorSet = sets[index];
            if (oldPool)
                vkDestroyDescriptorPool(device, oldPool, nullptr);
        }

        void createDescriptorResources(std::uint32_t initialTextureCount)
        {
            std::array<VkDescriptorSetLayoutBinding, 12> bindings{};
            for (std::uint32_t index = 0; index < 7; ++index)
            {
                bindings[index].binding = index;
                bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[index].descriptorCount = 1;
                bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
                    | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            bindings[7].binding = 7;
            bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[7].descriptorCount = 1;
            bindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
                | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[8].binding = 8;
            bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[8].descriptorCount = 1;
            bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[9].binding = 9;
            bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[9].descriptorCount = 1;
            bindings[9].stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[10].binding = 10;
            bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[10].descriptorCount = 1;
            bindings[10].stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
            bindings[11].binding = 11;
            bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[11].descriptorCount = maxSceneTextures;
            bindings[11].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            std::array<VkDescriptorBindingFlags, bindings.size()> bindingFlags{};
            bindingFlags[11] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
                | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
            VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
            flagsInfo.bindingCount = static_cast<std::uint32_t>(bindingFlags.size());
            flagsInfo.pBindingFlags = bindingFlags.data();
            VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                &flagsInfo};
            layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
            check(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout),
                "vkCreateDescriptorSetLayout");

            replaceDescriptorPool(grownDescriptorCapacity(std::max(1u, initialTextureCount)));

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
            check(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
                "vkCreatePipelineLayout");
        }

        template <class T, std::size_t Extent>
        static VkDeviceSize byteSize(std::span<T, Extent> values)
        {
            return values.size_bytes();
        }

        void uploadScene(const PackedMesh& mesh, std::span<const SceneInstance> sceneInstances,
            std::span<const SceneMaterial> sceneMaterials)
        {
            if (mesh.meshlets.empty() || sceneInstances.empty())
                throw std::runtime_error("The renderer needs at least one meshlet and one instance");
            instanceData.clear();
            instanceData.reserve(sceneInstances.size());
            std::vector<GpuMaterial> gpuMaterials;
            if (sceneMaterials.empty())
                gpuMaterials.push_back({glm::vec4(0.7f, 0.72f, 0.68f, 1.0f), glm::uvec4(0), {}});
            else
            {
                gpuMaterials.reserve(sceneMaterials.size());
                for (const auto& material : sceneMaterials)
                    gpuMaterials.push_back(makeGpuMaterial(material));
            }
            glm::vec3 sceneMinimum(std::numeric_limits<float>::max());
            glm::vec3 sceneMaximum(std::numeric_limits<float>::lowest());
            for (const auto& source : sceneInstances)
            {
                const std::uint32_t meshletOffset = source.meshletCount == 0 ? 0 : source.meshletOffset;
                const std::uint32_t rangeMeshletCount = source.meshletCount == 0
                    ? static_cast<std::uint32_t>(mesh.meshlets.size()) : source.meshletCount;
                if (static_cast<std::size_t>(meshletOffset) + rangeMeshletCount > mesh.meshlets.size())
                    throw std::runtime_error("Scene instance meshlet range is out of bounds");
                glm::vec4 worldBounds = source.bounds;
                worldBounds.x = -worldBounds.x;
                instanceData.push_back({source.transform, worldBounds, source.lodRange,
                    glm::uvec4(meshletOffset, rangeMeshletCount, source.active ? 1u : 0u,
                        source.overlay ? 1u : 0u),
                    glm::uvec4(source.skinOffset, source.skinCount, 0u, 0u)});
                sceneMinimum = glm::min(sceneMinimum, glm::vec3(worldBounds) - worldBounds.w);
                sceneMaximum = glm::max(sceneMaximum, glm::vec3(worldBounds) + worldBounds.w);
            }
            sceneCenter = (sceneMinimum + sceneMaximum) * 0.5f;
            sceneRadius = std::max(glm::distance(sceneCenter, sceneMaximum), 1.0f);
            std::vector<glm::uvec2> candidateSources;
            candidateSources.reserve(sceneInstances.size() * mesh.meshlets.size());
            for (std::uint32_t instanceIndex = 0; instanceIndex < sceneInstances.size(); ++instanceIndex)
            {
                const auto& range = instanceData[instanceIndex].meshletRange;
                for (std::uint32_t meshletIndex = range.x; meshletIndex < range.x + range.y; ++meshletIndex)
                    candidateSources.emplace_back(instanceIndex, meshletIndex);
            }
            candidateCount = static_cast<std::uint32_t>(candidateSources.size());
            vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
            indexCount = static_cast<std::uint32_t>(mesh.indices.size());
            meshletCount = static_cast<std::uint32_t>(mesh.meshlets.size());
            meshletData = mesh.meshlets;
            instanceCount = static_cast<std::uint32_t>(instanceData.size());
            materialCount = static_cast<std::uint32_t>(gpuMaterials.size());
            rendererStats.sourceCandidates = candidateCount;

            vertices = createBuffer(streamingBufferCapacity(byteSize(std::span(mesh.vertices))),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            indices = createBuffer(streamingBufferCapacity(byteSize(std::span(mesh.indices))),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            meshlets = createBuffer(streamingBufferCapacity(byteSize(std::span(mesh.meshlets))),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            instances = createBuffer(streamingBufferCapacity(byteSize(std::span(instanceData))),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            sources = createBuffer(streamingBufferCapacity(byteSize(std::span(candidateSources))),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            materials = createBuffer(streamingBufferCapacity(byteSize(std::span(gpuMaterials))),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            skinMatrices = createBuffer(streamingBufferCapacity(byteSize(std::span(skinData))),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

            const std::array destinations{
                &vertices, &indices, &meshlets, &instances, &sources, &materials, &skinMatrices};
            const std::array sizes{
                byteSize(std::span(mesh.vertices)), byteSize(std::span(mesh.indices)),
                byteSize(std::span(mesh.meshlets)), byteSize(std::span(instanceData)),
                byteSize(std::span(candidateSources)), byteSize(std::span(gpuMaterials)),
                byteSize(std::span(skinData))
            };
            std::array<VkDeviceSize, destinations.size()> offsets{};
            VkDeviceSize totalSize = 0;
            for (std::size_t index = 0; index < sizes.size(); ++index)
            {
                totalSize = (totalSize + 15) & ~VkDeviceSize(15);
                offsets[index] = totalSize;
                totalSize += sizes[index];
            }
            Buffer staging = createBuffer(totalSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            const std::array sourceData{
                static_cast<const void*>(mesh.vertices.data()), static_cast<const void*>(mesh.indices.data()),
                static_cast<const void*>(mesh.meshlets.data()), static_cast<const void*>(instanceData.data()),
                static_cast<const void*>(candidateSources.data()), static_cast<const void*>(gpuMaterials.data()),
                static_cast<const void*>(skinData.data())
            };
            for (std::size_t index = 0; index < sizes.size(); ++index)
                std::memcpy(static_cast<std::byte*>(staging.mapped) + offsets[index], sourceData[index], sizes[index]);
            check(vmaFlushAllocation(allocator, staging.allocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");

            VkCommandBuffer command = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocateInfo.commandPool = commandPool;
            allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocateInfo.commandBufferCount = 1;
            check(vkAllocateCommandBuffers(device, &allocateInfo, &command), "vkAllocateCommandBuffers");
            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(command, &beginInfo), "vkBeginCommandBuffer");
            for (std::size_t index = 0; index < sizes.size(); ++index)
            {
                const VkBufferCopy copy{offsets[index], 0, sizes[index]};
                vkCmdCopyBuffer(command, staging.handle, destinations[index]->handle, 1, &copy);
            }
            check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
            VkFence fence = VK_NULL_HANDLE;
            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check(vkCreateFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence");
            VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
            commandInfo.commandBuffer = command;
            VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            submitInfo.commandBufferInfoCount = 1;
            submitInfo.pCommandBufferInfos = &commandInfo;
            check(vkQueueSubmit2(queue, 1, &submitInfo, fence), "vkQueueSubmit2");
            check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "scene upload fence");
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, commandPool, 1, &command);
            destroyBuffer(staging);
        }

        void createFrameResources()
        {
            for (auto& frame : frames)
            {
                frame.uniform = createBuffer(sizeof(CameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
                frame.visible = createBuffer(
                    streamingBufferCapacity(candidateCount * sizeof(glm::uvec2)),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                frame.indirect = createBuffer(sizeof(VkDrawMeshTasksIndirectCommandEXT),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                        | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    true);
                VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
                check(vkCreateFence(device, &fenceInfo, nullptr, &frame.fence), "vkCreateFence");
                VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
                check(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &frame.acquired), "vkCreateSemaphore");
                check(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &frame.sparseReady),
                    "vkCreateSemaphore sparse-ready");
            }
        }

        void createShadowResources()
        {
            VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = VK_FORMAT_D32_SFLOAT;
            imageInfo.extent = {shadowExtent.width, shadowExtent.height, 1};
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            VmaAllocationCreateInfo allocationInfo{};
            allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            check(vmaCreateImage(allocator, &imageInfo, &allocationInfo,
                &shadowTarget.image, &shadowTarget.allocation, nullptr), "vmaCreateImage shadow");

            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = shadowTarget.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_D32_SFLOAT;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            check(vkCreateImageView(device, &viewInfo, nullptr, &shadowTarget.view),
                "vkCreateImageView shadow");

            VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            samplerInfo.compareEnable = VK_TRUE;
            samplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
            samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            samplerInfo.maxLod = 1.0f;
            check(vkCreateSampler(device, &samplerInfo, nullptr, &shadowSampler), "vkCreateSampler shadow");
        }

        static VkFormat textureFormat(TextureFormat format)
        {
            switch (format)
            {
                case TextureFormat::rgba8: return VK_FORMAT_R8G8B8A8_UNORM;
                case TextureFormat::rgba8Srgb: return VK_FORMAT_R8G8B8A8_SRGB;
                case TextureFormat::bc1: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
                case TextureFormat::bc1Srgb: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
                case TextureFormat::bc2: return VK_FORMAT_BC2_UNORM_BLOCK;
                case TextureFormat::bc2Srgb: return VK_FORMAT_BC2_SRGB_BLOCK;
                case TextureFormat::bc3: return VK_FORMAT_BC3_UNORM_BLOCK;
                case TextureFormat::bc3Srgb: return VK_FORMAT_BC3_SRGB_BLOCK;
                case TextureFormat::bc5: return VK_FORMAT_BC5_UNORM_BLOCK;
                case TextureFormat::bc7: return VK_FORMAT_BC7_UNORM_BLOCK;
                case TextureFormat::bc7Srgb: return VK_FORMAT_BC7_SRGB_BLOCK;
                case TextureFormat::bgra8: return VK_FORMAT_B8G8R8A8_UNORM;
            }
            return VK_FORMAT_R8G8B8A8_UNORM;
        }

        std::uint32_t findMemoryType(std::uint32_t bits, VkMemoryPropertyFlags properties) const
        {
            VkPhysicalDeviceMemoryProperties memoryProperties{};
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
            for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index)
            {
                if ((bits & (1u << index))
                    && (memoryProperties.memoryTypes[index].propertyFlags & properties) == properties)
                    return index;
            }
            throw std::runtime_error("No compatible Vulkan memory type was found");
        }

        bool sparseTextureFormatSupported(VkFormat format) const
        {
            std::uint32_t count = 0;
            vkGetPhysicalDeviceSparseImageFormatProperties(physicalDevice, format,
                VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_TILING_OPTIMAL, &count, nullptr);
            return count != 0;
        }

        TextureTarget createTextureTarget(const SceneTexture& texture, bool allowSparse = true)
        {
            TextureTarget target;
            const VkFormat format = textureFormat(texture.format);
            VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = format;
            imageInfo.extent = {texture.width, texture.height, 1};
            imageInfo.mipLevels = texture.mipLevels;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

            const std::uint32_t cappedMip = detail::sparseFirstResidentMip(
                texture.width, texture.height, texture.mipLevels, texture.mipLevels);
            const bool attemptSparse = allowSparse && sparseTexturesSupported && cappedMip != 0
                && sparseTextureFormatSupported(format);
            if (attemptSparse)
            {
                imageInfo.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT
                    | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
                if (vkCreateImage(device, &imageInfo, nullptr, &target.image) == VK_SUCCESS)
                {
                    VkMemoryRequirements memoryRequirements{};
                    vkGetImageMemoryRequirements(device, target.image, &memoryRequirements);
                    std::uint32_t sparseCount = 0;
                    vkGetImageSparseMemoryRequirements(
                        device, target.image, &sparseCount, nullptr);
                    std::vector<VkSparseImageMemoryRequirements> sparseRequirements(sparseCount);
                    if (sparseCount != 0)
                        vkGetImageSparseMemoryRequirements(device, target.image,
                            &sparseCount, sparseRequirements.data());

                    const VkSparseImageMemoryRequirements* color = nullptr;
                    for (const auto& requirement : sparseRequirements)
                    {
                        if (requirement.formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT)
                        {
                            color = &requirement;
                            break;
                        }
                    }
                    if (color && color->formatProperties.imageGranularity.width != 0
                        && color->formatProperties.imageGranularity.height != 0)
                    {
                        target.firstResidentMip = detail::sparseFirstResidentMip(
                            texture.width, texture.height, texture.mipLevels,
                            color->imageMipTailFirstLod);
                        VkDeviceSize memoryOffset = 0;
                        const VkExtent3D granularity = color->formatProperties.imageGranularity;
                        const std::uint32_t firstTailMip = std::min(
                            color->imageMipTailFirstLod, texture.mipLevels);
                        for (std::uint32_t mip = target.firstResidentMip;
                            mip < firstTailMip; ++mip)
                        {
                            const VkExtent3D mipExtent{
                                std::max(1u, texture.width >> mip),
                                std::max(1u, texture.height >> mip), 1};
                            for (std::uint32_t y = 0; y < mipExtent.height; y += granularity.height)
                            {
                                for (std::uint32_t x = 0; x < mipExtent.width; x += granularity.width)
                                {
                                    VkSparseImageMemoryBind bind{};
                                    bind.subresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0};
                                    bind.offset = {
                                        static_cast<std::int32_t>(x),
                                        static_cast<std::int32_t>(y), 0};
                                    bind.extent = {
                                        std::min(granularity.width, mipExtent.width - x),
                                        std::min(granularity.height, mipExtent.height - y), 1};
                                    bind.memoryOffset = memoryOffset;
                                    target.imageBinds.push_back(bind);
                                    memoryOffset += memoryRequirements.alignment;
                                }
                            }
                        }
                        if (color->imageMipTailSize != 0)
                        {
                            memoryOffset = alignUp(memoryOffset, memoryRequirements.alignment);
                            target.opaqueBinds.push_back({color->imageMipTailOffset,
                                color->imageMipTailSize, VK_NULL_HANDLE, memoryOffset, 0});
                            memoryOffset += color->imageMipTailSize;
                        }
                        for (const auto& requirement : sparseRequirements)
                        {
                            if (!(requirement.formatProperties.aspectMask & VK_IMAGE_ASPECT_METADATA_BIT)
                                || requirement.imageMipTailSize == 0)
                                continue;
                            memoryOffset = alignUp(memoryOffset, memoryRequirements.alignment);
                            target.opaqueBinds.push_back({requirement.imageMipTailOffset,
                                requirement.imageMipTailSize, VK_NULL_HANDLE, memoryOffset,
                                VK_SPARSE_MEMORY_BIND_METADATA_BIT});
                            memoryOffset += requirement.imageMipTailSize;
                        }
                        memoryOffset = alignUp(memoryOffset, memoryRequirements.alignment);
                        if (target.firstResidentMip != 0 && memoryOffset != 0
                            && memoryOffset < memoryRequirements.size)
                        {
                            VkMemoryAllocateInfo allocationInfo{
                                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                            allocationInfo.allocationSize = memoryOffset;
                            allocationInfo.memoryTypeIndex = findMemoryType(
                                memoryRequirements.memoryTypeBits,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                            check(vkAllocateMemory(device, &allocationInfo, nullptr,
                                &target.sparseMemory), "vkAllocateMemory sparse texture pages");
                            for (auto& bind : target.imageBinds)
                                bind.memory = target.sparseMemory;
                            for (auto& bind : target.opaqueBinds)
                                bind.memory = target.sparseMemory;
                            target.sparse = true;
                            target.virtualBytes = memoryRequirements.size;
                            target.residentBytes = memoryOffset;
                        }
                    }
                }
            }

            if (!target.sparse)
            {
                if (target.image)
                    vkDestroyImage(device, target.image, nullptr);
                target = {};
                imageInfo.flags = 0;
                VmaAllocationCreateInfo allocationInfo{};
                allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                check(vmaCreateImage(allocator, &imageInfo, &allocationInfo,
                    &target.image, &target.allocation, nullptr), "vmaCreateImage texture");
                VkMemoryRequirements memoryRequirements{};
                vkGetImageMemoryRequirements(device, target.image, &memoryRequirements);
                target.virtualBytes = memoryRequirements.size;
                target.residentBytes = memoryRequirements.size;
            }

            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = target.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = target.firstResidentMip;
            viewInfo.subresourceRange.levelCount = texture.mipLevels - target.firstResidentMip;
            viewInfo.subresourceRange.layerCount = 1;
            check(vkCreateImageView(device, &viewInfo, nullptr, &target.view), "vkCreateImageView texture");
            rendererStats.textureVirtualBytes += target.virtualBytes;
            rendererStats.textureResidentBytes += target.residentBytes;
            if (target.sparse)
                ++rendererStats.sparseTextures;
            return target;
        }

        bool submitSparseBinds(std::span<TextureTarget*> targets,
            VkSemaphore signalSemaphore = VK_NULL_HANDLE, VkFence fence = VK_NULL_HANDLE)
        {
            std::vector<VkSparseImageMemoryBindInfo> imageInfos;
            std::vector<VkSparseImageOpaqueMemoryBindInfo> opaqueInfos;
            std::vector<TextureTarget*> submitted;
            for (TextureTarget* target : targets)
            {
                if (!target || !target->sparse || target->sparseBound)
                    continue;
                if (!target->imageBinds.empty())
                    imageInfos.push_back({target->image,
                        static_cast<std::uint32_t>(target->imageBinds.size()),
                        target->imageBinds.data()});
                if (!target->opaqueBinds.empty())
                    opaqueInfos.push_back({target->image,
                        static_cast<std::uint32_t>(target->opaqueBinds.size()),
                        target->opaqueBinds.data()});
                submitted.push_back(target);
            }
            if (submitted.empty())
                return false;

            VkBindSparseInfo bindInfo{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
            bindInfo.imageBindCount = static_cast<std::uint32_t>(imageInfos.size());
            bindInfo.pImageBinds = imageInfos.data();
            bindInfo.imageOpaqueBindCount = static_cast<std::uint32_t>(opaqueInfos.size());
            bindInfo.pImageOpaqueBinds = opaqueInfos.data();
            if (signalSemaphore)
            {
                bindInfo.signalSemaphoreCount = 1;
                bindInfo.pSignalSemaphores = &signalSemaphore;
            }
            check(vkQueueBindSparse(queue, 1, &bindInfo, fence), "vkQueueBindSparse texture batch");
            for (TextureTarget* target : submitted)
                target->sparseBound = true;
            return true;
        }

        void uploadTextures(std::span<const SceneTexture> sceneTextures)
        {
            SceneTexture fallback;
            fallback.pixels = {std::byte{0xB3}, std::byte{0xB8}, std::byte{0xAD}, std::byte{0xFF}};
            fallback.mipOffsets = {0};
            std::vector<const SceneTexture*> textures;
            if (sceneTextures.empty())
                textures.push_back(&fallback);
            else
            {
                if (sceneTextures.size() > maxSceneTextures)
                    throw std::runtime_error("Scene exceeds the bindless texture capacity");
                for (const auto& texture : sceneTextures)
                    textures.push_back(&texture);
            }

            VkDeviceSize stagingSize = 0;
            std::vector<VkDeviceSize> stagingOffsets;
            std::vector<std::size_t> sourceOffsets;
            stagingOffsets.reserve(textures.size());
            sourceOffsets.reserve(textures.size());
            textureTargets.reserve(textures.size());
            rendererStats.residentTextures = static_cast<std::uint32_t>(textures.size());
            for (const SceneTexture* texture : textures)
            {
                if (texture->pixels.empty() || texture->width == 0 || texture->height == 0
                    || texture->mipLevels == 0 || texture->mipOffsets.size() < texture->mipLevels)
                    throw std::runtime_error("Scene texture has no image data");
                textureTargets.push_back(createTextureTarget(*texture));
                const auto& target = textureTargets.back();
                const std::size_t sourceOffset = texture->mipOffsets[target.firstResidentMip];
                if (sourceOffset >= texture->pixels.size())
                    throw std::runtime_error("Scene texture resident mip data is truncated");
                stagingSize = (stagingSize + 15) & ~VkDeviceSize(15);
                stagingOffsets.push_back(stagingSize);
                sourceOffsets.push_back(sourceOffset);
                stagingSize += texture->pixels.size() - sourceOffset;
            }
            std::vector<TextureTarget*> sparseTargets;
            sparseTargets.reserve(textureTargets.size());
            for (auto& target : textureTargets)
                sparseTargets.push_back(&target);
            VkFence sparseFence = VK_NULL_HANDLE;
            VkFenceCreateInfo sparseFenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check(vkCreateFence(device, &sparseFenceInfo, nullptr, &sparseFence),
                "vkCreateFence initial sparse batch");
            if (submitSparseBinds(sparseTargets, VK_NULL_HANDLE, sparseFence))
                check(vkWaitForFences(device, 1, &sparseFence, VK_TRUE, UINT64_MAX),
                    "initial sparse texture batch");
            vkDestroyFence(device, sparseFence, nullptr);

            Buffer staging = createBuffer(stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            for (std::size_t index = 0; index < textures.size(); ++index)
                std::memcpy(static_cast<std::byte*>(staging.mapped) + stagingOffsets[index],
                    textures[index]->pixels.data() + sourceOffsets[index],
                    textures[index]->pixels.size() - sourceOffsets[index]);
            check(vmaFlushAllocation(allocator, staging.allocation, 0, VK_WHOLE_SIZE),
                "vmaFlushAllocation textures");

            VkCommandBuffer command = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocateInfo.commandPool = commandPool;
            allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocateInfo.commandBufferCount = 1;
            check(vkAllocateCommandBuffers(device, &allocateInfo, &command), "vkAllocateCommandBuffers textures");
            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(command, &beginInfo), "vkBeginCommandBuffer textures");
            for (std::size_t index = 0; index < textures.size(); ++index)
            {
                const SceneTexture& texture = *textures[index];
                const TextureTarget& target = textureTargets[index];
                imageBarrier(command, textureTargets[index].image, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                    texture.mipLevels - target.firstResidentMip, target.firstResidentMip);
                for (std::uint32_t mip = target.firstResidentMip;
                    mip < texture.mipLevels; ++mip)
                {
                    const std::size_t offset = texture.mipOffsets[mip] - sourceOffsets[index];
                    VkBufferImageCopy copy{};
                    copy.bufferOffset = stagingOffsets[index] + offset;
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
                    copy.imageExtent = {
                        std::max(1u, texture.width >> mip), std::max(1u, texture.height >> mip), 1};
                    vkCmdCopyBufferToImage(command, staging.handle, textureTargets[index].image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
                }
                imageBarrier(command, textureTargets[index].image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                    texture.mipLevels - target.firstResidentMip, target.firstResidentMip);
            }
            check(vkEndCommandBuffer(command), "vkEndCommandBuffer textures");
            VkFence fence = VK_NULL_HANDLE;
            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check(vkCreateFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence textures");
            VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
            commandInfo.commandBuffer = command;
            VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            submitInfo.commandBufferInfoCount = 1;
            submitInfo.pCommandBufferInfos = &commandInfo;
            check(vkQueueSubmit2(queue, 1, &submitInfo, fence), "vkQueueSubmit2 textures");
            check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "texture upload fence");
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, commandPool, 1, &command);
            destroyBuffer(staging);

            VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
            check(vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler),
                "vkCreateSampler textures");
        }

        void updateDescriptorSet(Frame& frame)
        {
            const std::array<VkDescriptorBufferInfo, 8> bufferInfos{
                VkDescriptorBufferInfo{vertices.handle, 0, vertices.size},
                VkDescriptorBufferInfo{indices.handle, 0, indices.size},
                VkDescriptorBufferInfo{meshlets.handle, 0, meshlets.size},
                VkDescriptorBufferInfo{instances.handle, 0, instances.size},
                VkDescriptorBufferInfo{sources.handle, 0, sources.size},
                VkDescriptorBufferInfo{frame.visible.handle, 0, frame.visible.size},
                VkDescriptorBufferInfo{frame.indirect.handle, 0, frame.indirect.size},
                VkDescriptorBufferInfo{frame.uniform.handle, 0, frame.uniform.size}
            };
            std::array<VkWriteDescriptorSet, 12> writes{};
            for (std::uint32_t index = 0; index < 8; ++index)
            {
                writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[index].dstSet = frame.descriptorSet;
                writes[index].dstBinding = index;
                writes[index].descriptorCount = 1;
                writes[index].descriptorType = index == 7
                    ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[index].pBufferInfo = &bufferInfos[index];
            }
            const VkDescriptorImageInfo imageInfo{
                shadowSampler, shadowTarget.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[8].dstSet = frame.descriptorSet;
            writes[8].dstBinding = 8;
            writes[8].descriptorCount = 1;
            writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[8].pImageInfo = &imageInfo;
            std::vector<VkDescriptorImageInfo> textureInfos;
            textureInfos.reserve(textureTargets.size());
            for (const auto& texture : textureTargets)
                textureInfos.push_back({textureSampler, texture.view,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[9].dstSet = frame.descriptorSet;
            writes[9].dstBinding = 9;
            const VkDescriptorBufferInfo materialInfo{materials.handle, 0, materials.size};
            writes[9].descriptorCount = 1;
            writes[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[9].pBufferInfo = &materialInfo;
            writes[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[10].dstSet = frame.descriptorSet;
            writes[10].dstBinding = 10;
            const VkDescriptorBufferInfo skinInfo{skinMatrices.handle, 0, skinMatrices.size};
            writes[10].descriptorCount = 1;
            writes[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[10].pBufferInfo = &skinInfo;
            writes[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[11].dstSet = frame.descriptorSet;
            writes[11].dstBinding = 11;
            writes[11].descriptorCount = static_cast<std::uint32_t>(textureInfos.size());
            writes[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[11].pImageInfo = textureInfos.data();
            vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            frame.descriptorGeneration = descriptorGeneration;
        }

        void updateDescriptorSets()
        {
            for (auto& frame : frames)
                updateDescriptorSet(frame);
        }

        void ensureDescriptorCapacity(std::uint32_t required)
        {
            if (required <= descriptorCapacity)
                return;
            waitForFrames();
            replaceDescriptorPool(grownDescriptorCapacity(required));
            ++descriptorGeneration;
            updateDescriptorSets();
        }

        std::uint32_t appendTexture(SceneTexture texture)
        {
            if (textureTargets.size() >= maxSceneTextures)
                throw std::runtime_error("Scene exceeds the bindless texture capacity");
            if (texture.pixels.empty() || texture.width == 0 || texture.height == 0
                || texture.mipLevels == 0 || texture.mipOffsets.size() < texture.mipLevels)
                throw std::runtime_error("Streamed texture has no image data");
            const auto index = static_cast<std::uint32_t>(textureTargets.size());
            textureTargets.push_back(createTextureTarget(texture));
            pendingTextures.push_back(std::move(texture));
            rendererStats.residentTextures = static_cast<std::uint32_t>(textureTargets.size());
            return index;
        }

        std::uint32_t appendMaterial(const SceneMaterial& material)
        {
            const GpuMaterial gpu = makeGpuMaterial(material);
            const std::uint32_t index = materialCount++;
            pendingMaterials.push_back(gpu);
            return index;
        }

        MeshRange appendPackedMesh(PackedMesh mesh)
        {
            if (mesh.vertices.empty() || mesh.indices.empty() || mesh.meshlets.empty())
                throw std::runtime_error("Streamed mesh is empty");
            MeshRange range;
            range.vertexOffset = vertexCount;
            range.vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
            range.indexOffset = indexCount;
            range.indexCount = static_cast<std::uint32_t>(mesh.indices.size());
            range.meshletOffset = meshletCount;
            range.meshletCount = static_cast<std::uint32_t>(mesh.meshlets.size());
            range.bounds = mesh.bounds;
            for (auto& meshlet : mesh.meshlets)
            {
                if (meshlet.vertexOffset + meshlet.vertexCount > mesh.vertices.size()
                    || meshlet.indexOffset + meshlet.triangleCount * 3 > mesh.indices.size())
                    throw std::runtime_error("Streamed meshlet range is invalid");
                for (std::uint32_t index = 0; index < meshlet.triangleCount * 3; ++index)
                    if (mesh.indices[meshlet.indexOffset + index] >= meshlet.vertexCount)
                        throw std::runtime_error("Streamed meshlet has an invalid local index");
                if (meshlet.materialIndex >= materialCount)
                    throw std::runtime_error("Streamed meshlet has an invalid material index");
                meshlet.vertexOffset += vertexCount;
                meshlet.indexOffset += indexCount;
            }
            vertexCount += static_cast<std::uint32_t>(mesh.vertices.size());
            indexCount += static_cast<std::uint32_t>(mesh.indices.size());
            meshletCount += static_cast<std::uint32_t>(mesh.meshlets.size());
            meshletData.insert(meshletData.end(), mesh.meshlets.begin(), mesh.meshlets.end());
            pendingVertices.insert(pendingVertices.end(),
                std::make_move_iterator(mesh.vertices.begin()), std::make_move_iterator(mesh.vertices.end()));
            pendingIndices.insert(pendingIndices.end(),
                std::make_move_iterator(mesh.indices.begin()), std::make_move_iterator(mesh.indices.end()));
            pendingMeshlets.insert(pendingMeshlets.end(),
                std::make_move_iterator(mesh.meshlets.begin()), std::make_move_iterator(mesh.meshlets.end()));
            return range;
        }

        void replacePackedMesh(const MeshRange& range, PackedMesh mesh)
        {
            if (!detail::fixedTopologyReplacement(range, meshletData, mesh)
                || static_cast<std::uint64_t>(range.vertexOffset) + range.vertexCount > vertexCount
                || static_cast<std::uint64_t>(range.indexOffset) + range.indexCount > indexCount
                || static_cast<std::uint64_t>(range.meshletOffset) + range.meshletCount > meshletCount)
                throw std::runtime_error("Mesh replacement must preserve its fixed topology");

            for (std::size_t i = 0; i < mesh.meshlets.size(); ++i)
            {
                auto& meshlet = mesh.meshlets[i];
                if (meshlet.vertexOffset + meshlet.vertexCount > mesh.vertices.size()
                    || meshlet.indexOffset + meshlet.triangleCount * 3 > mesh.indices.size()
                    || meshlet.materialIndex >= materialCount)
                    throw std::runtime_error("Mesh replacement changed a meshlet topology");
                for (std::uint32_t index = 0; index < meshlet.triangleCount * 3; ++index)
                    if (mesh.indices[meshlet.indexOffset + index] >= meshlet.vertexCount)
                        throw std::runtime_error("Mesh replacement has an invalid local index");
                meshlet.vertexOffset += range.vertexOffset;
                meshlet.indexOffset += range.indexOffset;
            }

            std::copy(mesh.meshlets.begin(), mesh.meshlets.end(),
                meshletData.begin() + range.meshletOffset);
            const std::size_t pendingVertexBase = vertexCount - pendingVertices.size();
            const std::size_t pendingMeshletBase = meshletCount - pendingMeshlets.size();
            if (range.vertexOffset >= pendingVertexBase && range.meshletOffset >= pendingMeshletBase)
            {
                // A newly streamed cell can be rebuilt before its first upload.
                // Replace that queued data, rather than issue overlapping GPU copies.
                std::copy(mesh.vertices.begin(), mesh.vertices.end(),
                    pendingVertices.begin() + (range.vertexOffset - pendingVertexBase));
                std::copy(mesh.meshlets.begin(), mesh.meshlets.end(),
                    pendingMeshlets.begin() + (range.meshletOffset - pendingMeshletBase));
                return;
            }
            const auto existing = std::find_if(pendingMeshPatches.begin(), pendingMeshPatches.end(),
                [&range](const MeshPatch& patch) {
                    return patch.range.vertexOffset == range.vertexOffset
                        && patch.range.indexOffset == range.indexOffset
                        && patch.range.meshletOffset == range.meshletOffset;
                });
            if (existing != pendingMeshPatches.end())
                existing->mesh = std::move(mesh);
            else
                pendingMeshPatches.push_back({range, std::move(mesh)});
        }

        std::uint32_t appendInstance(SceneInstance source)
        {
            if (source.meshletCount == 0 || source.meshletOffset + source.meshletCount > meshletCount)
                throw std::runtime_error("Streamed instance has an invalid meshlet range");
            if (source.skinCount != 0
                && static_cast<std::size_t>(source.skinOffset) + source.skinCount > skinData.size())
                throw std::runtime_error("Streamed instance has an invalid skin palette range");
            source.bounds.x = -source.bounds.x;
            const GpuInstance gpu{source.transform, source.bounds, source.lodRange,
                glm::uvec4(source.meshletOffset, source.meshletCount,
                    source.active ? 1u : 0u, source.overlay ? 1u : 0u),
                glm::uvec4(source.skinOffset, source.skinCount, 0u, 0u)};
            std::vector<glm::uvec2> candidates;
            candidates.reserve(source.meshletCount);
            for (std::uint32_t meshlet = source.meshletOffset;
                meshlet < source.meshletOffset + source.meshletCount; ++meshlet)
                candidates.emplace_back(instanceCount, meshlet);

            const std::uint32_t index = instanceCount++;
            instanceData.push_back(gpu);
            pendingInstances.push_back(gpu);
            pendingSources.insert(pendingSources.end(), candidates.begin(), candidates.end());
            candidateCount += source.meshletCount;
            rendererStats.sourceCandidates = candidateCount;

            const glm::vec3 oldMinimum = sceneCenter - sceneRadius;
            const glm::vec3 oldMaximum = sceneCenter + sceneRadius;
            const glm::vec3 newMinimum = glm::vec3(source.bounds) - source.bounds.w;
            const glm::vec3 newMaximum = glm::vec3(source.bounds) + source.bounds.w;
            const glm::vec3 minimum = glm::min(oldMinimum, newMinimum);
            const glm::vec3 maximum = glm::max(oldMaximum, newMaximum);
            sceneCenter = (minimum + maximum) * 0.5f;
            sceneRadius = std::max(glm::distance(sceneCenter, maximum), 1.0f);
            return index;
        }

        std::uint32_t appendSkinPalette(std::span<const glm::mat4> transforms)
        {
            if (transforms.empty())
                throw std::runtime_error("Skin palette cannot be empty");
            if (skinData.size() + transforms.size() > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("Skin palette index exceeds 32 bits");
            const auto offset = static_cast<std::uint32_t>(skinData.size());
            skinData.insert(skinData.end(), transforms.begin(), transforms.end());
            skinDataDirty = true;
            return offset;
        }

        void updateSkinPalette(std::uint32_t offset, std::span<const glm::mat4> transforms)
        {
            if (transforms.empty()
                || static_cast<std::size_t>(offset) + transforms.size() > skinData.size())
                throw std::runtime_error("Skin palette update is out of bounds");
            std::copy(transforms.begin(), transforms.end(), skinData.begin() + offset);
            skinDataDirty = true;
        }

        void setActive(std::span<const std::uint32_t> instanceIndices, bool active)
        {
            const std::uint32_t value = active ? 1u : 0u;
            const std::size_t pendingBase = instanceData.size() - pendingInstances.size();
            for (const std::uint32_t index : instanceIndices)
            {
                if (index >= instanceData.size())
                    throw std::runtime_error("Streamed instance index is out of bounds");
                if (instanceData[index].meshletRange.z != value)
                {
                    instanceData[index].meshletRange.z = value;
                    if (index >= pendingBase)
                        pendingInstances[index - pendingBase].meshletRange.z = value;
                    else
                        pendingInstanceUpdates.push_back(index);
                }
            }
        }

        void setSelection(std::span<const std::uint32_t> instanceIndices,
            std::optional<std::uint32_t> active)
        {
            std::vector<std::uint32_t> selected(instanceIndices.begin(), instanceIndices.end());
            std::sort(selected.begin(), selected.end());
            selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
            if (!selected.empty() && selected.back() >= instanceData.size())
                throw std::runtime_error("Selected instance index is out of bounds");
            if (active && !std::binary_search(selected.begin(), selected.end(), *active))
                throw std::runtime_error("Active instance is not selected");
            if (selected == selectedInstances && active == activeSelection)
                return;

            std::vector<std::uint32_t> changed = selectedInstances;
            changed.insert(changed.end(), selected.begin(), selected.end());
            std::sort(changed.begin(), changed.end());
            changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
            const std::size_t pendingBase = instanceData.size() - pendingInstances.size();
            for (const std::uint32_t index : changed)
            {
                const std::uint32_t category = active == index ? 2u
                    : std::binary_search(selected.begin(), selected.end(), index) ? 1u : 0u;
                // skinRange.z is transient selection metadata, not part of SceneInstance/ESM.
                if (instanceData[index].skinRange.z == category)
                    continue;
                instanceData[index].skinRange.z = category;
                if (index >= pendingBase)
                    pendingInstances[index - pendingBase].skinRange.z = category;
                else
                    pendingInstanceUpdates.push_back(index);
            }
            selectedInstances = std::move(selected);
            activeSelection = active;
        }

        void updateTransform(std::uint32_t index, const glm::mat4& transform, glm::vec4 bounds)
        {
            if (index >= instanceData.size())
                throw std::runtime_error("Streamed instance index is out of bounds");
            bounds.x = -bounds.x;
            instanceData[index].transform = transform;
            instanceData[index].bounds = bounds;
            const std::size_t pendingBase = instanceData.size() - pendingInstances.size();
            if (index >= pendingBase)
                pendingInstances[index - pendingBase] = instanceData[index];
            else
                pendingInstanceUpdates.push_back(index);
            sceneRadius = std::max(sceneRadius,
                glm::distance(sceneCenter, glm::vec3(bounds)) + bounds.w);
        }

        bool recordStreamingUploads(Frame& frame, VkCommandBuffer command)
        {
            const bool hasUploads = !pendingVertices.empty() || !pendingIndices.empty()
                || !pendingMeshlets.empty() || !pendingInstances.empty() || !pendingSources.empty()
                || !pendingMaterials.empty() || !pendingTextures.empty()
                || !pendingInstanceUpdates.empty() || !pendingMeshPatches.empty() || skinDataDirty;
            if (!hasUploads)
            {
                if (frame.descriptorGeneration != descriptorGeneration)
                    updateDescriptorSet(frame);
                return false;
            }

            ensureDescriptorCapacity(static_cast<std::uint32_t>(textureTargets.size()));
            bool buffersChanged = false;
            buffersChanged |= ensureBuffer(vertices,
                static_cast<VkDeviceSize>(vertexCount) * sizeof(GpuVertex),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            buffersChanged |= ensureBuffer(indices,
                static_cast<VkDeviceSize>(indexCount) * sizeof(std::uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            buffersChanged |= ensureBuffer(meshlets,
                static_cast<VkDeviceSize>(meshletCount) * sizeof(GpuMeshlet),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            buffersChanged |= ensureBuffer(instances,
                static_cast<VkDeviceSize>(instanceCount) * sizeof(GpuInstance),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            buffersChanged |= ensureBuffer(sources,
                static_cast<VkDeviceSize>(candidateCount) * sizeof(glm::uvec2),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            buffersChanged |= ensureBuffer(materials,
                static_cast<VkDeviceSize>(materialCount) * sizeof(GpuMaterial),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            buffersChanged |= ensureBuffer(skinMatrices,
                static_cast<VkDeviceSize>(skinData.size()) * sizeof(glm::mat4),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            const VkDeviceSize visibleSize = static_cast<VkDeviceSize>(candidateCount) * sizeof(glm::uvec2);
            for (auto& targetFrame : frames)
                buffersChanged |= ensureBuffer(
                    targetFrame.visible, visibleSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
            if (buffersChanged)
            {
                ++descriptorGeneration;
                updateDescriptorSets();
            }

            struct BufferUpload
            {
                Buffer* destination = nullptr;
                VkDeviceSize destinationOffset = 0;
                VkDeviceSize stagingOffset = 0;
                VkDeviceSize size = 0;
                const void* data = nullptr;
            };
            std::vector<BufferUpload> uploads;
            VkDeviceSize stagingSize = 0;
            auto addUpload = [&](Buffer& destination, VkDeviceSize destinationOffset,
                                 const void* data, VkDeviceSize size) {
                if (size == 0)
                    return;
                stagingSize = (stagingSize + 15) & ~VkDeviceSize(15);
                uploads.push_back({&destination, destinationOffset, stagingSize, size, data});
                stagingSize += size;
            };
            addUpload(vertices,
                static_cast<VkDeviceSize>(vertexCount - pendingVertices.size()) * sizeof(GpuVertex),
                pendingVertices.data(), pendingVertices.size() * sizeof(GpuVertex));
            addUpload(indices,
                static_cast<VkDeviceSize>(indexCount - pendingIndices.size()) * sizeof(std::uint32_t),
                pendingIndices.data(), pendingIndices.size() * sizeof(std::uint32_t));
            addUpload(meshlets,
                static_cast<VkDeviceSize>(meshletCount - pendingMeshlets.size()) * sizeof(GpuMeshlet),
                pendingMeshlets.data(), pendingMeshlets.size() * sizeof(GpuMeshlet));
            for (const auto& patch : pendingMeshPatches)
            {
                addUpload(vertices,
                    static_cast<VkDeviceSize>(patch.range.vertexOffset) * sizeof(GpuVertex),
                    patch.mesh.vertices.data(), patch.mesh.vertices.size() * sizeof(GpuVertex));
                addUpload(meshlets,
                    static_cast<VkDeviceSize>(patch.range.meshletOffset) * sizeof(GpuMeshlet),
                    patch.mesh.meshlets.data(), patch.mesh.meshlets.size() * sizeof(GpuMeshlet));
            }
            std::sort(pendingInstanceUpdates.begin(), pendingInstanceUpdates.end());
            pendingInstanceUpdates.erase(
                std::unique(pendingInstanceUpdates.begin(), pendingInstanceUpdates.end()),
                pendingInstanceUpdates.end());
            constexpr std::size_t fullInstanceUploadThreshold = 2048;
            if (pendingInstanceUpdates.size() >= fullInstanceUploadThreshold)
            {
                addUpload(instances, 0, instanceData.data(),
                    instanceData.size() * sizeof(GpuInstance));
            }
            else
            {
                addUpload(instances,
                    static_cast<VkDeviceSize>(instanceCount - pendingInstances.size())
                        * sizeof(GpuInstance),
                    pendingInstances.data(), pendingInstances.size() * sizeof(GpuInstance));
                for (std::size_t first = 0; first < pendingInstanceUpdates.size();)
                {
                    std::size_t last = first + 1;
                    while (last < pendingInstanceUpdates.size()
                        && pendingInstanceUpdates[last] == pendingInstanceUpdates[last - 1] + 1)
                        ++last;
                    const std::uint32_t instanceIndex = pendingInstanceUpdates[first];
                    const std::size_t count = last - first;
                    addUpload(instances,
                        static_cast<VkDeviceSize>(instanceIndex) * sizeof(GpuInstance),
                        instanceData.data() + instanceIndex, count * sizeof(GpuInstance));
                    first = last;
                }
            }
            addUpload(sources,
                static_cast<VkDeviceSize>(candidateCount - pendingSources.size()) * sizeof(glm::uvec2),
                pendingSources.data(), pendingSources.size() * sizeof(glm::uvec2));
            addUpload(materials,
                static_cast<VkDeviceSize>(materialCount - pendingMaterials.size()) * sizeof(GpuMaterial),
                pendingMaterials.data(), pendingMaterials.size() * sizeof(GpuMaterial));
            if (skinDataDirty)
                addUpload(skinMatrices, 0, skinData.data(), skinData.size() * sizeof(glm::mat4));

            const std::size_t firstTexture = textureTargets.size() - pendingTextures.size();
            std::vector<VkDeviceSize> textureOffsets;
            std::vector<std::size_t> textureSourceOffsets;
            textureOffsets.reserve(pendingTextures.size());
            textureSourceOffsets.reserve(pendingTextures.size());
            std::vector<TextureTarget*> sparseTargets;
            sparseTargets.reserve(pendingTextures.size());
            for (std::size_t index = 0; index < pendingTextures.size(); ++index)
            {
                const auto& texture = pendingTextures[index];
                auto& target = textureTargets[firstTexture + index];
                const std::size_t sourceOffset = texture.mipOffsets[target.firstResidentMip];
                if (sourceOffset >= texture.pixels.size())
                    throw std::runtime_error("Streamed texture resident mip data is truncated");
                stagingSize = (stagingSize + 15) & ~VkDeviceSize(15);
                textureOffsets.push_back(stagingSize);
                textureSourceOffsets.push_back(sourceOffset);
                stagingSize += texture.pixels.size() - sourceOffset;
                sparseTargets.push_back(&target);
            }
            const bool waitForSparse = submitSparseBinds(sparseTargets, frame.sparseReady);

            Buffer staging = createBuffer(stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            for (const auto& upload : uploads)
                std::memcpy(static_cast<std::byte*>(staging.mapped) + upload.stagingOffset,
                    upload.data, static_cast<std::size_t>(upload.size));
            for (std::size_t index = 0; index < pendingTextures.size(); ++index)
                std::memcpy(static_cast<std::byte*>(staging.mapped) + textureOffsets[index],
                    pendingTextures[index].pixels.data() + textureSourceOffsets[index],
                    pendingTextures[index].pixels.size() - textureSourceOffsets[index]);
            check(vmaFlushAllocation(allocator, staging.allocation, 0, VK_WHOLE_SIZE),
                "vmaFlushAllocation streaming batch");

            // Scene buffers are shared by both in-flight frames: wait for earlier
            // shader reads before overwriting existing geometry/instance/skin data.
            if (!pendingMeshPatches.empty() || !pendingInstanceUpdates.empty() || skinDataDirty)
            {
                VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                dependency.memoryBarrierCount = 1;
                dependency.pMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(command, &dependency);
            }
            for (const auto& upload : uploads)
            {
                const VkBufferCopy copy{upload.stagingOffset, upload.destinationOffset, upload.size};
                vkCmdCopyBuffer(command, staging.handle, upload.destination->handle, 1, &copy);
            }
            for (std::size_t index = 0; index < pendingTextures.size(); ++index)
            {
                const auto& texture = pendingTextures[index];
                auto& target = textureTargets[firstTexture + index];
                imageBarrier(command, target.image, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                    texture.mipLevels - target.firstResidentMip, target.firstResidentMip);
                for (std::uint32_t mip = target.firstResidentMip;
                    mip < texture.mipLevels; ++mip)
                {
                    VkBufferImageCopy copy{};
                    copy.bufferOffset = textureOffsets[index] + texture.mipOffsets[mip]
                        - textureSourceOffsets[index];
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
                    copy.imageExtent = {
                        std::max(1u, texture.width >> mip), std::max(1u, texture.height >> mip), 1};
                    vkCmdCopyBufferToImage(command, staging.handle, target.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
                }
                imageBarrier(command, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                    texture.mipLevels - target.firstResidentMip, target.firstResidentMip);
            }
            if (!uploads.empty())
            {
                VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                dependency.memoryBarrierCount = 1;
                dependency.pMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(command, &dependency);
            }

            frame.retiredUploads.push_back(staging);
            if (!pendingTextures.empty())
                ++descriptorGeneration;
            pendingVertices.clear();
            pendingIndices.clear();
            pendingMeshlets.clear();
            pendingInstances.clear();
            pendingInstanceUpdates.clear();
            pendingSources.clear();
            pendingMaterials.clear();
            pendingTextures.clear();
            pendingMeshPatches.clear();
            skinDataDirty = false;
            if (frame.descriptorGeneration != descriptorGeneration)
                updateDescriptorSet(frame);
            return waitForSparse;
        }

        VkShaderModule shaderModule(const std::vector<std::uint32_t>& code)
        {
            VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            createInfo.codeSize = code.size() * sizeof(std::uint32_t);
            createInfo.pCode = code.data();
            VkShaderModule result = VK_NULL_HANDLE;
            check(vkCreateShaderModule(device, &createInfo, nullptr, &result), "vkCreateShaderModule");
            return result;
        }

        void createPipelines()
        {
            const auto computeCode = compileShader(computeShader, shaderc_compute_shader, "cull.comp");
            const auto meshCode = compileShader(meshShader, shaderc_mesh_shader, "scene.mesh");
            const auto fragmentCode = compileShader(fragmentShader, shaderc_fragment_shader, "scene.frag");
            const auto shadowCode = compileShader(shadowMeshShader, shaderc_mesh_shader, "shadow.mesh");
            const VkShaderModule computeModule = shaderModule(computeCode);
            const VkShaderModule meshModule = shaderModule(meshCode);
            const VkShaderModule fragmentModule = shaderModule(fragmentCode);
            const VkShaderModule shadowModule = shaderModule(shadowCode);

            VkComputePipelineCreateInfo computeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            computeInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            computeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeInfo.stage.module = computeModule;
            computeInfo.stage.pName = "main";
            computeInfo.layout = pipelineLayout;
            check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, nullptr, &computePipeline),
                "vkCreateComputePipelines");

            const VkSpecializationMapEntry decalPassEntry{0, 0, sizeof(std::uint32_t)};
            const std::uint32_t regularPassValue = 0;
            const std::uint32_t decalPassValue = 1;
            const VkSpecializationInfo regularPassSpecialization{
                1, &decalPassEntry, sizeof(regularPassValue), &regularPassValue};
            const VkSpecializationInfo decalPassSpecialization{
                1, &decalPassEntry, sizeof(decalPassValue), &decalPassValue};
            const std::array stages{
                VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule, "main",
                    &regularPassSpecialization},
                VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, "main", nullptr}
            };
            VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo inputAssembly{
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewport.viewportCount = 1;
            viewport.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rasterization{
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rasterization.polygonMode = VK_POLYGON_MODE_FILL;
            rasterization.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasterization.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo multisample{
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            depth.depthTestEnable = VK_TRUE;
            depth.depthWriteEnable = VK_TRUE;
            depth.depthCompareOp = VK_COMPARE_OP_GREATER;
            VkPipelineColorBlendAttachmentState blendAttachment{};
            blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blend.attachmentCount = 1;
            blend.pAttachments = &blendAttachment;
            constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
            dynamic.pDynamicStates = dynamicStates.data();
            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &swapchainFormat;
            rendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
            VkGraphicsPipelineCreateInfo graphicsInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
            graphicsInfo.stageCount = static_cast<std::uint32_t>(stages.size());
            graphicsInfo.pStages = stages.data();
            graphicsInfo.pVertexInputState = &vertexInput;
            graphicsInfo.pInputAssemblyState = &inputAssembly;
            graphicsInfo.pViewportState = &viewport;
            graphicsInfo.pRasterizationState = &rasterization;
            graphicsInfo.pMultisampleState = &multisample;
            graphicsInfo.pDepthStencilState = &depth;
            graphicsInfo.pColorBlendState = &blend;
            graphicsInfo.pDynamicState = &dynamic;
            graphicsInfo.layout = pipelineLayout;
            check(vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1, &graphicsInfo, nullptr, &graphicsPipeline),
                "vkCreateGraphicsPipelines");

            auto decalStages = stages;
            decalStages[0].pSpecializationInfo = &decalPassSpecialization;
            VkPipelineRasterizationStateCreateInfo decalRasterization = rasterization;
            decalRasterization.depthBiasEnable = VK_TRUE;
            decalRasterization.depthBiasConstantFactor = 1.0f;
            decalRasterization.depthBiasSlopeFactor = 1.0f;
            VkPipelineDepthStencilStateCreateInfo decalDepth = depth;
            decalDepth.depthWriteEnable = VK_FALSE;
            VkPipelineColorBlendAttachmentState decalBlendAttachment = blendAttachment;
            decalBlendAttachment.blendEnable = VK_TRUE;
            decalBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            decalBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            decalBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            decalBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            decalBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            decalBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
            VkPipelineColorBlendStateCreateInfo decalBlend = blend;
            decalBlend.pAttachments = &decalBlendAttachment;
            VkGraphicsPipelineCreateInfo decalInfo = graphicsInfo;
            decalInfo.stageCount = static_cast<std::uint32_t>(decalStages.size());
            decalInfo.pStages = decalStages.data();
            decalInfo.pRasterizationState = &decalRasterization;
            decalInfo.pDepthStencilState = &decalDepth;
            decalInfo.pColorBlendState = &decalBlend;
            check(vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1, &decalInfo, nullptr, &decalPipeline),
                "vkCreateGraphicsPipelines decal");

            VkPipelineShaderStageCreateInfo shadowStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            shadowStage.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
            shadowStage.module = shadowModule;
            shadowStage.pName = "main";
            VkPipelineRasterizationStateCreateInfo shadowRasterization = rasterization;
            shadowRasterization.depthBiasEnable = VK_TRUE;
            shadowRasterization.depthBiasConstantFactor = 1.25f;
            shadowRasterization.depthBiasSlopeFactor = 1.75f;
            VkPipelineColorBlendStateCreateInfo shadowBlend{
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            VkPipelineDepthStencilStateCreateInfo shadowDepthState = depth;
            shadowDepthState.depthCompareOp = VK_COMPARE_OP_LESS;
            VkPipelineRenderingCreateInfo shadowRendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            shadowRendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
            VkGraphicsPipelineCreateInfo shadowInfo = graphicsInfo;
            shadowInfo.pNext = &shadowRendering;
            shadowInfo.stageCount = 1;
            shadowInfo.pStages = &shadowStage;
            shadowInfo.pRasterizationState = &shadowRasterization;
            shadowInfo.pDepthStencilState = &shadowDepthState;
            shadowInfo.pColorBlendState = &shadowBlend;
            check(vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1, &shadowInfo, nullptr, &shadowPipeline),
                "vkCreateGraphicsPipelines shadow");

            vkDestroyShaderModule(device, shadowModule, nullptr);
            vkDestroyShaderModule(device, fragmentModule, nullptr);
            vkDestroyShaderModule(device, meshModule, nullptr);
            vkDestroyShaderModule(device, computeModule, nullptr);
        }

        void createSelectionPipelines()
        {
            const VkShaderModule meshModule = shaderModule(
                compileShader(meshShader, shaderc_mesh_shader, "selection.mesh"));
            const VkShaderModule fragmentModule = shaderModule(
                compileShader(fragmentShader, shaderc_fragment_shader, "selection.frag"));
            const VkShaderModule vertexModule = shaderModule(
                compileShader(outlineVertexShader, shaderc_vertex_shader, "outline.vert"));
            const VkShaderModule outlineModule = shaderModule(
                compileShader(outlineFragmentShader, shaderc_fragment_shader, "outline.frag"));
            const VkSpecializationMapEntry meshEntry{0, 0, sizeof(std::uint32_t)};
            const VkSpecializationMapEntry fragmentEntry{1, 0, sizeof(std::uint32_t)};
            const std::uint32_t maskPass = 2;
            const VkSpecializationInfo meshSpecialization{1, &meshEntry, sizeof(maskPass), &maskPass};
            const VkSpecializationInfo fragmentSpecialization{1, &fragmentEntry, sizeof(maskPass), &maskPass};
            std::array stages{
                VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule, "main", &meshSpecialization},
                VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, "main", &fragmentSpecialization}
            };
            VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewport.viewportCount = viewport.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rasterization.polygonMode = VK_POLYGON_MODE_FILL;
            rasterization.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasterization.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            depth.depthTestEnable = VK_TRUE;
            depth.depthWriteEnable = VK_FALSE;
            depth.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
            VkPipelineColorBlendAttachmentState attachment{};
            attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
            VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blend.attachmentCount = 1;
            blend.pAttachments = &attachment;
            constexpr std::array states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(states.size());
            dynamic.pDynamicStates = states.data();
            const VkFormat maskFormat = VK_FORMAT_R8_UNORM;
            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &maskFormat;
            rendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
            VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
            info.stageCount = static_cast<std::uint32_t>(stages.size());
            info.pStages = stages.data();
            info.pVertexInputState = &vertexInput;
            info.pInputAssemblyState = &assembly;
            info.pViewportState = &viewport;
            info.pRasterizationState = &rasterization;
            info.pMultisampleState = &multisample;
            info.pDepthStencilState = &depth;
            info.pColorBlendState = &blend;
            info.pDynamicState = &dynamic;
            info.layout = pipelineLayout;
            check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                &selectionPipeline), "vkCreateGraphicsPipelines selection");

            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertexModule;
            stages[0].pSpecializationInfo = nullptr;
            stages[1].module = outlineModule;
            stages[1].pSpecializationInfo = nullptr;
            rasterization.cullMode = VK_CULL_MODE_NONE;
            depth.depthTestEnable = VK_FALSE;
            attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            rendering.pColorAttachmentFormats = &swapchainFormat;
            info.layout = outlinePipelineLayout;
            check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                &outlinePipeline), "vkCreateGraphicsPipelines outline");
            vkDestroyShaderModule(device, outlineModule, nullptr);
            vkDestroyShaderModule(device, vertexModule, nullptr);
            vkDestroyShaderModule(device, fragmentModule, nullptr);
            vkDestroyShaderModule(device, meshModule, nullptr);
        }

        void destroySelectionTargets()
        {
            if (selectionDescriptorPool)
                vkDestroyDescriptorPool(device, selectionDescriptorPool, nullptr);
            selectionDescriptorPool = VK_NULL_HANDLE;
            for (auto& frame : frames)
            {
                if (frame.selectionMask.view)
                    vkDestroyImageView(device, frame.selectionMask.view, nullptr);
                if (frame.selectionMask.image)
                    vmaDestroyImage(allocator, frame.selectionMask.image, frame.selectionMask.allocation);
                frame.selectionMask = {};
                frame.selectionDescriptorSet = VK_NULL_HANDLE;
            }
        }

        void ensureSelectionResources()
        {
            if (selectionDescriptorPool)
                return;
            if (!selectionDescriptorSetLayout)
            {
                VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
                VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                layoutInfo.bindingCount = 1;
                layoutInfo.pBindings = &binding;
                check(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                    &selectionDescriptorSetLayout), "vkCreateDescriptorSetLayout selection");
                VkPipelineLayoutCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
                pipelineInfo.setLayoutCount = 1;
                pipelineInfo.pSetLayouts = &selectionDescriptorSetLayout;
                check(vkCreatePipelineLayout(device, &pipelineInfo, nullptr,
                    &outlinePipelineLayout), "vkCreatePipelineLayout outline");
                VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_NEAREST;
                samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW
                    = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                check(vkCreateSampler(device, &samplerInfo, nullptr,
                    &selectionSampler), "vkCreateSampler selection");
                createSelectionPipelines();
            }
            const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frameCount};
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets = frameCount;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &size;
            check(vkCreateDescriptorPool(device, &poolInfo, nullptr,
                &selectionDescriptorPool), "vkCreateDescriptorPool selection");
            for (auto& frame : frames)
            {
                VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                imageInfo.imageType = VK_IMAGE_TYPE_2D;
                imageInfo.format = VK_FORMAT_R8_UNORM;
                imageInfo.extent = {extent.width, extent.height, 1};
                imageInfo.mipLevels = imageInfo.arrayLayers = 1;
                imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
                imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
                imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                VmaAllocationCreateInfo allocationInfo{};
                allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                check(vmaCreateImage(allocator, &imageInfo, &allocationInfo,
                    &frame.selectionMask.image, &frame.selectionMask.allocation, nullptr),
                    "vmaCreateImage selection");
                VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                viewInfo.image = frame.selectionMask.image;
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = VK_FORMAT_R8_UNORM;
                viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                check(vkCreateImageView(device, &viewInfo, nullptr,
                    &frame.selectionMask.view), "vkCreateImageView selection");
                VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                allocateInfo.descriptorPool = selectionDescriptorPool;
                allocateInfo.descriptorSetCount = 1;
                allocateInfo.pSetLayouts = &selectionDescriptorSetLayout;
                check(vkAllocateDescriptorSets(device, &allocateInfo,
                    &frame.selectionDescriptorSet), "vkAllocateDescriptorSets selection");
                VkDescriptorImageInfo descriptor{selectionSampler, frame.selectionMask.view,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = frame.selectionDescriptorSet;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write.pImageInfo = &descriptor;
                vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
            }
        }

        void imageBarrier(VkCommandBuffer command, VkImage image, VkImageLayout oldLayout,
            VkImageLayout newLayout, VkImageAspectFlags aspect, std::uint32_t levelCount = 1,
            std::uint32_t baseMipLevel = 0)
        {
            VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            barrier.srcStageMask = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                ? VK_PIPELINE_STAGE_2_NONE : oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                    : oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                        ? VK_PIPELINE_STAGE_2_TRANSFER_BIT : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            // First-use swapchain transitions must execute after the acquire
            // semaphore wait, even though there is no previous image contents.
            if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barrier.srcAccessMask = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                ? VK_ACCESS_2_NONE : oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                    : oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                        ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_MEMORY_WRITE_BIT;
            barrier.dstStageMask = newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                ? VK_PIPELINE_STAGE_2_NONE : newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                    : newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                        ? VK_PIPELINE_STAGE_2_TRANSFER_BIT
                    : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
                        | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                        | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            barrier.dstAccessMask = newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                ? VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
                : newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                    ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                    : newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                        ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                        : newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                            ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_NONE;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = aspect;
            barrier.subresourceRange.baseMipLevel = baseMipLevel;
            barrier.subresourceRange.levelCount = levelCount;
            barrier.subresourceRange.layerCount = 1;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.imageMemoryBarrierCount = 1;
            dependency.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command, &dependency);
        }

        void render(const glm::mat4& view, const glm::mat4& projection,
            const glm::vec3& cameraPosition, const glm::vec3& worldOrigin,
            std::optional<RenderViewport> requestedViewport, const UiDrawCallback& drawUi)
        {
            const auto cpuStart = std::chrono::steady_clock::now();
            Frame& frame = frames[currentFrame];
            check(vkWaitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX), "frame fence");
            if (frame.submitted)
            {
                std::array<std::uint64_t, 2> timestamps{};
                if (vkGetQueryPoolResults(device, queryPool, currentFrame * 2, 2,
                    sizeof(timestamps), timestamps.data(), sizeof(std::uint64_t),
                    VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
                    rendererStats.gpuFrameMs = static_cast<double>(timestamps[1] - timestamps[0])
                        * timestampPeriod / 1'000'000.0;
                check(vmaInvalidateAllocation(allocator, frame.indirect.allocation, 0, sizeof(std::uint32_t)),
                    "vmaInvalidateAllocation");
                rendererStats.visibleMeshlets = std::min(
                    *static_cast<const std::uint32_t*>(frame.indirect.mapped), candidateCount);
            }
            for (auto& upload : frame.retiredUploads)
                destroyBuffer(upload);
            frame.retiredUploads.clear();

            std::uint32_t imageIndex = 0;
            const VkResult acquire = vkAcquireNextImageKHR(
                device, swapchain, UINT64_MAX, frame.acquired, VK_NULL_HANDLE, &imageIndex);
            if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
            {
                recreateSwapchain();
                return;
            }
            if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
                check(acquire, "vkAcquireNextImageKHR");
            check(vkResetCommandBuffer(frame.commandBuffer, 0), "vkResetCommandBuffer");

            const auto clampedOffset = [](std::int32_t offset, std::uint32_t limit) {
                return static_cast<std::uint32_t>(std::clamp(offset, 0,
                    static_cast<std::int32_t>(limit)));
            };
            const std::uint32_t viewportX = requestedViewport
                ? clampedOffset(requestedViewport->x, extent.width) : 0;
            const std::uint32_t viewportY = requestedViewport
                ? clampedOffset(requestedViewport->y, extent.height) : 0;
            const std::uint32_t viewportWidth = std::min(
                requestedViewport ? requestedViewport->width : extent.width,
                extent.width - viewportX);
            const std::uint32_t viewportHeight = std::min(
                requestedViewport ? requestedViewport->height : extent.height,
                extent.height - viewportY);
            const bool drawWorld = viewportWidth != 0 && viewportHeight != 0;
            const bool drawSelection = drawWorld && !selectedInstances.empty();
            if (drawSelection)
                ensureSelectionResources();

            CameraData camera;
            camera.viewProjection = projection * view;
            const glm::vec3 lightDirection = glm::normalize(glm::vec3(0.35f, 0.8f, 0.25f));
            const glm::vec3 localSceneCenter = sceneCenter - worldOrigin;
            const glm::mat4 lightView = glm::lookAtRH(
                localSceneCenter + lightDirection * sceneRadius * 2.5f,
                localSceneCenter, glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 lightProjection = glm::orthoRH_ZO(-sceneRadius, sceneRadius,
                -sceneRadius, sceneRadius, 0.1f, sceneRadius * 5.0f);
            lightProjection[1][1] *= -1.0f;
            camera.lightViewProjection = lightProjection * lightView;
            camera.planes = extractFrustum(camera.viewProjection);
            camera.cameraPosition = glm::vec4(cameraPosition, 1.0f);
            camera.worldOrigin = glm::vec4(worldOrigin, 0.0f);
            int logicalWidth = 0;
            int logicalHeight = 0;
            SDL_GetWindowSize(window, &logicalWidth, &logicalHeight);
            camera.counts = glm::uvec4(candidateCount, candidateCount,
                requestedViewport ? std::max(viewportWidth, 1u)
                                  : static_cast<std::uint32_t>(std::max(logicalWidth, 1)),
                requestedViewport ? std::max(viewportHeight, 1u)
                                  : static_cast<std::uint32_t>(std::max(logicalHeight, 1)));
            std::memcpy(frame.uniform.mapped, &camera, sizeof(camera));
            check(vmaFlushAllocation(allocator, frame.uniform.allocation, 0, sizeof(camera)),
                "vmaFlushAllocation");

            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(frame.commandBuffer, &beginInfo), "vkBeginCommandBuffer");
            vkCmdResetQueryPool(frame.commandBuffer, queryPool, currentFrame * 2, 2);
            vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                queryPool, currentFrame * 2);
            const bool waitForSparse = recordStreamingUploads(frame, frame.commandBuffer);
            const VkDrawMeshTasksIndirectCommandEXT indirectCommand{0, 1, 1};
            vkCmdUpdateBuffer(frame.commandBuffer, frame.indirect.handle, 0,
                sizeof(indirectCommand), &indirectCommand);
            VkBufferMemoryBarrier2 resetBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
            resetBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            resetBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            resetBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            resetBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            resetBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            resetBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            resetBarrier.buffer = frame.indirect.handle;
            resetBarrier.size = VK_WHOLE_SIZE;
            VkDependencyInfo resetDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            resetDependency.bufferMemoryBarrierCount = 1;
            resetDependency.pBufferMemoryBarriers = &resetBarrier;
            vkCmdPipelineBarrier2(frame.commandBuffer, &resetDependency);

            vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
            vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelineLayout, 0, 1, &frame.descriptorSet, 0, nullptr);
            vkCmdDispatch(frame.commandBuffer, (candidateCount + 63) / 64, 1, 1);
            std::array<VkBufferMemoryBarrier2, 2> cullBarriers{};
            for (auto& barrier : cullBarriers)
            {
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                    | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
                barrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
                    | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.size = VK_WHOLE_SIZE;
            }
            cullBarriers[0].buffer = frame.indirect.handle;
            cullBarriers[1].buffer = frame.visible.handle;
            VkDependencyInfo cullDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            cullDependency.bufferMemoryBarrierCount = static_cast<std::uint32_t>(cullBarriers.size());
            cullDependency.pBufferMemoryBarriers = cullBarriers.data();
            vkCmdPipelineBarrier2(frame.commandBuffer, &cullDependency);

            imageBarrier(frame.commandBuffer, shadowTarget.image,
                shadowTarget.initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                         : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            shadowTarget.initialized = true;
            VkRenderingAttachmentInfo shadowDepth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            shadowDepth.imageView = shadowTarget.view;
            shadowDepth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            shadowDepth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            shadowDepth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            shadowDepth.clearValue.depthStencil = {1.0f, 0};
            VkRenderingInfo shadowRendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
            shadowRendering.renderArea.extent = shadowExtent;
            shadowRendering.layerCount = 1;
            shadowRendering.pDepthAttachment = &shadowDepth;
            vkCmdBeginRendering(frame.commandBuffer, &shadowRendering);
            const VkViewport shadowViewport{0.0f, 0.0f, static_cast<float>(shadowExtent.width),
                static_cast<float>(shadowExtent.height), 0.0f, 1.0f};
            const VkRect2D shadowScissor{{0, 0}, shadowExtent};
            vkCmdSetViewport(frame.commandBuffer, 0, 1, &shadowViewport);
            vkCmdSetScissor(frame.commandBuffer, 0, 1, &shadowScissor);
            vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
            vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout, 0, 1, &frame.descriptorSet, 0, nullptr);
            drawMeshTasksIndirect(frame.commandBuffer, frame.indirect.handle, 0, 1,
                sizeof(VkDrawMeshTasksIndirectCommandEXT));
            vkCmdEndRendering(frame.commandBuffer);
            imageBarrier(frame.commandBuffer, shadowTarget.image,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

            imageBarrier(frame.commandBuffer, swapchainImages[imageIndex], swapchainLayouts[imageIndex],
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
            swapchainLayouts[imageIndex] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            DepthTarget& depth = depthTargets[imageIndex];
            if (!depth.initialized)
            {
                imageBarrier(frame.commandBuffer, depth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
                depth.initialized = true;
            }

            VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            color.imageView = swapchainViews[imageIndex];
            color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color.clearValue.color = {{0.055f, 0.075f, 0.09f, 1.0f}};
            VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            depthAttachment.imageView = depth.view;
            depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = drawSelection
                ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.clearValue.depthStencil = {0.0f, 0};
            VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
            rendering.renderArea.extent = extent;
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color;
            rendering.pDepthAttachment = &depthAttachment;
            vkCmdBeginRendering(frame.commandBuffer, &rendering);
            if (drawWorld)
            {
                const VkViewport viewport{static_cast<float>(viewportX), static_cast<float>(viewportY),
                    static_cast<float>(viewportWidth), static_cast<float>(viewportHeight), 0.0f, 1.0f};
                const VkRect2D scissor{{static_cast<std::int32_t>(viewportX),
                    static_cast<std::int32_t>(viewportY)}, {viewportWidth, viewportHeight}};
                vkCmdSetViewport(frame.commandBuffer, 0, 1, &viewport);
                vkCmdSetScissor(frame.commandBuffer, 0, 1, &scissor);
                vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
                vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipelineLayout, 0, 1, &frame.descriptorSet, 0, nullptr);
                drawMeshTasksIndirect(frame.commandBuffer, frame.indirect.handle, 0, 1,
                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
                vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, decalPipeline);
                drawMeshTasksIndirect(frame.commandBuffer, frame.indirect.handle, 0, 1,
                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
            }
            if (drawSelection)
            {
                vkCmdEndRendering(frame.commandBuffer);
                // Preserve scene visibility, including LAND, water and unselected objects.
                // Reuse the scene geometry/shader alpha and distance discard paths.
                imageBarrier(frame.commandBuffer, depth.image,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
                imageBarrier(frame.commandBuffer, frame.selectionMask.image,
                    frame.selectionMask.initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                    : VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
                frame.selectionMask.initialized = true;
                VkRenderingAttachmentInfo maskColor{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                maskColor.imageView = frame.selectionMask.view;
                maskColor.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                maskColor.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                maskColor.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                VkRenderingInfo maskRendering = rendering;
                maskRendering.pColorAttachments = &maskColor;
                vkCmdBeginRendering(frame.commandBuffer, &maskRendering);
                vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, selectionPipeline);
                drawMeshTasksIndirect(frame.commandBuffer, frame.indirect.handle, 0, 1,
                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
                vkCmdEndRendering(frame.commandBuffer);
                imageBarrier(frame.commandBuffer, frame.selectionMask.image,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
                imageBarrier(frame.commandBuffer, depth.image,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
                imageBarrier(frame.commandBuffer, swapchainImages[imageIndex],
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
                color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                vkCmdBeginRendering(frame.commandBuffer, &rendering);
                vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, outlinePipeline);
                vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    outlinePipelineLayout, 0, 1, &frame.selectionDescriptorSet, 0, nullptr);
                vkCmdDraw(frame.commandBuffer, 3, 1, 0, 0);
            }
            if (drawUi)
                drawUi(frame.commandBuffer);
            vkCmdEndRendering(frame.commandBuffer);

            imageBarrier(frame.commandBuffer, swapchainImages[imageIndex],
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_ASPECT_COLOR_BIT);
            swapchainLayouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                queryPool, currentFrame * 2 + 1);
            check(vkEndCommandBuffer(frame.commandBuffer), "vkEndCommandBuffer");
            check(vkResetFences(device, 1, &frame.fence), "vkResetFences");

            std::array<VkSemaphoreSubmitInfo, 2> waitInfos{};
            waitInfos[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            waitInfos[0].semaphore = frame.acquired;
            waitInfos[0].stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            waitInfos[1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            waitInfos[1].semaphore = frame.sparseReady;
            waitInfos[1].stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
            commandInfo.commandBuffer = frame.commandBuffer;
            VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            signalInfo.semaphore = presentSemaphores[imageIndex];
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            submitInfo.waitSemaphoreInfoCount = waitForSparse ? 2u : 1u;
            submitInfo.pWaitSemaphoreInfos = waitInfos.data();
            submitInfo.commandBufferInfoCount = 1;
            submitInfo.pCommandBufferInfos = &commandInfo;
            submitInfo.signalSemaphoreInfoCount = 1;
            submitInfo.pSignalSemaphoreInfos = &signalInfo;
            check(vkQueueSubmit2(queue, 1, &submitInfo, frame.fence), "vkQueueSubmit2");
            frame.submitted = true;

            VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &presentSemaphores[imageIndex];
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &swapchain;
            presentInfo.pImageIndices = &imageIndex;
            const VkResult present = vkQueuePresentKHR(queue, &presentInfo);
            if (present == VK_ERROR_OUT_OF_DATE_KHR || present == VK_SUBOPTIMAL_KHR)
                recreateSwapchain();
            else
                check(present, "vkQueuePresentKHR");

            currentFrame = (currentFrame + 1) % frameCount;
            rendererStats.cpuFrameMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cpuStart).count();
        }

        [[nodiscard]] UiContext uiContext() const noexcept
        {
            return {instance, physicalDevice, device, queue, queueFamily, swapchainFormat,
                VK_FORMAT_D32_SFLOAT,
                minSwapchainImageCount, static_cast<std::uint32_t>(swapchainImages.size()),
                swapchainGeneration};
        }
    };

    VulkanRenderer::VulkanRenderer(
        SDL_Window* window, const PackedMesh& mesh, std::span<const SceneInstance> instances,
        std::span<const SceneMaterial> materials, std::span<const SceneTexture> textures)
        : mImpl(std::make_unique<Impl>(window, mesh, instances, materials, textures))
    {
    }

    VulkanRenderer::~VulkanRenderer() = default;

    void VulkanRenderer::resize()
    {
        mImpl->recreateSwapchain();
    }

    void VulkanRenderer::render(
        const glm::mat4& view, const glm::mat4& projection,
        const glm::vec3& cameraPosition, const glm::vec3& worldOrigin,
        std::optional<RenderViewport> viewport, const UiDrawCallback& drawUi)
    {
        mImpl->render(view, projection, cameraPosition, worldOrigin, viewport, drawUi);
    }

    std::uint32_t VulkanRenderer::addTexture(SceneTexture texture)
    {
        return mImpl->appendTexture(std::move(texture));
    }

    std::uint32_t VulkanRenderer::addMaterial(const SceneMaterial& material)
    {
        return mImpl->appendMaterial(material);
    }

    MeshRange VulkanRenderer::addMesh(PackedMesh mesh)
    {
        return mImpl->appendPackedMesh(std::move(mesh));
    }

    void VulkanRenderer::updateMesh(const MeshRange& range, PackedMesh mesh)
    {
        mImpl->replacePackedMesh(range, std::move(mesh));
    }

    std::uint32_t VulkanRenderer::addInstance(SceneInstance instance)
    {
        return mImpl->appendInstance(instance);
    }

    std::uint32_t VulkanRenderer::addSkinPalette(std::span<const glm::mat4> transforms)
    {
        return mImpl->appendSkinPalette(transforms);
    }

    void VulkanRenderer::updateSkinPalette(
        std::uint32_t offset, std::span<const glm::mat4> transforms)
    {
        mImpl->updateSkinPalette(offset, transforms);
    }

    void VulkanRenderer::updateInstanceTransform(std::uint32_t instance,
        const glm::mat4& transform, const glm::vec4& bounds)
    {
        mImpl->updateTransform(instance, transform, bounds);
    }

    void VulkanRenderer::setInstancesActive(
        std::span<const std::uint32_t> instances, bool active)
    {
        mImpl->setActive(instances, active);
    }

    void VulkanRenderer::setSelection(std::span<const std::uint32_t> instances,
        std::optional<std::uint32_t> activeInstance)
    {
        mImpl->setSelection(instances, activeInstance);
    }

    const RendererStats& VulkanRenderer::stats() const noexcept
    {
        return mImpl->rendererStats;
    }

    const std::string& VulkanRenderer::deviceName() const noexcept
    {
        return mImpl->gpuName;
    }

    UiContext VulkanRenderer::uiContext() const noexcept
    {
        return mImpl->uiContext();
    }
}
