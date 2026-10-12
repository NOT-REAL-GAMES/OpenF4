#include "render/vulkan_renderer.hpp"

#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <deque>
#include <cctype>
#include <execution>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <chrono>
#include <compare>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <type_traits>
#include <utility>
#include <list>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

// MinGW/GCC: the GPU submit thread is a stdexec run_loop (P2300 std::execution) fed through
// a counting_scope; MSVC has no senders yet and uses a plain std::thread with the same contract.
#if defined(OPENF4_HAS_STDEXEC) && __has_include(<stdexec/execution.hpp>) \
    && __has_include(<exec/single_thread_context.hpp>)
#include <exec/single_thread_context.hpp>
#include <stdexec/execution.hpp>
#define OPENF4_SUBMIT_STDEXEC 1
#else
#define OPENF4_SUBMIT_STDEXEC 0
#endif

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
        constexpr std::uint32_t timestampsPerFrame = 5;
        constexpr std::uint32_t indirectCommandCount = 3; // opaque, decal, shadow
        constexpr VkDeviceSize opaqueCommandOffset = 0;
        constexpr VkDeviceSize decalCommandOffset = sizeof(VkDrawMeshTasksIndirectCommandEXT);
        constexpr VkDeviceSize shadowCommandOffset = 2 * sizeof(VkDrawMeshTasksIndirectCommandEXT);
        // Cascaded shadow maps share one depth atlas: cascade c lives in tile (c & 1, c >> 1).
        constexpr std::uint32_t maxShadowCascades = 4;

        bool environmentFlagSet(const char* name)
        {
#ifdef _WIN32
            char* value = nullptr;
            std::size_t size = 0;
            _dupenv_s(&value, &size, name);
            const bool set = value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
            std::free(value);
            return set;
#else
            const char* value = std::getenv(name);
            return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
#endif
        }

        // True only when the variable is exactly "0" (opt-out switches for default-on features).
        bool environmentFlagCleared(const char* name)
        {
#ifdef _WIN32
            char* value = nullptr;
            std::size_t size = 0;
            _dupenv_s(&value, &size, name);
            const bool cleared = value != nullptr && std::strcmp(value, "0") == 0;
            std::free(value);
            return cleared;
#else
            const char* value = std::getenv(name);
            return value != nullptr && std::strcmp(value, "0") == 0;
#endif
        }

        VKAPI_ATTR VkBool32 VKAPI_CALL validationMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
            VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* message, void*)
        {
            std::fprintf(stderr, "Vulkan Validation %s [%s]: %s\n",
                severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "Error" : "Warning",
                message && message->pMessageIdName ? message->pMessageIdName : "unknown",
                message && message->pMessage ? message->pMessage : "No validation message text");
            std::fflush(stderr);
            return VK_FALSE;
        }

        // Meshlet culling. One pass over every (instance, meshlet) candidate writes three
        // lists into the per-frame visible buffer (capacity C = counts.y, buffer holds 2C):
        //   [0, opaque)          main view, opaque      -> indirect command 0
        //   (C - decal, C - 1]   main view, decals      -> indirect command 1 (filled downwards)
        //   [C, C + shadow)      light view casters     -> indirect command 2
        // so the decal and shadow passes no longer launch a workgroup for every visible meshlet.
        const char* computeShader = R"(
#version 460
layout(local_size_x = 64) in;

struct Vertex { vec4 position; vec3 normal; uint blendWeights; vec2 texCoord; uint blendWeightsExtra; uint color; uint boneIndices; uint boneWeights; uvec2 padding; };
struct Meshlet { uint vertexOffset; uint indexOffset; uint vertexCount; uint triangleCount; vec4 sphere; uint materialIndex; uint cone; uint coneFlags; uint padding2; };
struct Instance { mat4 transform; vec4 bounds; vec4 lodRange; uvec4 meshletRange; uvec4 skinRange; };
struct Material { vec4 baseColor; uvec4 textureIndices; vec4 parameters; uvec4 extraTextureIndices; };

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
    vec4 lightPlanes[6];
    vec4 cull; // x: pixels per unit radius at unit distance (0 = off), y: min radius px, z: shadow distance, w: min caster radius
    vec4 cullFlags; // x: cone culling on, z: far end of shadow cascade 0 (0 = fixed caster threshold)
} camera;
layout(std430, binding = 9) readonly buffer Materials { Material materials[]; };

void main()
{
    uint sourceIndex = gl_GlobalInvocationID.x;
    if (sourceIndex >= camera.counts.x)
        return;

    uvec2 source = sources[sourceIndex];
    if (camera.cameraPosition.w >= 0.0 && source.x < uint(camera.cameraPosition.w))
        return;
    Instance instance = instances[source.x];
    if (instance.meshletRange.z == 0)
        return;
    Meshlet meshlet = meshlets[source.y];
    Material material = materials[meshlet.materialIndex];
    bool decal = material.parameters.w > 0.5;
    bool mainVisible = true;
    bool shadowVisible = instance.meshletRange.w == 0 && camera.cull.z > 0.0 && !decal;
    if (instance.meshletRange.w == 0) {
        vec3 center;
        float radius;
        bool skinned = instance.skinRange.y != 0;
        if (skinned) {
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
        float distanceToCamera = distance(camera.cameraPosition.xz, center.xz);
        if (distanceToCamera + radius < instance.lodRange.x
            || distanceToCamera - radius > instance.lodRange.w)
            return;

        for (uint plane = 0; plane < 6; ++plane) {
            if (dot(camera.planes[plane].xyz, center) + camera.planes[plane].w < -radius)
                mainVisible = false;
        }
        // Terrain, terrain LOD and water never drop out (no holes in the ground).
        bool ground = material.parameters.z > 0.5;
        vec3 toCenter = center - camera.cameraPosition.xyz;
        float eyeDistance = length(toCenter);
        // Sub-pixel meshlets: projected bounding-sphere radius below the threshold.
        if (mainVisible && !ground && camera.cull.x > 0.0
            && radius * camera.cull.x < camera.cull.y * max(eyeDistance - radius, 0.0))
            mainVisible = false;
        // Back-facing meshlets of single-sided materials (their fragments would all be
        // discarded by the !gl_FrontFacing test). The cone is stored as an outward axis in
        // mesh space plus the sine of its half-angle; only built when winding and vertex
        // normals agree for every triangle.
        bool singleSided = material.parameters.z > 0.5 || material.extraTextureIndices.x == 0u;
        if (mainVisible && !skinned && singleSided && camera.cullFlags.x > 0.0 && (meshlet.coneFlags & 1u) != 0u) {
            vec4 cone = unpackSnorm4x8(meshlet.cone);
            vec3 axis = mat3(instance.transform) * cone.xyz;
            axis.x = -axis.x;
            float axisLength = length(axis);
            if (axisLength > 0.0) {
                axis /= axisLength;
                if (dot(toCenter, axis) >= cone.w * eyeDistance + radius)
                    mainVisible = false;
            }
        }
        if (shadowVisible) {
            for (uint plane = 0; plane < 6; ++plane) {
                if (dot(camera.lightPlanes[plane].xyz, center) + camera.lightPlanes[plane].w < -radius)
                    shadowVisible = false;
            }
            // cull.w is half a texel of the finest cascade. Cascade texels grow roughly linearly
            // with view distance, so the smallest caster that can still matter grows with it.
            // The size test uses the whole object: dropping the small meshlets of a large model
            // ate holes into its shadow that grew with distance (a fake "shadow fade").
            float casterMinimum = camera.cull.w * (camera.cullFlags.z > 0.0
                ? max(1.0, distanceToCamera / camera.cullFlags.z) : 1.0);
            float casterRadius = skinned ? radius : max(radius, instance.bounds.w);
            if (distanceToCamera - radius > camera.cull.z || casterRadius < casterMinimum)
                shadowVisible = false;
        }
    }

    uint capacity = camera.counts.y;
    if (mainVisible) {
        if (decal) {
            uint destination = atomicAdd(command[3], 1);
            if (destination < capacity)
                visible[capacity - 1u - destination] = source;
        } else {
            uint destination = atomicAdd(command[0], 1);
            if (destination < capacity)
                visible[destination] = source;
        }
    }
    if (shadowVisible) {
        uint destination = atomicAdd(command[6], 1);
        if (destination < capacity)
            visible[capacity + destination] = source;
    }
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
layout(constant_id = 2) const uint visibleList = 0; // 0: opaque list (front), 1: decal list (back)

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
    uvec2 candidate = visible[visibleList == 1u
        ? camera.counts.y - 1u - gl_WorkGroupID.x : gl_WorkGroupID.x];
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
    vec4 lightPlanes[6];
    vec4 cull;
    vec4 cullFlags; // y: profiling switch, skip the alpha-cutout discard
    mat4 cascadeViewProjection[4];
    vec4 cascadeTexel;  // world units per shadow texel, per cascade
    vec4 shadowParams;  // x: cascade count, y: shadow depth per world unit, z: PCF softness (world units), w: texels per tile
    vec4 cascadeSplits; // far view depth of each cascade slice
    vec4 cameraForward; // xyz: view direction, w: 1 when cascades are split by view depth
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

const vec2 shadowPoisson[12] = vec2[](
    vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457), vec2(-0.203, 0.621),
    vec2(0.962, -0.195), vec2(0.473, -0.480), vec2(0.519, 0.767), vec2(0.185, -0.893),
    vec2(0.507, 0.064), vec2(0.896, 0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));

// Cascaded shadow lookup. The cascade is chosen by view depth (cascadeSplits) rather than by
// "first light-space tile that holds the point": the tile edges are squares in light space,
// which projected onto the ground gave the diagonal seams. The last cascadeBlend of every slice
// samples both cascades and cross-fades them smoothly (no dither, no fade-to-lit in between),
// and only the final slice fades to lit at the far end of the shadow range. Each sample is a
// rotated 12-tap Poisson PCF on top of the hardware 2x2 compare filter.
const float cascadeBlend = 0.2;

float cascadeSample(uint cascade, vec3 position, vec3 geometricNormal, float lightCosine,
    float noise, out bool inside)
{
    float tileTexels = max(camera.shadowParams.w, 1.0);
    float texel = camera.cascadeTexel[cascade];
    float kernelTexels = clamp(camera.shadowParams.z / texel, 1.0, 3.0);
    // Normal offset grows at grazing light angles, where depth acne is worst.
    vec3 offsetPosition = position
        + geometricNormal * texel * (0.75 + 1.5 * (1.0 - lightCosine));
    vec3 ndc = (camera.cascadeViewProjection[cascade] * vec4(offsetPosition, 1.0)).xyz;
    float limit = 1.0 - 2.0 * (kernelTexels + 2.0) / tileTexels;
    inside = max(abs(ndc.x), abs(ndc.y)) < limit && ndc.z > 0.0 && ndc.z < 1.0;
    if (!inside)
        return 1.0;
    // Receiver slope: across the PCF footprint a surface tilted against the light changes light
    // depth by about kernel * texel * tan(angle); without this the outer taps self-shadow.
    float tangent = min(sqrt(max(1.0 - lightCosine * lightCosine, 0.0)) / max(lightCosine, 0.2), 4.0);
    float biasTexels = 0.5 + 0.35 * kernelTexels * tangent;
    vec2 uv = (vec2(float(cascade & 1u), float(cascade >> 1u)) + ndc.xy * 0.5 + 0.5) * 0.5;
    float reference = ndc.z - camera.shadowParams.y * texel * biasTexels;
    float atlasStep = kernelTexels * 0.5 / tileTexels;
    float angle = noise * 6.28318530718;
    mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    float lit = 0.0;
    for (int tap = 0; tap < 12; ++tap)
        lit += texture(shadowMap, vec3(uv + rotation * shadowPoisson[tap] * atlasStep, reference));
    return lit / 12.0;
}

float cascadedShadow(vec3 position, vec3 geometricNormal, vec3 lightDirection)
{
    uint count = uint(camera.shadowParams.x);
    float noise = interleavedGradientNoise(gl_FragCoord.xy);
    float lightCosine = clamp(dot(geometricNormal, lightDirection), 0.0, 1.0);
    bool inside = false;
    if (camera.cameraForward.w < 0.5 || count == 0u) // one map fitted to a small scene
        return cascadeSample(0u, position, geometricNormal, lightCosine, noise, inside);
    float depth = dot(position - camera.cameraPosition.xyz, camera.cameraForward.xyz);
    uint cascade = 0u;
    while (cascade + 1u < count && depth > camera.cascadeSplits[cascade])
        ++cascade;
    float shadow = cascadeSample(cascade, position, geometricNormal, lightCosine, noise, inside);
    // Safety net (normal offset pushed the point over a tile edge): use a coarser cascade.
    while (!inside && cascade + 1u < count) {
        ++cascade;
        shadow = cascadeSample(cascade, position, geometricNormal, lightCosine, noise, inside);
    }
    if (!inside)
        return 1.0;
    float sliceFar = camera.cascadeSplits[cascade];
    float sliceNear = cascade == 0u ? 0.0 : camera.cascadeSplits[cascade - 1u];
    float blendStart = sliceFar - cascadeBlend * (sliceFar - sliceNear);
    if (depth > blendStart) {
        float t = smoothstep(blendStart, sliceFar, depth);
        if (cascade + 1u < count) {
            bool nextInside = false;
            float next = cascadeSample(cascade + 1u, position, geometricNormal, lightCosine, noise, nextInside);
            if (nextInside)
                shadow = mix(shadow, next, t);
        } else {
            shadow = mix(shadow, 1.0, t); // end of the shadow range
        }
    }
    return shadow;
}

void main()
{
    Instance instance = instances[fragmentInstance];
    Material material = materials[fragmentMaterial];
    if (!gl_FrontFacing && (material.parameters.z > 0.5 || material.extraTextureIndices.x == 0u))
        discard;
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
    if (!gl_FrontFacing) normal = -normal;
    vec3 geometricNormal = normal;
    if ((camera.cameraPosition.w >= 0.0 || material.extraTextureIndices.y != 0u)
        && material.parameters.z < 0.5 && material.textureIndices.y != 0u) {
        vec3 dp1 = dFdx(fragmentWorldPosition), dp2 = dFdy(fragmentWorldPosition);
        vec2 uv1 = dFdx(fragmentTexCoord), uv2 = dFdy(fragmentTexCoord);
        float determinant = uv1.x * uv2.y - uv1.y * uv2.x;
        if (abs(determinant) > 0.0000001) {
            vec3 tangent = (dp1 * uv2.y - dp2 * uv1.y) / determinant;
            vec3 bitangent = (dp2 * uv1.x - dp1 * uv2.x) / determinant;
            if (dot(tangent,tangent) > 0.0000001 && dot(bitangent,bitangent) > 0.0000001) {
                vec2 xy = texture(diffuseTextures[nonuniformEXT(material.textureIndices.y)], fragmentTexCoord).rg * 2.0 - 1.0;
                normal = normalize(normalize(tangent) * xy.x + normalize(bitangent) * xy.y
                    + normal * sqrt(max(0.0, 1.0 - dot(xy,xy))));
            }
        }
    }
    vec3 lightDirection = normalize(vec3(0.35, 0.8, 0.25));
    bool terrainLodMaterial = material.parameters.z > 1.5 && material.parameters.z < 2.5;
    float shadow = terrainLodMaterial
        ? 1.0 : cascadedShadow(fragmentWorldPosition, geometricNormal, lightDirection);
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
    if (material.parameters.y > 0.0 && opacity < material.parameters.y && camera.cullFlags.y == 0.0)
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
    vec4 lightPlanes[6];
    vec4 cull;
    vec4 cullFlags;
    mat4 cascadeViewProjection[4];
    vec4 cascadeTexel;
    vec4 shadowParams;
} camera;
layout(std430, binding = 10) readonly buffer SkinMatrices { mat4 skinMatrices[]; };
layout(push_constant) uniform ShadowCascade { uint index; } shadowCascade;

layout(location = 0) out vec2 shadowTexCoord[];
layout(location = 1) out flat uint shadowMaterial[];
layout(location = 2) out float shadowVertexAlpha[];

void main()
{
    uvec2 candidate = visible[camera.counts.y + gl_WorkGroupID.x];
    Instance instance = instances[candidate.x];
    Meshlet meshlet = meshlets[candidate.y];
    uint cascade = min(shadowCascade.index, 3u);
    mat4 lightMatrix = camera.cascadeViewProjection[cascade];
    bool rejected = instance.meshletRange.w != 0;
    // The shared caster list covers every cascade; drop meshlets outside this cascade's tile
    // (and sub-texel ones in the coarser cascades) before emitting any geometry.
    if (!rejected && instance.skinRange.y == 0) {
        vec3 center = (instance.transform * vec4(meshlet.sphere.xyz, 1.0)).xyz;
        center.x = -center.x;
        center -= camera.worldOrigin.xyz;
        float scale = max(length(instance.transform[0].xyz),
            max(length(instance.transform[1].xyz), length(instance.transform[2].xyz)));
        float radius = meshlet.sphere.w * scale;
        float texel = camera.cascadeTexel[cascade];
        vec2 ndc = (lightMatrix * vec4(center, 1.0)).xy;
        float ndcRadius = radius * 2.0 / max(texel * camera.shadowParams.w, 0.0001);
        if (any(greaterThan(abs(ndc), vec2(1.0 + ndcRadius)))
            || (cascade > 0u && max(radius, instance.bounds.w) < 0.25 * texel))
            rejected = true;
    }
    if (rejected) {
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
        gl_MeshVerticesEXT[vertexIndex].gl_Position = lightMatrix * world;
        shadowTexCoord[vertexIndex] = vertex.texCoord;
        shadowMaterial[vertexIndex] = meshlet.materialIndex;
        shadowVertexAlpha[vertexIndex] = unpackUnorm4x8(vertex.color).a;
    }
    for (uint primitiveIndex = gl_LocalInvocationIndex;
         primitiveIndex < meshlet.triangleCount; primitiveIndex += gl_WorkGroupSize.x) {
        uint first = meshlet.indexOffset + primitiveIndex * 3;
        gl_PrimitiveTriangleIndicesEXT[primitiveIndex] =
            uvec3(indices[first], indices[first + 1], indices[first + 2]);
    }
}
)";

        // Alpha-tested casters: cutout foliage, bare branches and fences used to cast the
        // shadow of their whole card. Same test as the scene fragment shader.
        const char* shadowFragmentShader = R"(
#version 460
#extension GL_EXT_nonuniform_qualifier : require

struct Material { vec4 baseColor; uvec4 textureIndices; vec4 parameters; uvec4 extraTextureIndices; };
layout(std430, binding = 9) readonly buffer Materials { Material materials[]; };
layout(std140, binding = 7) uniform Camera {
    mat4 viewProjection;
    mat4 lightViewProjection;
    vec4 planes[6];
    vec4 cameraPosition;
    vec4 worldOrigin;
    uvec4 counts;
    vec4 lightPlanes[6];
    vec4 cull;
    vec4 cullFlags; // y: profiling switch, skip the alpha-cutout discard
} camera;
layout(binding = 11) uniform sampler2D diffuseTextures[];

layout(location = 0) in vec2 shadowTexCoord;
layout(location = 1) in flat uint shadowMaterial;
layout(location = 2) in float shadowVertexAlpha;

void main()
{
    Material material = materials[shadowMaterial];
    if (material.parameters.y > 0.0 && material.parameters.z < 0.5 && camera.cullFlags.y == 0.0) {
        float opacity = texture(diffuseTextures[nonuniformEXT(material.textureIndices.x)], shadowTexCoord).a
            * material.baseColor.a * shadowVertexAlpha;
        if (opacity < material.parameters.y)
            discard;
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

        // One sparse page (the image's sparse block size, normally 64 KiB) inside a pooled block.
        struct SparsePage
        {
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkDeviceSize offset = 0;
        };

        // Sparse textures draw their pages from large pooled blocks instead of one
        // vkAllocateMemory per texture; the next block is allocated on a worker thread
        // before the pool runs dry, so binding pages never allocates on the frame.
        struct SparsePagePool
        {
            std::uint32_t memoryType = 0;
            VkDeviceSize pageSize = 0;
            VkDeviceSize blockBytes = 0;
            std::vector<VkDeviceMemory> blocks;
            std::vector<SparsePage> freePages;
            std::future<VkDeviceMemory> pendingBlock;
        };

        // Virtual texture: the image is created with its full mip chain, but only mips
        // [firstResidentMip, mipLevels) are backed, uploaded and visible through the view.
        // Streaming lowers firstResidentMip one level at a time until it reaches targetMip.
        struct TextureTarget
        {
            VkImage image = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkFormat format = VK_FORMAT_UNDEFINED;
            VkExtent2D extent{1, 1};
            SparsePagePool* pool = nullptr;
            std::vector<SparsePage> pages;
            // Binds not yet submitted with vkQueueBindSparse (cleared once submitted).
            std::vector<VkSparseImageMemoryBind> imageBinds;
            std::vector<VkSparseMemoryBind> opaqueBinds;
            VkExtent3D granularity{};
            VkDeviceSize virtualBytes = 0;
            VkDeviceSize residentBytes = 0;
            std::uint32_t mipLevels = 1;
            std::uint32_t firstResidentMip = 0;
            std::uint32_t targetMip = 0;
            std::uint64_t bindTicket = 0;           // binder-thread ticket carrying this texture's latest binds
            VkImageView heldView = VK_NULL_HANDLE;  // view kept off the descriptor until the bind is submitted
            VkImageView nextView = VK_NULL_HANDLE;  // binder-created view for the pending mip promotion
            bool mipBindPending = false;            // next finer mip is bound (or binding), not yet uploaded
            bool sparse = false;
            // Residency (texture budget). pageMips[i] is the mip of pages[i] (0xFF: tail/metadata).
            std::vector<std::uint8_t> pageMips;
            VkDeviceSize opaqueBytes = 0;           // mip tail + metadata pages, resident while alive
            std::uint32_t tailFirstLod = 0;
            std::uint32_t floorMip = 0;             // coarsest streamed state; never unloaded past it
            std::uint32_t desiredMip = 0;           // residency pass: finest mip worth having
            std::uint32_t evictMip = 0;             // pending unload: the view's next base mip
            float distance = std::numeric_limits<float>::infinity();
            bool evictPending = false;
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
            std::array<glm::vec4, 6> lightPlanes{};
            glm::vec4 cull{};
            glm::vec4 cullFlags{};
            std::array<glm::mat4, maxShadowCascades> cascadeViewProjection{
                glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)};
            glm::vec4 cascadeTexel{1.0f};
            glm::vec4 shadowParams{};
            glm::vec4 cascadeSplits{0.0f};  // far view depth of each cascade slice
            glm::vec4 cameraForward{0.0f};  // w = 1: cascades selected by view depth
        };

        static_assert(sizeof(GpuVertex) == 64);
        static_assert(sizeof(GpuMeshlet) == 48);
        static_assert(sizeof(GpuInstance) == 128);
        static_assert(sizeof(GpuMaterial) == 64);
        static_assert(sizeof(CameraData) == 720);

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
                    material.decal ? 1.0f : 0.0f), glm::uvec4(material.doubleSided ? 1u : 0u,
                        material.normalMapped ? 1u : 0u, 0u, 0u)};
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
            Buffer staging;                             // persistent upload staging, grows only
            std::vector<std::uint32_t> dirtyTextures;   // bindless slots to rewrite in this set
            bool submitted = false;
            std::uint64_t submitTicket = 0;  // submit-thread ticket of the last frame recorded here
            std::uint64_t descriptorGeneration = 0;
        };

        SDL_Window* window = nullptr;
        VkInstance instance = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT validationMessenger = VK_NULL_HANDLE;
        PFN_vkDestroyDebugUtilsMessengerEXT destroyValidationMessenger = nullptr;
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
        VkPipeline selectionDecalPipeline = VK_NULL_HANDLE;
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
        std::list<SparsePagePool> sparsePools;
        std::map<VkFormat, bool> sparseFormatSupport;
        // Sparse image pool. On its own a sparse vkCreateImage costs ~0.1 ms, but mid-flight it
        // regularly sat in a ~25-35 ms driver-lock stall together with the submit thread, the
        // binder and every other worker (fly hitches). Images for the common texture shapes are
        // created up front at startup and handed out by exact shape; a shape running low is
        // refilled in one serial batch on the refill thread (a shape missed twice gets a small
        // bucket of its own), and images of destroyed textures are unbound and recycled.
        // OPENF4_SPARSE_IMAGE_POOL=0 disables the pool, =N scales the startup counts (default 1).
        struct SparseImageKey
        {
            VkFormat format = VK_FORMAT_UNDEFINED;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint32_t mipLevels = 0;
            auto operator<=>(const SparseImageKey&) const = default;
        };
        struct SparseImageShape
        {
            bool valid = false;  // false: this shape cannot be a sparse texture (use VMA)
            VkMemoryRequirements memoryRequirements{};
            VkExtent3D granularity{};
            std::uint32_t mipTailFirstLod = 0;             // raw imageMipTailFirstLod
            std::vector<VkSparseMemoryBind> opaqueRanges;  // mip tail + metadata, memory unset
        };
        struct SparseImageBucket
        {
            std::optional<SparseImageShape> shape;
            std::vector<VkImage> free;
            std::uint32_t refillTarget = 0;  // free images a refill restores; 0 = cold shape
            std::uint32_t misses = 0;
            bool refilling = false;
        };
        bool sparseImagePoolEnabled = false;
        std::mutex sparseImagePoolMutex;
        std::map<SparseImageKey, SparseImageBucket> sparseImagePool;  // guarded by sparseImagePoolMutex
        std::deque<SparseImageKey> sparseImageRefills;                // guarded by sparseImagePoolMutex
        bool sparseImageRefillStop = false;                           // guarded by sparseImagePoolMutex
        std::condition_variable sparseImageRefillWake;
        std::thread sparseImageRefiller;
        std::atomic<std::uint32_t> sparseImagePoolHits{0};
        std::atomic<std::uint32_t> sparseImagePoolMisses{0};
        std::atomic<std::uint32_t> sparseImagePoolRefilled{0};
        std::atomic<std::uint32_t> sparseImagePoolRecycled{0};
        // vkQueueBindSparse blocks for as long as a worker's vkCreateImage holds the driver's
        // allocation lock (~25 ms), so sparse binds are submitted by a binder thread on a second
        // queue of the graphics family. The frame thread only hands over bind lists; a texture or
        // mip is uploaded once its bind has been submitted, and the graphics submit waits on that
        // timeline value (already signalled by a submitted bind, so never wait-before-signal).
        // OPENF4_SPARSE_BIND_SYNC=1 keeps the old frame-thread binds.
        struct SparseImageBinds
        {
            VkImage image = VK_NULL_HANDLE;
            std::vector<VkSparseImageMemoryBind> tiles;
            std::vector<VkSparseMemoryBind> opaque;
        };
        // Device calls on the frame thread (vkCreateImageView too) stall behind the same lock, so
        // the binder also creates the view each mip promotion will switch to.
        struct TextureViewRequest
        {
            std::uint32_t texture = 0;
            VkImage image = VK_NULL_HANDLE;
            VkFormat format = VK_FORMAT_UNDEFINED;
            std::uint32_t baseMip = 0;
            std::uint32_t mipLevels = 1;
        };
        struct SparseBindJob
        {
            std::uint64_t ticket = 0;
            std::vector<SparseImageBinds> images;
            std::vector<TextureViewRequest> views;
        };
        std::vector<std::pair<std::uint32_t, VkImageView>> binderViews;  // guarded by sparseBindMutex
        VkQueue sparseQueue = VK_NULL_HANDLE;  // null: binds go out on the frame queue
        VkSemaphore sparseTimeline = VK_NULL_HANDLE;
        bool sparseQueueAvailable = false;
        std::thread sparseBinder;
        std::mutex sparseBindMutex;
        std::condition_variable sparseBindWake;
        std::condition_variable sparseBindDone;
        std::deque<SparseBindJob> sparseBindJobs;
        bool sparseBinderStop = false;
        std::uint64_t sparseTicketIssued = 0;  // frame thread only
        std::atomic<std::uint64_t> sparseTicketSubmitted{0};
        std::atomic<VkResult> sparseBindResult{VK_SUCCESS};
        std::uint64_t frameSparseWait = 0;     // timeline value this frame's submit waits on
        std::vector<std::uint32_t> boundPromotions;  // textures whose next mip is bound, awaiting upload
        struct StreamingTexture
        {
            std::uint32_t index = 0;
            SceneTexture source;  // CPU copy of the full chain, released once fully resident
        };
        std::vector<StreamingTexture> streamingTextures;
        std::size_t streamingCursor = 0;
        // Result of the thread-safe half of texture creation (see planTextureImage).
        struct TexturePlan
        {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkMemoryRequirements memoryRequirements{};
            bool sparse = false;
            VkExtent3D granularity{};
            std::uint32_t tailFirstLod = 0;
            std::uint32_t targetMip = 0;
            std::uint32_t firstResidentMip = 0;
            std::vector<VkSparseMemoryBind> opaqueRanges;  // mip tail + metadata; memory bound later
            VkImageView view = VK_NULL_HANDLE;  // created with the image, off the frame thread
            SparseImageKey poolKey;             // sparse: the image pool shape it returns to
        };
        struct CreatingTexture
        {
            std::uint32_t index = 0;
            SceneTexture texture;
            std::future<TexturePlan> plan;
            std::optional<TexturePlan> ready;  // created, held back by the texture budget
        };
        std::vector<CreatingTexture> creatingTextures;
        struct RetiredView
        {
            VkImageView view = VK_NULL_HANDLE;
            std::uint64_t serial = 0;
        };
        std::vector<RetiredView> retiredViews;
        std::uint64_t frameSerial = 0;
        // GPU submit thread. vkQueueSubmit2 (and present) wait behind the driver allocation lock
        // a worker's vkCreateImage holds (~25 ms), so the frame thread only records and hands the
        // submit + present to one FIFO thread. Tickets count handed-off frames. Before a frame slot
        // is reused the frame thread waits for that slot's last ticket (two frames back): its fence
        // is then really submitted, and with one frame in flight on the thread at most one acquired
        // image is ever unpresented. queueMutex serialises every use of `queue` across threads;
        // swapchainMutex covers vkAcquireNextImageKHR vs vkQueuePresentKHR. UI frames (editor)
        // submit inline because the UI backend may use the queue while recording.
        // OPENF4_SUBMIT_THREAD=0 submits inline on the frame thread.
        struct SubmitJob
        {
            std::uint64_t ticket = 0;
            std::array<VkSemaphoreSubmitInfo, 3> waits{};
            std::uint32_t waitCount = 0;
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            VkSemaphore presentSemaphore = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
            VkSwapchainKHR swapchain = VK_NULL_HANDLE;
            std::uint32_t imageIndex = 0;
        };
        std::mutex queueMutex;
        std::mutex swapchainMutex;
        bool submitThreadEnabled = false;
        std::uint64_t submitTicketIssued = 0;           // frame thread only
        std::mutex submitMutex;
        std::condition_variable submitDone;
        std::uint64_t submitTicketDone = 0;             // guarded by submitMutex
        VkResult submitError = VK_SUCCESS;              // guarded by submitMutex
        const char* submitErrorWhat = "";               // guarded by submitMutex
        std::atomic<bool> swapchainStale{false};        // present reported out of date / suboptimal
        std::atomic<std::uint64_t> submitThreadMaxUs{0};  // slowest submit+present since last frame
#if OPENF4_SUBMIT_STDEXEC
        std::optional<exec::single_thread_context> submitContext;
        std::optional<stdexec::counting_scope> submitScope;
#else
        std::thread submitThread;
        std::condition_variable submitWake;
        std::deque<SubmitJob> submitJobs;               // guarded by submitMutex
        bool submitStop = false;                        // guarded by submitMutex
#endif
        // Scene buffers are re-allocated (2x) on a worker thread once they pass 75% full,
        // so growing them no longer stalls the frame on an allocation and a synchronous copy.
        struct PendingGrowth
        {
            Buffer* target = nullptr;
            std::future<Buffer> buffer;
            VkDeviceSize capacity = 0;  // counts against the memory budget while in flight
        };
        std::vector<PendingGrowth> pendingGrowths;
        // Unified GPU memory budget (OPENF4_MEMORY_PRESET / --memory-preset = ultralow 1 GiB | low
        // 2 GiB | medium 4 GiB (default) | veryhigh = a third of the largest device-local heap, at most
        // 12 GiB; OPENF4_MEMORY_BUDGET_MB overrides, 0 = uncapped). The budget is the TOTAL for scene
        // buffers (every scene buffer, both frames' cull lists and in-flight growths) plus resident
        // texture memory; the two never add up past it.
        // - The scene reserves OPENF4_SCENE_SHARE percent of it up front (default 50, 10..90;
        //   OPENF4_SCENE_RESERVE_MB sets MiB instead, 0 = grow on demand) so streaming never grows a
        //   scene buffer mid-flight in the common case.
        // - Textures get whatever the scene has not allocated (minus a growth headroom of 1/16 while a
        //   scene buffer is nearly full); the residency pass raises the distance mip bias and unloads
        //   finer mips to fit as that moves. OPENF4_TEXTURE_BUDGET_MB caps textures further.
        // - The scene may grow past its reservation into texture memory down to a texture floor of 1/4
        //   (geometry wins: textures unload mips to make room); appends past that are refused.
        static constexpr VkDeviceSize sceneReserveUnits = 4096;  // shares below sum to this
        VkDeviceSize sceneReserveBytes = 0;
        VkDeviceSize maxStorageRange = 0xFFFFFFFFull;
        VkDeviceSize memoryBudgetBytes = VkDeviceSize(4096) << 20;   // total, 0 = uncapped
        VkDeviceSize textureFloorBytes = VkDeviceSize(1024) << 20;   // textures always keep this much
        VkDeviceSize sceneHeadroomBytes = VkDeviceSize(256) << 20;   // kept free for scene growth
        VkDeviceSize sceneDemandBytes = 0;    // scene growth refused for texture-held room: textures yield it
        VkDeviceSize textureBudgetCapBytes = 0;                      // OPENF4_TEXTURE_BUDGET_MB (0 = none)
        VkDeviceSize sceneCapBytes = 0;       // most the scene may take: budget - texture floor (0 = uncapped)
        VkDeviceSize textureBudgetBytes = VkDeviceSize(2048) << 20;  // live, see refreshMemoryBudget()
        const char* memoryPresetName = "medium";
        // Texture residency: per-material texture ids, per-instance unique material sets, and the
        // distance-based mip choice. OPENF4_TEXTURE_MIP_DISTANCE (world units, default 1536) is
        // where a texture first drops one level below its cap; every doubling drops another.
        // OPENF4_TEXTURE_MIP_BIAS adds a constant bias. The budget adds its own bias on top.
        std::vector<std::array<std::uint32_t, 8>> materialTextureIds;
        std::vector<std::vector<std::uint32_t>> materialSets;          // unique materials per meshlet range
        std::unordered_map<std::uint64_t, std::uint32_t> materialSetByRange;
        std::vector<std::uint32_t> instanceMaterialSet;                // per instance -> materialSets
        std::vector<std::uint32_t> materialDistanceBits;               // scratch (float bits, atomic_ref)
        std::vector<std::uint32_t> textureDistanceBits;                // scratch (float bits, atomic_ref)
        struct RetiredPage
        {
            SparsePagePool* pool = nullptr;
            SparsePage page;
            std::uint64_t serial = 0;
        };
        std::vector<RetiredPage> retiredPages;                         // unloaded mips, freed after in-flight frames
        VkDeviceSize retiredPageBytes = 0;
        std::vector<std::uint32_t> evictingTextures;                   // waiting for their coarser view
        std::vector<std::size_t> promotionOrder;                       // streamingTextures positions, nearest first
        float residencyBias = 0.0f;
        float textureMipDistance = 1536.0f;
        float textureMipUserBias = 0.0f;
        glm::vec3 streamCamera{};                                      // camera in instance-bounds space
        bool streamCameraValid = false;
        // The pass runs every residencyInterval frames, after the camera moved residencyMoveDistance,
        // or when the streaming set changed (promotionOrder holds positions into streamingTextures).
        static constexpr std::uint64_t residencyInterval = 8;
        static constexpr float residencyMoveDistance = 128.0f;
        std::uint64_t residencySerial = 0;
        glm::vec3 residencyCamera{std::numeric_limits<float>::max()};
        bool residencyDirty = true;
        // Retired buffers and image views are released in per-frame batches on a worker: freeing a
        // retired scene buffer costs 10-30 ms, and any destroy on the frame thread can stall behind a
        // worker's vkCreateImage holding the driver's allocation lock.
        std::vector<std::future<void>> pendingFrees;
        static constexpr VkBufferUsageFlags sceneBufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        struct TextureStreamTuning
        {
            bool enabled = true;                       // OPENF4_TEXTURE_STREAMING=0 uploads full chains
            std::uint32_t startDimension = 128;        // OPENF4_TEXTURE_STREAM_START: first visible mip size
            VkDeviceSize frameBudget = 16ull << 20;    // OPENF4_TEXTURE_STREAM_MB: promoted bytes per frame
            std::uint32_t maxPromotions = 256;         // mip promotions per frame
        } textureStreamTuning = readTextureStreamTuning();

        static TextureStreamTuning readTextureStreamTuning()
        {
            TextureStreamTuning tuning;
            if (const char* value = std::getenv("OPENF4_TEXTURE_STREAMING"); value && *value == '0')
                tuning.enabled = false;
            if (const char* value = std::getenv("OPENF4_TEXTURE_STREAM_START"); value && *value)
            {
                const long parsed = std::strtol(value, nullptr, 10);
                if (parsed >= 16 && parsed <= 4096)
                    tuning.startDimension = static_cast<std::uint32_t>(parsed);
            }
            if (const char* value = std::getenv("OPENF4_TEXTURE_STREAM_MB"); value && *value)
            {
                const double parsed = std::strtod(value, nullptr);
                if (std::isfinite(parsed) && parsed >= 0.25 && parsed <= 256.0)
                    tuning.frameBudget = static_cast<VkDeviceSize>(parsed * 1048576.0);
            }
            return tuning;
        }
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
        std::map<std::uint32_t, GpuMaterial> pendingMaterialUpdates;
        struct PendingTexture
        {
            std::uint32_t index = 0;
            SceneTexture texture;
        };
        std::vector<PendingTexture> pendingTextures;
        std::vector<PendingTexture> bindingTextures;  // waiting for the binder thread to submit their binds
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
        double textureCreateAccumMs = 0.0;
        double sparsePoolStallAccumMs = 0.0;
        // OPENF4_STREAM_PROFILE logs worker / submit-thread stalls with their start time (@ms since
        // renderer creation) so overlapping driver-lock stalls can be told apart from their cause.
        const std::chrono::steady_clock::time_point traceEpoch = std::chrono::steady_clock::now();
        double traceAt(std::chrono::steady_clock::time_point at) const
        {
            return std::chrono::duration<double, std::milli>(at - traceEpoch).count();
        }
        const bool profileUploads = [] {
            const char* value = std::getenv("OPENF4_STREAM_PROFILE");
            return value && *value && *value != '0';
        }();
        // GPU culling knobs; defaults are the shipped behaviour, env vars exist for A/B profiling.
        struct CullTuning
        {
            float shadowDistance = 12288.0f;  // shadows reach 1.5x this (4.5 cells: past the 4-cell object band)
            float minRadiusPixels = 0.5f;     // meshlets whose projected radius is smaller are dropped (0 = off)
            bool coneCulling = true;          // back-facing meshlet cones of single-sided materials
            bool debugNoAlphaTest = false;    // profiling only: measures what the alpha-cutout discard costs
            std::uint32_t shadowCascades = 4; // cascades (atlas tiles) for large worlds, 1..4
            std::uint32_t shadowTile = 2048;  // texels per cascade tile (atlas is 2x2 tiles)
            float shadowSoftness = 3.0f;      // PCF radius in world units, clamped to 1..3 texels
        } cullTuning = readCullTuning();

        static CullTuning readCullTuning()
        {
            CullTuning tuning;
            const auto number = [](const char* name, float fallback) {
                const char* value = std::getenv(name);
                if (!value || !*value) return fallback;
                char* end = nullptr;
                const float parsed = std::strtof(value, &end);
                return end != value && std::isfinite(parsed) && parsed >= 0.0f ? parsed : fallback;
            };
            tuning.shadowDistance = std::max(number("OPENF4_SHADOW_DISTANCE", tuning.shadowDistance), 256.0f);
            tuning.minRadiusPixels = number("OPENF4_MESHLET_MIN_PIXELS", tuning.minRadiusPixels);
            if (const char* cone = std::getenv("OPENF4_CONE_CULL"); cone && *cone == '0')
                tuning.coneCulling = false;
            if (const char* alpha = std::getenv("OPENF4_DEBUG_NO_ALPHA_TEST"); alpha && *alpha == '1')
                tuning.debugNoAlphaTest = true;
            tuning.shadowCascades = static_cast<std::uint32_t>(std::clamp(
                number("OPENF4_SHADOW_CASCADES", 4.0f), 1.0f, static_cast<float>(maxShadowCascades)));
            tuning.shadowTile = static_cast<std::uint32_t>(std::clamp(
                number("OPENF4_SHADOW_RESOLUTION", 2048.0f), 512.0f, 4096.0f)) & ~255u;
            tuning.shadowSoftness = number("OPENF4_SHADOW_SOFTNESS", tuning.shadowSoftness);
            return tuning;
        }
        struct PreviewCheckpoint
        {
            std::uint32_t vertices, indices, meshlets, instances, candidates, materials, textures;
            glm::vec3 center;
            float radius;
            RendererStats stats;
            std::vector<std::uint32_t> selection;
            std::optional<std::uint32_t> active;
        };
        std::optional<PreviewCheckpoint> preview;

        Impl(SDL_Window* targetWindow, const PackedMesh& mesh,
            std::span<const SceneInstance> sceneInstances,
            std::span<const SceneMaterial> sceneMaterials,
            std::span<const SceneTexture> sceneTextures)
            : window(targetWindow)
        {
            try
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
                warmSparsePool();
                createPipelines();
                updateDescriptorSets();

                VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
                queryInfo.queryCount = frameCount * timestampsPerFrame;
                check(vkCreateQueryPool(device, &queryInfo, nullptr, &queryPool), "vkCreateQueryPool");
            }
            catch (...)
            {
                destroy();
                throw;
            }
        }

        ~Impl() { destroy(); }

        void destroy()
        {
            stopSubmitThread();
            stopSparseBinder();
            stopSparseImageRefiller();
            if (device)
                vkDeviceWaitIdle(device);
            if (sparseTimeline)
                vkDestroySemaphore(device, sparseTimeline, nullptr);
            sparseTimeline = VK_NULL_HANDLE;
            for (const auto& [index, view] : binderViews)  // pre-created views never claimed
                if (view)
                    vkDestroyImageView(device, view, nullptr);
            binderViews.clear();
            if (queryPool)
                vkDestroyQueryPool(device, queryPool, nullptr);
            if (selectionPipeline)
                vkDestroyPipeline(device, selectionPipeline, nullptr);
            if (selectionDecalPipeline)
                vkDestroyPipeline(device, selectionDecalPipeline, nullptr);
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
            for (auto& growth : pendingGrowths)
            {
                try
                {
                    Buffer prepared = growth.buffer.get();
                    destroyBuffer(prepared);
                }
                catch (...)
                {
                }
            }
            pendingGrowths.clear();
            for (auto& done : pendingFrees)
                done.wait();
            pendingFrees.clear();
            for (auto& entry : creatingTextures)
            {
                try
                {
                    TexturePlan plan = entry.ready ? std::move(*entry.ready) : entry.plan.get();
                    destroyPlan(plan);
                }
                catch (...)
                {
                }
            }
            creatingTextures.clear();
            for (auto& frame : frames)
            {
                for (auto& upload : frame.retiredUploads)
                    destroyBuffer(upload);
                frame.retiredUploads.clear();
                destroyBuffer(frame.staging);
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
            flushRetiredViews();
            for (auto& texture : textureTargets)
                destroyTextureTarget(texture, false);
            textureTargets.clear();
            streamingTextures.clear();
            if (profileUploads && sparseImagePoolEnabled)
                std::printf("Sparse image pool: %u hits, %u misses, %u refilled, %u recycled\n",
                    sparseImagePoolHits.load(), sparseImagePoolMisses.load(),
                    sparseImagePoolRefilled.load(), sparseImagePoolRecycled.load());
            for (auto& [key, bucket] : sparseImagePool)
                for (const VkImage image : bucket.free)
                    vkDestroyImage(device, image, nullptr);
            sparseImagePool.clear();
            for (auto& pool : sparsePools)
            {
                if (pool.pendingBlock.valid())
                    if (const VkDeviceMemory memory = pool.pendingBlock.get())
                        pool.blocks.push_back(memory);
                for (const VkDeviceMemory memory : pool.blocks)
                    vkFreeMemory(device, memory, nullptr);
            }
            sparsePools.clear();
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
            if (validationMessenger)
                destroyValidationMessenger(instance, validationMessenger, nullptr);
            if (instance)
                vkDestroyInstance(instance, nullptr);
        }

        void createInstance()
        {
            Uint32 extensionCount = 0;
            const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
            if (!extensions)
                throw std::runtime_error(SDL_GetError());
            std::vector<const char*> enabledExtensions(extensions, extensions + extensionCount);
            constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
            const char* requestedValidation = std::getenv("OPENF4_VULKAN_VALIDATION");
            const bool validation = requestedValidation && std::string_view(requestedValidation) == "1";
            VkDebugUtilsMessengerCreateInfoEXT debugInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            if (validation)
            {
                std::uint32_t layerCount = 0;
                check(vkEnumerateInstanceLayerProperties(&layerCount, nullptr), "vkEnumerateInstanceLayerProperties");
                std::vector<VkLayerProperties> layers(layerCount);
                check(vkEnumerateInstanceLayerProperties(&layerCount, layers.data()), "vkEnumerateInstanceLayerProperties");
                if (std::ranges::none_of(layers, [validationLayer](const auto& layer) {
                    return std::string_view(layer.layerName) == validationLayer;
                })) throw std::runtime_error("Requested Vulkan validation layer is unavailable: VK_LAYER_KHRONOS_validation");
                std::uint32_t debugExtensionCount = 0;
                check(vkEnumerateInstanceExtensionProperties(nullptr, &debugExtensionCount, nullptr), "vkEnumerateInstanceExtensionProperties");
                std::vector<VkExtensionProperties> debugExtensions(debugExtensionCount);
                check(vkEnumerateInstanceExtensionProperties(nullptr, &debugExtensionCount, debugExtensions.data()), "vkEnumerateInstanceExtensionProperties");
                if (std::ranges::none_of(debugExtensions, [](const auto& extension) {
                    return std::string_view(extension.extensionName) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
                })) throw std::runtime_error("Requested Vulkan validation requires VK_EXT_debug_utils");
                if (std::ranges::none_of(enabledExtensions, [](const char* extension) {
                    return std::string_view(extension) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
                })) enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                    | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                    | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                debugInfo.pfnUserCallback = validationMessage;
            }
            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.pApplicationName = "OpenF4";
            application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
            application.pEngineName = "OpenF4";
            application.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
            application.apiVersion = VK_API_VERSION_1_3;
            VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            createInfo.pApplicationInfo = &application;
            createInfo.enabledExtensionCount = static_cast<std::uint32_t>(enabledExtensions.size());
            createInfo.ppEnabledExtensionNames = enabledExtensions.data();
            if (validation)
            {
                createInfo.enabledLayerCount = 1;
                createInfo.ppEnabledLayerNames = &validationLayer;
                createInfo.pNext = &debugInfo;
            }
            check(vkCreateInstance(&createInfo, nullptr, &instance), "vkCreateInstance");
            if (validation)
            {
                const auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
                destroyValidationMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
                if (!createMessenger || !destroyValidationMessenger)
                    throw std::runtime_error("Vulkan validation debug messenger entry points are unavailable");
                check(createMessenger(instance, &debugInfo, nullptr, &validationMessenger), "vkCreateDebugUtilsMessengerEXT");
                std::fputs("Vulkan validation enabled: VK_LAYER_KHRONOS_validation\n", stderr);
                std::fflush(stderr);
            }
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
                        sparseQueueAvailable = families[family].queueCount >= 2
                            && !environmentFlagSet("OPENF4_SPARSE_BIND_SYNC");
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
            maxStorageRange = properties.limits.maxStorageBufferRange;
        }

        void createDevice()
        {
            const bool dedicatedSparseQueue = sparseTexturesSupported && sparseQueueAvailable;
            constexpr std::array<float, 2> priorities{1.0f, 1.0f};
            VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queueInfo.queueFamilyIndex = queueFamily;
            queueInfo.queueCount = dedicatedSparseQueue ? 2u : 1u;
            queueInfo.pQueuePriorities = priorities.data();
            VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
            meshFeatures.meshShader = VK_TRUE;
            VkPhysicalDeviceVulkan12Features features12{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &meshFeatures};
            features12.runtimeDescriptorArray = VK_TRUE;
            features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
            features12.descriptorBindingPartiallyBound = VK_TRUE;
            features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
            features12.timelineSemaphore = VK_TRUE;  // core in Vulkan 1.2; sparse binder sync
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
            if (dedicatedSparseQueue)
                startSparseBinder();
            startSubmitThread();
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

        Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool hostVisible = false,
            bool dedicated = false)
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
            // Worker-thread growth buffers take their own memory so they never hold a VMA
            // block-vector lock that the frame thread needs.
            if (dedicated)
                allocationInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
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
            drainSubmits();  // a frame fence is only meaningful once the submit thread submitted it
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
            {
                std::lock_guard queueLock(queueMutex);
                check(vkQueueSubmit2(queue, 1, &submitInfo, fence), "vkQueueSubmit2 stream");
            }
            check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "stream upload fence");
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, commandPool, 1, &command);
        }

        // Grows a shared scene buffer without stalling: the replacement normally comes from a
        // worker-thread allocation started earlier (prefetchSceneBuffer), the old contents are
        // copied on the GPU inside this frame's command buffer, and the old buffer is retired
        // with this frame (the other in-flight frame keeps reading it until its set is rewritten).
        bool growSceneBuffer(Frame& frame, VkCommandBuffer command, Buffer& buffer,
            VkDeviceSize required, bool& readBarrierRecorded)
        {
            if (buffer.size >= required)
                return false;
            Buffer replacement;
            if (const auto found = std::find_if(pendingGrowths.begin(), pendingGrowths.end(),
                    [&](const PendingGrowth& growth) { return growth.target == &buffer; });
                found != pendingGrowths.end())
            {
                Buffer prepared;
                try
                {
                    prepared = found->buffer.get();
                }
                catch (...)
                {
                }
                pendingGrowths.erase(found);
                if (prepared.handle && prepared.size >= required)
                    replacement = prepared;
                else
                    destroyBuffer(prepared);
            }
            const bool prefetched = replacement.handle != VK_NULL_HANDLE;
            if (!replacement.handle)
                replacement = createBuffer(grownSceneCapacity(buffer.size, required), sceneBufferUsage);
            if (profileUploads)
                std::printf("    renderer: scene buffer grew %.1f -> %.1f MiB (%s)\n",
                    static_cast<double>(buffer.size) / 1048576.0,
                    static_cast<double>(replacement.size) / 1048576.0,
                    prefetched ? "prefetched on worker" : "allocated on frame");
            if (buffer.handle)
            {
                if (!readBarrierRecorded)
                {
                    // Earlier submissions may still be writing the old buffer through transfers.
                    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    dependency.memoryBarrierCount = 1;
                    dependency.pMemoryBarriers = &barrier;
                    vkCmdPipelineBarrier2(command, &dependency);
                    readBarrierRecorded = true;
                }
                // Only the live prefix matters: a reserved buffer can be mostly empty.
                const VkBufferCopy copy{0, 0, std::min(buffer.size, required)};
                vkCmdCopyBuffer(command, buffer.handle, replacement.handle, 1, &copy);
                frame.retiredUploads.push_back(buffer);
            }
            buffer = replacement;
            return true;
        }

        // 2x growth, kept inside maxStorageBufferRange (descriptor ranges use the whole buffer).
        VkDeviceSize grownSceneCapacity(VkDeviceSize current, VkDeviceSize required) const
        {
            const VkDeviceSize limit = maxStorageRange & ~VkDeviceSize(255);
            // Under the unified memory budget grow by 1.5x so one buffer does not grab the room
            // textures (or the other scene buffers) could use.
            const VkDeviceSize doubled = std::max<VkDeviceSize>(sceneCapBytes != 0 ? current + current / 2 : current * 2, 256);
            VkDeviceSize capacity = std::max(required, std::min(doubled, limit));
            if (sceneCapBytes != 0)
                capacity = std::max(required, std::min(capacity, current + sceneCapRoom()));
            return capacity;
        }

        void prefetchSceneBuffer(Buffer& buffer, VkDeviceSize used)
        {
            if (!buffer.handle || used * 4 < buffer.size * 3)
                return;
            if (std::any_of(pendingGrowths.begin(), pendingGrowths.end(),
                    [&](const PendingGrowth& growth) { return growth.target == &buffer; }))
                return;
            const VkDeviceSize capacity = grownSceneCapacity(buffer.size, used);
            if (capacity <= buffer.size)
                return;
            pendingGrowths.push_back({&buffer, std::async(std::launch::async, [this, capacity] {
                return timedWorker("scene buffer alloc", [&] {
                    return createBuffer(capacity, sceneBufferUsage, false, true);
                });
            }), capacity});
        }

        // Per-frame staging persists across frames; a fresh host-visible allocation for every
        // upload batch put a vkAllocateMemory on the frame whenever a batch outgrew VMA's blocks.
        Buffer& frameStaging(Frame& frame, VkDeviceSize size)
        {
            if (frame.staging.size < size)
            {
                if (frame.staging.handle)
                    frame.retiredUploads.push_back(frame.staging);
                frame.staging = createBuffer(
                    std::max<VkDeviceSize>({size, frame.staging.size * 2, VkDeviceSize(16) << 20}),
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            }
            return frame.staging;
        }

        // Main-view lists (opaque front, decals back) share [0, C); shadow casters use [C, 2C).
        static VkDeviceSize visibleListBytes(std::uint32_t candidates)
        {
            return static_cast<VkDeviceSize>(candidates) * 2u * sizeof(glm::uvec2);
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

        // FIFO (vsync) by default. OPENF4_PRESENT_MODE=mailbox|immediate lets profiling see
        // real GPU throughput instead of a frame rate quantised to 60/30/20 fps.
        VkPresentModeKHR requestedPresentMode() const
        {
            const char* value = std::getenv("OPENF4_PRESENT_MODE");
            if (!value || !*value)
                return VK_PRESENT_MODE_FIFO_KHR;
            const std::string_view name(value);
            const VkPresentModeKHR wanted = name == "immediate" ? VK_PRESENT_MODE_IMMEDIATE_KHR
                : name == "mailbox" ? VK_PRESENT_MODE_MAILBOX_KHR
                : name == "relaxed" ? VK_PRESENT_MODE_FIFO_RELAXED_KHR : VK_PRESENT_MODE_FIFO_KHR;
            std::uint32_t count = 0;
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &count, nullptr);
            std::vector<VkPresentModeKHR> modes(count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &count, modes.data());
            return std::ranges::find(modes, wanted) != modes.end() ? wanted : VK_PRESENT_MODE_FIFO_KHR;
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
            createInfo.presentMode = requestedPresentMode();
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
            drainSubmits();
            swapchainStale.store(false);
            {
                std::scoped_lock lock(queueMutex, swapchainMutex);
                check(vkDeviceWaitIdle(device), "vkDeviceWaitIdle during resize");
            }
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
            bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT
                | VK_SHADER_STAGE_FRAGMENT_BIT;
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

            // Size the bindless array for the maximum up front: growing it later meant a full
            // GPU wait plus a rewrite of every texture descriptor on the frame that crossed it.
            (void)initialTextureCount;
            replaceDescriptorPool(maxSceneTextures);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
            // Shadow pass: cascade index (selects matrix and atlas tile).
            const VkPushConstantRange cascadeRange{VK_SHADER_STAGE_MESH_BIT_EXT, 0, sizeof(std::uint32_t)};
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &cascadeRange;
            check(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
                "vkCreatePipelineLayout");
        }

        template <class T, std::size_t Extent>
        static VkDeviceSize byteSize(std::span<T, Extent> values)
        {
            return values.size_bytes();
        }

        VkDeviceSize largestDeviceLocalHeap() const
        {
            VkPhysicalDeviceMemoryProperties memory{};
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memory);
            VkDeviceSize largestLocal = 0;
            for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap)
                if (memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    largestLocal = std::max(largestLocal, memory.memoryHeaps[heap].size);
            return largestLocal;
        }

        static VkDeviceSize envMiB(const char* name, long long minimum, long long maximum, VkDeviceSize fallback)
        {
            if (const char* value = std::getenv(name); value && *value)
            {
                const long long parsed = std::strtoll(value, nullptr, 10);
                if (parsed >= minimum && parsed <= maximum)
                    return static_cast<VkDeviceSize>(parsed);
                std::fprintf(stderr, "Ignoring %s=%s (%lld..%lld)\n", name, value, minimum, maximum);
            }
            return fallback;
        }

        // Resolves the memory preset into the unified budget, the up-front scene reservation, the
        // scene's maximum share and the initial texture budget. Returns the reservation in bytes.
        VkDeviceSize resolveSceneReservation()
        {
            const VkDeviceSize largestLocal = largestDeviceLocalHeap();
            VkDeviceSize presetMiB = 4096;
            memoryPresetName = "medium";
            if (const char* value = std::getenv("OPENF4_MEMORY_PRESET"); value && *value)
            {
                std::string key;
                for (const char* c = value; *c; ++c)
                    if (*c != '-' && *c != '_' && *c != ' ')
                        key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*c))));
                if (key == "ultralow") { presetMiB = 1024; memoryPresetName = "ultra low"; }
                else if (key == "low") { presetMiB = 2048; memoryPresetName = "low"; }
                else if (key == "medium") { presetMiB = 4096; memoryPresetName = "medium"; }
                else if (key == "veryhigh")
                {
                    presetMiB = std::min<VkDeviceSize>((largestLocal / 3) >> 20, 12288);
                    if (presetMiB == 0)
                        presetMiB = 4096;
                    memoryPresetName = "very high";
                }
                else
                    std::fprintf(stderr, "Unknown OPENF4_MEMORY_PRESET '%s' (ultralow, low, medium, veryhigh); using medium\n", value);
            }
            memoryBudgetBytes = envMiB("OPENF4_MEMORY_BUDGET_MB", 0, 65536, presetMiB) << 20;
            // Never more than half the largest device-local heap (very high is a third already).
            if (largestLocal != 0 && memoryBudgetBytes != 0)
                memoryBudgetBytes = std::min(memoryBudgetBytes, (largestLocal / 2) & ~((VkDeviceSize(1) << 20) - 1));
            textureBudgetCapBytes = envMiB("OPENF4_TEXTURE_BUDGET_MB", 64, 65536, 0) << 20;
            VkDeviceSize reserve = 0;
            if (memoryBudgetBytes != 0)
            {
                textureFloorBytes = memoryBudgetBytes / 4;
                sceneHeadroomBytes = memoryBudgetBytes / 16;
                sceneCapBytes = memoryBudgetBytes - textureFloorBytes;
                const VkDeviceSize sharePercent = envMiB("OPENF4_SCENE_SHARE", 10, 90, 50);
                reserve = memoryBudgetBytes / 100 * sharePercent;
                reserve = std::min(envMiB("OPENF4_SCENE_RESERVE_MB", 0, 65536, reserve >> 20) << 20, sceneCapBytes);
                textureBudgetBytes = memoryBudgetBytes - reserve;
            }
            else
            {
                // Uncapped (old behaviour): scene grows on demand, textures up to half the heap.
                sceneCapBytes = 0;
                textureFloorBytes = 0;
                sceneHeadroomBytes = 0;
                reserve = envMiB("OPENF4_SCENE_RESERVE_MB", 0, 65536, 0) << 20;
                textureBudgetBytes = largestLocal != 0 ? largestLocal / 2 : VkDeviceSize(4096) << 20;
            }
            if (largestLocal != 0)
                reserve = std::min(reserve, largestLocal / 2);
            if (textureBudgetCapBytes != 0)
                textureBudgetBytes = std::min(textureBudgetBytes, textureBudgetCapBytes);
            sceneDemandBytes = 0;
            if (const char* value = std::getenv("OPENF4_TEXTURE_MIP_DISTANCE"); value && *value)
            {
                const double parsed = std::strtod(value, nullptr);
                if (std::isfinite(parsed) && parsed >= 16.0 && parsed <= 1.0e6)
                    textureMipDistance = static_cast<float>(parsed);
            }
            if (const char* value = std::getenv("OPENF4_TEXTURE_MIP_BIAS"); value && *value)
            {
                const double parsed = std::strtod(value, nullptr);
                if (std::isfinite(parsed) && parsed >= -4.0 && parsed <= 16.0)
                    textureMipUserBias = static_cast<float>(parsed);
            }
            rendererStats.memoryBudgetBytes = memoryBudgetBytes;
            rendererStats.sceneCapBytes = sceneCapBytes;
            rendererStats.textureBudgetBytes = textureBudgetBytes;
            rendererStats.memoryPreset = memoryPresetName;
            const auto mib = [](VkDeviceSize bytes) { return static_cast<double>(bytes) / 1048576.0; };
            if (memoryBudgetBytes != 0)
                std::printf("Renderer: memory preset %s: %.0f MiB total for scene + textures (scene reserve %.0f MiB, "
                            "scene max %.0f MiB, textures get the rest, at least %.0f MiB), mip distance %.0f\n",
                    memoryPresetName, mib(memoryBudgetBytes), mib(reserve), mib(sceneCapBytes),
                    mib(textureFloorBytes), textureMipDistance);
            else
                std::printf("Renderer: memory budget uncapped (scene reserve %.0f MiB, texture budget %.0f MiB), mip distance %.0f\n",
                    mib(reserve), mib(textureBudgetBytes), textureMipDistance);
            return reserve;
        }

        // Every scene buffer plus both frames' cull lists, as allocated, plus in-flight growths.
        VkDeviceSize sceneCapacityBytes() const
        {
            VkDeviceSize total = vertices.size + indices.size + meshlets.size + instances.size
                + sources.size + materials.size + skinMatrices.size;
            for (const auto& frame : frames)
                total += frame.visible.size;
            // A prefetched growth replaces its buffer: only the extra size stays allocated (the old
            // buffer lives on just until the swap's frame retires).
            for (const auto& growth : pendingGrowths)
                if (growth.target && growth.capacity > growth.target->size)
                    total += growth.capacity - growth.target->size;
            return total;
        }

        // Bytes of the scene buffers actually populated.
        VkDeviceSize sceneUsedBytes() const
        {
            return VkDeviceSize(vertexCount) * sizeof(GpuVertex) + VkDeviceSize(indexCount) * sizeof(std::uint32_t)
                + VkDeviceSize(meshletCount) * sizeof(GpuMeshlet) + VkDeviceSize(instanceCount) * sizeof(GpuInstance)
                + VkDeviceSize(candidateCount) * sizeof(glm::uvec2) + VkDeviceSize(materialCount) * sizeof(GpuMaterial)
                + VkDeviceSize(skinData.size()) * sizeof(glm::mat4)
                + VkDeviceSize(frames.size()) * visibleListBytes(candidateCount);
        }

        // Room the scene may still allocate: inside its maximum share and next to the textures that
        // are resident right now (so scene + textures never exceed the budget).
        VkDeviceSize sceneCapRoom() const
        {
            const VkDeviceSize used = sceneCapacityBytes();
            const VkDeviceSize textures = rendererStats.textureResidentBytes;
            const VkDeviceSize limit = std::min(sceneCapBytes,
                memoryBudgetBytes > textures ? memoryBudgetBytes - textures : VkDeviceSize(0));
            return limit > used ? limit - used : 0;
        }

        // Live texture budget: the unified budget minus what the scene holds (and, while a scene
        // buffer is nearly full or a growth was refused, room kept free for it), never below the floor.
        void refreshMemoryBudget()
        {
            const VkDeviceSize scene = sceneCapacityBytes();
            rendererStats.sceneAllocatedBytes = scene;
            rendererStats.sceneUsedBytes = sceneUsedBytes();
            rendererStats.scenePeakUsedBytes = std::max<std::uint64_t>(
                rendererStats.scenePeakUsedBytes, rendererStats.sceneUsedBytes);
            if (sceneCapBytes == 0)
                return;
            const auto nearlyFull = [](const Buffer& buffer, VkDeviceSize used) {
                return buffer.handle && used * 4 >= buffer.size * 3;
            };
            const bool sceneMayGrow = scene < sceneCapBytes
                && (nearlyFull(vertices, VkDeviceSize(vertexCount) * sizeof(GpuVertex))
                    || nearlyFull(indices, VkDeviceSize(indexCount) * sizeof(std::uint32_t))
                    || nearlyFull(meshlets, VkDeviceSize(meshletCount) * sizeof(GpuMeshlet))
                    || nearlyFull(instances, VkDeviceSize(instanceCount) * sizeof(GpuInstance))
                    || nearlyFull(sources, VkDeviceSize(candidateCount) * sizeof(glm::uvec2)));
            const VkDeviceSize held = scene + std::max(sceneDemandBytes, sceneMayGrow ? sceneHeadroomBytes : VkDeviceSize(0));
            VkDeviceSize budget = memoryBudgetBytes > held ? memoryBudgetBytes - held : 0;
            const VkDeviceSize free = memoryBudgetBytes > scene ? memoryBudgetBytes - scene : 0;
            budget = std::max(budget, std::min(textureFloorBytes, free));
            if (textureBudgetCapBytes != 0)
                budget = std::min(budget, textureBudgetCapBytes);
            if (budget < textureBudgetBytes)
                residencyDirty = true;  // unload promptly when the scene takes room
            textureBudgetBytes = budget;
            rendererStats.textureBudgetBytes = budget;
            rendererStats.textureBudgetMinBytes = rendererStats.textureBudgetMinBytes == 0
                ? budget : std::min<std::uint64_t>(rendererStats.textureBudgetMinBytes, budget);
        }

        // Scene cap: throws (before any state changes) when an append would need the scene
        // buffers to grow past the cap. Callers already treat a failed append as a skipped model.
        void reserveSceneGrowth(VkDeviceSize vertexBytes, VkDeviceSize indexBytes, VkDeviceSize meshletBytes,
            VkDeviceSize instanceBytes, VkDeviceSize newCandidates, VkDeviceSize materialBytes, VkDeviceSize skinBytes)
        {
            if (sceneCapBytes == 0)
                return;
            const auto need = [](const Buffer& buffer, VkDeviceSize required) {
                return required > buffer.size ? required - buffer.size : VkDeviceSize(0);
            };
            VkDeviceSize extra = 0;
            extra += need(vertices, VkDeviceSize(vertexCount) * sizeof(GpuVertex) + vertexBytes);
            extra += need(indices, VkDeviceSize(indexCount) * sizeof(std::uint32_t) + indexBytes);
            extra += need(meshlets, VkDeviceSize(meshletCount) * sizeof(GpuMeshlet) + meshletBytes);
            extra += need(instances, VkDeviceSize(instanceCount) * sizeof(GpuInstance) + instanceBytes);
            const VkDeviceSize candidates = VkDeviceSize(candidateCount) + newCandidates;
            extra += need(sources, candidates * sizeof(glm::uvec2));
            extra += need(materials, VkDeviceSize(materialCount) * sizeof(GpuMaterial) + materialBytes);
            extra += need(skinMatrices, VkDeviceSize(skinData.size()) * sizeof(glm::mat4) + skinBytes);
            for (const auto& frame : frames)
                extra += need(frame.visible, visibleListBytes(static_cast<std::uint32_t>(
                    std::min<VkDeviceSize>(candidates, std::numeric_limits<std::uint32_t>::max()))));
            if (extra != 0 && extra > sceneCapRoom())
            {
                ++rendererStats.sceneCapRejects;
                // Room the textures hold but the scene could have: make them unload mips for it.
                const VkDeviceSize used = sceneCapacityBytes();
                if (used + extra <= sceneCapBytes)
                {
                    sceneDemandBytes = std::max(sceneDemandBytes, (extra + (VkDeviceSize(16) << 20) - 1)
                        & ~((VkDeviceSize(16) << 20) - 1));
                    refreshMemoryBudget();
                }
                throw std::runtime_error("memory budget reached (" + std::to_string(memoryBudgetBytes >> 20)
                    + " MiB scene + textures, preset " + memoryPresetName + ")");
            }
            if (extra != 0 && sceneDemandBytes != 0)
                sceneDemandBytes = extra >= sceneDemandBytes ? 0 : sceneDemandBytes - extra;
        }

        // Share is in sceneReserveUnits of the reservation (vertices 2304, indices 768, meshlets
        // 192, instances 96, sources 128 (+2x256 for the two frames' cull lists), materials 32,
        // skin 64). Falls back to the old 2x sizing if the reserved allocation fails.
        Buffer createSceneBuffer(VkDeviceSize populated, VkDeviceSize share)
        {
            const VkDeviceSize minimum = streamingBufferCapacity(populated);
            const VkDeviceSize limit = maxStorageRange & ~VkDeviceSize(255);
            const VkDeviceSize reserved = std::min(
                (sceneReserveBytes / sceneReserveUnits * share) & ~VkDeviceSize(255), limit);
            if (reserved > minimum)
            {
                try
                {
                    return createBuffer(reserved, sceneBufferUsage);
                }
                catch (const std::exception&)
                {
                }
            }
            return createBuffer(minimum, sceneBufferUsage);
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
            materialTextureIds.clear();
            for (const auto& material : sceneMaterials)
                materialTextureIds.push_back(materialTextures(material));
            materialTextureIds.resize(gpuMaterials.size());
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
            instanceMaterialSet.clear();
            for (const auto& instance : instanceData)
                instanceMaterialSet.push_back(materialSetFor(instance.meshletRange.x, instance.meshletRange.y));
            materialCount = static_cast<std::uint32_t>(gpuMaterials.size());
            rendererStats.sourceCandidates = candidateCount;

            sceneReserveBytes = resolveSceneReservation();
            const auto reserveStart = std::chrono::steady_clock::now();
            vertices = createSceneBuffer(byteSize(std::span(mesh.vertices)), 2304);
            indices = createSceneBuffer(byteSize(std::span(mesh.indices)), 768);
            meshlets = createSceneBuffer(byteSize(std::span(mesh.meshlets)), 192);
            instances = createSceneBuffer(byteSize(std::span(instanceData)), 96);
            sources = createSceneBuffer(byteSize(std::span(candidateSources)), 128);
            materials = createSceneBuffer(byteSize(std::span(gpuMaterials)), 32);
            skinMatrices = createSceneBuffer(byteSize(std::span(skinData)), 64);
            {
                const VkDeviceSize total = vertices.size + indices.size + meshlets.size + instances.size
                    + sources.size + materials.size + skinMatrices.size + frameCount * sources.size * 2;
                std::printf("Renderer: scene buffers reserved %.0f MiB (vertices %.0f, indices %.0f, cull lists %.0f; %.1f ms)\n",
                    static_cast<double>(total) / 1048576.0, static_cast<double>(vertices.size) / 1048576.0,
                    static_cast<double>(indices.size) / 1048576.0,
                    static_cast<double>(frameCount * sources.size * 2) / 1048576.0,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - reserveStart).count());
            }

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
                if (sizes[index] != 0)
                {
                    const VkBufferCopy copy{offsets[index], 0, sizes[index]};
                    vkCmdCopyBuffer(command, staging.handle, destinations[index]->handle, 1, &copy);
                }
                // Touch the reserved tail now, during load, so the first streamed write into it
                // never pays for the OS / driver paging in fresh memory on a gameplay frame.
                const VkDeviceSize tail = (sizes[index] + 3) & ~VkDeviceSize(3);
                if (tail < destinations[index]->size)
                    vkCmdFillBuffer(command, destinations[index]->handle, tail, VK_WHOLE_SIZE, 0);
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
            {
                std::lock_guard queueLock(queueMutex);
                check(vkQueueSubmit2(queue, 1, &submitInfo, fence), "vkQueueSubmit2");
            }
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
                // Sized for every candidate the sources buffer can hold, so the cull lists grow
                // only when sources does.
                frame.visible = createBuffer(
                    std::max(streamingBufferCapacity(visibleListBytes(candidateCount)), sources.size * 2),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                frame.indirect = createBuffer(indirectCommandCount * sizeof(VkDrawMeshTasksIndirectCommandEXT),
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
            shadowExtent = {cullTuning.shadowTile * 2, cullTuning.shadowTile * 2};
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

        bool sparseTextureFormatSupported(VkFormat format)
        {
            if (const auto found = sparseFormatSupport.find(format); found != sparseFormatSupport.end())
                return found->second;
            std::uint32_t count = 0;
            vkGetPhysicalDeviceSparseImageFormatProperties(physicalDevice, format,
                VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_TILING_OPTIMAL, &count, nullptr);
            return sparseFormatSupport[format] = count != 0;
        }

        // Worker-side driver calls can hold driver locks the frame thread then waits on;
        // OPENF4_STREAM_PROFILE logs the slow ones so frame hitches can be correlated.
        template <class Work>
        auto timedWorker(const char* what, Work&& work) -> decltype(work()) // explicit: GCC needs it for calls placed before this definition
        {
            const auto start = std::chrono::steady_clock::now();
            const auto report = [&] {
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                if (profileUploads && ms >= 4.0)
                    std::printf("    renderer worker: %s %.2f ms @%.1f\n", what, ms, traceAt(start));
            };
            if constexpr (std::is_void_v<decltype(work())>)
            {
                work();
                report();
            }
            else
            {
                auto result = work();
                report();
                return result;
            }
        }

        VkDeviceMemory allocateSparseBlock(std::uint32_t memoryType, VkDeviceSize bytes)
        {
            VkMemoryAllocateInfo allocationInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocationInfo.allocationSize = bytes;
            allocationInfo.memoryTypeIndex = memoryType;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            return vkAllocateMemory(device, &allocationInfo, nullptr, &memory) == VK_SUCCESS
                ? memory : VK_NULL_HANDLE;
        }

        SparsePagePool& sparsePool(std::uint32_t memoryType, VkDeviceSize pageSize)
        {
            for (auto& pool : sparsePools)
                if (pool.memoryType == memoryType && pool.pageSize == pageSize)
                    return pool;
            auto& pool = sparsePools.emplace_back();
            pool.memoryType = memoryType;
            pool.pageSize = std::max<VkDeviceSize>(pageSize, 1);
            constexpr VkDeviceSize blockTarget = VkDeviceSize(64) << 20;
            pool.blockBytes = pool.pageSize * std::max<VkDeviceSize>(1, blockTarget / pool.pageSize);
            return pool;
        }

        void prefetchSparseBlock(SparsePagePool& pool)
        {
            if (pool.pendingBlock.valid() || pool.freePages.size() >= pool.blockBytes / pool.pageSize / 2)
                return;
            const std::uint32_t memoryType = pool.memoryType;
            const VkDeviceSize bytes = pool.blockBytes;
            pool.pendingBlock = std::async(std::launch::async, [this, memoryType, bytes] {
                return timedWorker("sparse block alloc", [&] { return allocateSparseBlock(memoryType, bytes); });
            });
        }

        SparsePage takeSparsePage(SparsePagePool& pool)
        {
            const auto addBlock = [&](VkDeviceMemory memory) {
                pool.blocks.push_back(memory);
                for (VkDeviceSize page = pool.blockBytes / pool.pageSize; page-- > 0;)
                    pool.freePages.push_back({memory, page * pool.pageSize});
            };
            if (pool.freePages.empty())
            {
                const auto stallStart = std::chrono::steady_clock::now();
                if (pool.pendingBlock.valid())
                    if (const VkDeviceMemory memory = pool.pendingBlock.get())
                        addBlock(memory);
                if (pool.freePages.empty())
                {
                    const VkDeviceMemory memory = allocateSparseBlock(pool.memoryType, pool.blockBytes);
                    if (!memory)
                        throw std::runtime_error("vkAllocateMemory sparse texture page block failed");
                    addBlock(memory);
                }
                sparsePoolStallAccumMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - stallStart).count();
            }
            const SparsePage page = pool.freePages.back();
            pool.freePages.pop_back();
            prefetchSparseBlock(pool);
            return page;
        }

        void bindSparseTiles(TextureTarget& target, std::uint32_t mip)
        {
            const VkExtent3D granularity = target.granularity;
            const std::uint32_t width = std::max(1u, target.extent.width >> mip);
            const std::uint32_t height = std::max(1u, target.extent.height >> mip);
            for (std::uint32_t y = 0; y < height; y += granularity.height)
            {
                for (std::uint32_t x = 0; x < width; x += granularity.width)
                {
                    const SparsePage page = takeSparsePage(*target.pool);
                    VkSparseImageMemoryBind bind{};
                    bind.subresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0};
                    bind.offset = {static_cast<std::int32_t>(x), static_cast<std::int32_t>(y), 0};
                    bind.extent = {std::min(granularity.width, width - x),
                        std::min(granularity.height, height - y), 1};
                    bind.memory = page.memory;
                    bind.memoryOffset = page.offset;
                    target.imageBinds.push_back(bind);
                    target.pages.push_back(page);
                    target.pageMips.push_back(static_cast<std::uint8_t>(mip));
                    target.residentBytes += target.pool->pageSize;
                    rendererStats.textureResidentBytes += target.pool->pageSize;
                }
            }
        }

        void bindSparseOpaque(TextureTarget& target, VkDeviceSize resourceOffset, VkDeviceSize size,
            VkSparseMemoryBindFlags flags)
        {
            const VkDeviceSize pageSize = target.pool->pageSize;
            for (VkDeviceSize offset = 0; offset < size; offset += pageSize)
            {
                const SparsePage page = takeSparsePage(*target.pool);
                target.opaqueBinds.push_back({resourceOffset + offset, std::min(pageSize, size - offset),
                    page.memory, page.offset, flags});
                target.pages.push_back(page);
                target.pageMips.push_back(0xFF);
                target.opaqueBytes += pageSize;
                target.residentBytes += pageSize;
                rendererStats.textureResidentBytes += pageSize;
            }
        }

        VkImageView tryCreateTextureView(VkImage image, VkFormat format, std::uint32_t baseMip,
            std::uint32_t mipLevels) const
        {
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = baseMip;
            viewInfo.subresourceRange.levelCount = mipLevels - baseMip;
            viewInfo.subresourceRange.layerCount = 1;
            VkImageView view = VK_NULL_HANDLE;
            return vkCreateImageView(device, &viewInfo, nullptr, &view) == VK_SUCCESS ? view : VK_NULL_HANDLE;
        }

        VkImageView createTextureView(VkImage image, VkFormat format, std::uint32_t baseMip,
            std::uint32_t mipLevels)
        {
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = baseMip;
            viewInfo.subresourceRange.levelCount = mipLevels - baseMip;
            viewInfo.subresourceRange.layerCount = 1;
            VkImageView view = VK_NULL_HANDLE;
            check(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView texture");
            return view;
        }

        void destroyTextureTarget(TextureTarget& texture, bool recycle = true)
        {
            if (texture.sparse && texture.bindTicket != 0)
                drainSparseBinds(texture.bindTicket, true);
            if (texture.view)
                vkDestroyImageView(device, texture.view, nullptr);
            if (texture.heldView)
                vkDestroyImageView(device, texture.heldView, nullptr);
            if (texture.nextView)
                vkDestroyImageView(device, texture.nextView, nullptr);
            if (texture.sparse)
            {
                if (texture.image)
                {
                    const SparseImageKey key{texture.format, texture.extent.width, texture.extent.height,
                        texture.mipLevels};
                    if (recycle && unbindSparseImage(key, texture.image))
                        recycleSparseImage(key, texture.image);
                    else
                        vkDestroyImage(device, texture.image, nullptr);
                }
                if (texture.pool)
                    texture.pool->freePages.insert(
                        texture.pool->freePages.end(), texture.pages.begin(), texture.pages.end());
            }
            else if (texture.image)
                vmaDestroyImage(allocator, texture.image, texture.allocation);
            texture = {};
        }

        // An old view stays alive until neither frame's descriptor set nor any in-flight
        // command buffer can still reference it.
        void retireView(VkImageView view)
        {
            if (view)
                retiredViews.push_back({view, frameSerial});
        }

        void flushRetiredViews()
        {
            for (const auto& retired : retiredViews)
                vkDestroyImageView(device, retired.view, nullptr);
            retiredViews.clear();
        }

        void markTextureDirty(std::uint32_t index)
        {
            for (auto& frame : frames)
                frame.dirtyTextures.push_back(index);
        }

        static VkDeviceSize mipByteSize(const SceneTexture& texture, std::uint32_t mip)
        {
            const std::size_t end = mip + 1 < texture.mipLevels
                ? texture.mipOffsets[mip + 1] : texture.pixels.size();
            return end - texture.mipOffsets[mip];
        }

        static VkImageCreateInfo textureImageInfo(const SceneTexture& texture)
        {
            VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = textureFormat(texture.format);
            imageInfo.extent = {texture.width, texture.height, 1};
            imageInfo.mipLevels = texture.mipLevels;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            return imageInfo;
        }

        // Main-thread decision (caches format support): is a sparse virtual image worth trying?
        bool shouldAttemptSparse(const SceneTexture& texture, bool stream)
        {
            const std::uint32_t cappedMip = detail::sparseFirstResidentMip(
                texture.width, texture.height, texture.mipLevels, texture.mipLevels);
            const std::uint32_t streamMip = stream ? detail::textureStreamInitialMip(texture.width,
                texture.height, texture.mipLevels, 0, texture.mipLevels,
                textureStreamTuning.startDimension) : 0;
            return sparseTexturesSupported && (cappedMip != 0 || streamMip != 0)
                && sparseTextureFormatSupported(textureFormat(texture.format));
        }

        // Thread-safe half of texture creation (device and VMA calls only): creates the image and
        // decides its residency. Streamed textures run this on a worker, because vkCreateImage of
        // a sparse image (GPU VA reservation) or a new VMA block can take 10-30 ms on its own.
        // With stream=true only a small low-resolution mip range is planned resident; finer mips
        // are bound and uploaded later by the streaming pass. stream=false keeps the full
        // (capped) chain resident.
        TexturePlan planTextureImage(VkImageCreateInfo imageInfo, bool attemptSparse, bool stream,
            std::uint32_t startDimension)
        {
            TexturePlan plan;
            const std::uint32_t width = imageInfo.extent.width;
            const std::uint32_t height = imageInfo.extent.height;
            const std::uint32_t mipLevels = imageInfo.mipLevels;
            const std::uint32_t streamMip = stream ? detail::textureStreamInitialMip(
                width, height, mipLevels, 0, mipLevels, startDimension) : 0;
            if (attemptSparse)
            {
                const SparseImageKey key{imageInfo.format, width, height, mipLevels};
                SparseImageShape shape;
                plan.image = acquireSparseImage(key, shape);
                if (plan.image)
                {
                    const std::uint32_t tailFirstLod = std::min(shape.mipTailFirstLod, mipLevels);
                    const std::uint32_t targetMip = detail::sparseFirstResidentMip(
                        width, height, mipLevels, shape.mipTailFirstLod);
                    const std::uint32_t initialMip = stream
                        ? detail::textureStreamInitialMip(width, height, mipLevels, targetMip,
                            tailFirstLod, startDimension)
                        : targetMip;
                    if (initialMip != 0)
                    {
                        plan.sparse = true;
                        plan.poolKey = key;
                        plan.memoryRequirements = shape.memoryRequirements;
                        plan.granularity = shape.granularity;
                        plan.tailFirstLod = tailFirstLod;
                        plan.targetMip = targetMip;
                        plan.firstResidentMip = initialMip;
                        plan.opaqueRanges = std::move(shape.opaqueRanges);
                    }
                    else
                    {
                        recycleSparseImage(key, plan.image);
                        plan.image = VK_NULL_HANDLE;
                    }
                }
            }
            if (!plan.sparse)
            {
                imageInfo.flags = 0;
                VmaAllocationCreateInfo allocationInfo{};
                allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                check(vmaCreateImage(allocator, &imageInfo, &allocationInfo,
                    &plan.image, &plan.allocation, nullptr), "vmaCreateImage texture");
                vkGetImageMemoryRequirements(device, plan.image, &plan.memoryRequirements);
                plan.targetMip = 0;
                plan.firstResidentMip = streamMip;
            }
            if (plan.image)
                plan.view = tryCreateTextureView(plan.image, imageInfo.format, plan.firstResidentMip,
                    imageInfo.mipLevels);
            return plan;
        }

        void destroyPlan(TexturePlan& plan)
        {
            if (plan.view)
                vkDestroyImageView(device, plan.view, nullptr);
            if (plan.sparse && plan.image)
                recycleSparseImage(plan.poolKey, plan.image);
            else if (plan.image)
                vmaDestroyImage(allocator, plan.image, plan.allocation);
            plan = {};
        }

        // Main-thread half: binds the planned resident range from the sparse page pool (no
        // allocation unless the pool's worker-allocated next block is late) and creates the view.
        TextureTarget finishTextureTarget(TexturePlan plan, const SceneTexture& texture)
        {
            TextureTarget target;
            target.image = plan.image;
            target.allocation = plan.allocation;
            target.format = textureFormat(texture.format);
            target.extent = {texture.width, texture.height};
            target.mipLevels = texture.mipLevels;
            target.targetMip = plan.targetMip;
            target.firstResidentMip = plan.firstResidentMip;
            target.floorMip = plan.firstResidentMip;
            target.desiredMip = plan.targetMip;
            target.tailFirstLod = plan.sparse ? plan.tailFirstLod : texture.mipLevels;
            target.virtualBytes = plan.memoryRequirements.size;
            if (plan.sparse)
            {
                target.sparse = true;
                target.pool = &sparsePool(findMemoryType(plan.memoryRequirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), plan.memoryRequirements.alignment);
                target.granularity = plan.granularity;
                for (std::uint32_t mip = plan.firstResidentMip; mip < plan.tailFirstLod; ++mip)
                    bindSparseTiles(target, mip);
                for (const auto& range : plan.opaqueRanges)
                    bindSparseOpaque(target, range.resourceOffset, range.size, range.flags);
                ++rendererStats.sparseTextures;
            }
            else
            {
                target.residentBytes = plan.memoryRequirements.size;
                rendererStats.textureResidentBytes += target.residentBytes;
            }
            target.view = plan.view ? plan.view : createTextureView(target.image, target.format,
                target.firstResidentMip, texture.mipLevels);
            rendererStats.textureVirtualBytes += target.virtualBytes;
            return target;
        }

        TextureTarget createTextureTarget(const SceneTexture& texture, bool stream)
        {
            return finishTextureTarget(planTextureImage(textureImageInfo(texture),
                shouldAttemptSparse(texture, stream), stream, textureStreamTuning.startDimension), texture);
        }

        static VkDeviceSize sparseMipPages(VkExtent3D granularity, std::uint32_t width, std::uint32_t height,
            std::uint32_t mip)
        {
            const std::uint32_t w = std::max(1u, width >> mip);
            const std::uint32_t h = std::max(1u, height >> mip);
            const std::uint32_t gw = std::max(1u, granularity.width);
            const std::uint32_t gh = std::max(1u, granularity.height);
            return VkDeviceSize((w + gw - 1) / gw) * ((h + gh - 1) / gh);
        }

        // Bytes finishTextureTarget will bind for a plan (its initial resident range).
        static VkDeviceSize planResidentBytes(const TexturePlan& plan, const SceneTexture& texture)
        {
            if (!plan.sparse)
                return plan.memoryRequirements.size;
            const VkDeviceSize page = std::max<VkDeviceSize>(plan.memoryRequirements.alignment, 1);
            VkDeviceSize bytes = 0;
            for (std::uint32_t mip = plan.firstResidentMip; mip < plan.tailFirstLod; ++mip)
                bytes += sparseMipPages(plan.granularity, texture.width, texture.height, mip) * page;
            for (const auto& range : plan.opaqueRanges)
                bytes += (range.size + page - 1) / page * page;
            return bytes;
        }

        // Page bytes of one sparse mip level (0 inside the mip tail).
        static VkDeviceSize mipPageBytes(const TextureTarget& target, std::uint32_t mip)
        {
            if (!target.sparse || !target.pool || mip >= target.tailFirstLod)
                return 0;
            return sparseMipPages(target.granularity, target.extent.width, target.extent.height, mip)
                * target.pool->pageSize;
        }

        // Bound bytes of a sparse texture whose finest resident mip is firstMip.
        static VkDeviceSize residentBytesAt(const TextureTarget& target, std::uint32_t firstMip)
        {
            VkDeviceSize bytes = target.opaqueBytes;
            for (std::uint32_t mip = firstMip; mip < target.tailFirstLod; ++mip)
                bytes += mipPageBytes(target, mip);
            return bytes;
        }

        static std::uint32_t floatBits(float value) { return std::bit_cast<std::uint32_t>(value); }

        static void atomicMinBits(std::uint32_t& slot, std::uint32_t value)
        {
            // Non-negative floats order like their bit patterns.
            std::atomic_ref<std::uint32_t> ref(slot);
            std::uint32_t current = ref.load(std::memory_order_relaxed);
            while (value < current && !ref.compare_exchange_weak(current, value, std::memory_order_relaxed))
            {
            }
        }

        // Finest useful mip for a texture at distance d with total bias (log2 units).
        std::uint32_t distanceMip(const TextureTarget& target, float distance, float bias) const
        {
            const std::uint32_t coarsest = std::max(target.floorMip, target.targetMip);
            if (!std::isfinite(distance))
                return coarsest;
            const float level = std::log2(std::max(distance, 1.0f) / textureMipDistance) + 1.0f + bias;
            if (level <= 0.0f)
                return target.targetMip;
            const float mip = static_cast<float>(target.targetMip) + std::floor(level);
            return mip >= static_cast<float>(coarsest) ? coarsest : static_cast<std::uint32_t>(mip);
        }

        static std::array<std::uint32_t, 8> materialTextures(const SceneMaterial& material)
        {
            if (material.terrain)
                return {material.terrainTextures.x, material.terrainTextures.y, material.terrainTextures.z,
                    material.terrainTextures.w, material.terrainTexturesExtra.x, material.terrainTexturesExtra.y,
                    material.terrainTexturesExtra.z, material.terrainTexturesExtra.w};
            return {material.diffuseTexture, material.normalTexture, material.grayscaleTexture, 0, 0, 0, 0, 0};
        }

        std::uint32_t materialSetFor(std::uint32_t meshletOffset, std::uint32_t count)
        {
            const std::uint64_t key = (std::uint64_t(meshletOffset) << 32) | count;
            if (const auto found = materialSetByRange.find(key); found != materialSetByRange.end())
                return found->second;
            std::vector<std::uint32_t> set;
            for (std::uint32_t meshlet = meshletOffset; meshlet < meshletOffset + count && meshlet < meshletData.size(); ++meshlet)
                set.push_back(meshletData[meshlet].materialIndex);
            std::sort(set.begin(), set.end());
            set.erase(std::unique(set.begin(), set.end()), set.end());
            const auto id = static_cast<std::uint32_t>(materialSets.size());
            materialSets.push_back(std::move(set));
            materialSetByRange.emplace(key, id);
            return id;
        }

        // Releases pages of unloaded mips once no in-flight frame can sample them.
        void releaseRetiredPages()
        {
            std::erase_if(retiredPages, [&](const RetiredPage& retired) {
                if (frameSerial < retired.serial + frameCount + 1)
                    return false;
                retired.pool->freePages.push_back(retired.page);
                rendererStats.textureResidentBytes -= retired.pool->pageSize;
                retiredPageBytes -= retired.pool->pageSize;
                return true;
            });
        }

        // Applies unloads whose coarser view is ready: the view switches, and the dropped mips'
        // pages return to the pool after the frames that may still sample them.
        void applyTextureEvictions()
        {
            if (evictingTextures.empty())
                return;
            if (sparseQueue)
                takeBinderViews();
            const std::uint64_t submitted = sparseQueue ? submittedSparseTicket() : ~std::uint64_t(0);
            std::vector<std::uint32_t> waiting;
            for (const std::uint32_t index : evictingTextures)
            {
                auto& target = textureTargets[index];
                if (!target.evictPending)
                    continue;
                if (target.bindTicket > submitted)
                {
                    waiting.push_back(index);
                    continue;
                }
                VkImageView view = std::exchange(target.nextView, VK_NULL_HANDLE);
                if (!view)
                    view = createTextureView(target.image, target.format, target.evictMip, target.mipLevels);
                retireView(target.view);
                target.view = view;
                const std::uint32_t from = target.firstResidentMip;
                const std::uint32_t to = target.evictMip;
                std::size_t keep = 0;
                for (std::size_t page = 0; page < target.pages.size(); ++page)
                {
                    const std::uint32_t mip = target.pageMips[page];
                    if (mip != 0xFF && mip >= from && mip < to)
                    {
                        retiredPages.push_back({target.pool, target.pages[page], frameSerial});
                        retiredPageBytes += target.pool->pageSize;
                        target.residentBytes -= target.pool->pageSize;
                        continue;
                    }
                    target.pages[keep] = target.pages[page];
                    target.pageMips[keep] = target.pageMips[page];
                    ++keep;
                }
                target.pages.resize(keep);
                target.pageMips.resize(keep);
                target.firstResidentMip = to;
                target.evictPending = false;
                rendererStats.textureEvictions += to - from;
                markTextureDirty(index);
            }
            evictingTextures = std::move(waiting);
        }
        // Residency pass (parallel): nearest referencing instance per texture, then the smallest
        // budget bias that fits every streamed texture's distance mip under the texture budget,
        // unload requests for textures finer than they should be, and a nearest-first promotion order.
        void updateTextureResidency()
        {
            const auto start = std::chrono::steady_clock::now();
            const float inf = std::numeric_limits<float>::infinity();
            const std::uint32_t infBits = floatBits(inf);
            const std::size_t materialTotal = materialTextureIds.size();
            materialDistanceBits.assign(materialTotal, infBits);
            textureDistanceBits.assign(textureTargets.size(), infBits);
            if (streamCameraValid)
            {
                constexpr std::uint32_t chunk = 1024;
                std::vector<std::uint32_t> chunks;
                const auto instanceTotal = static_cast<std::uint32_t>(
                    std::min(instanceData.size(), instanceMaterialSet.size()));
                for (std::uint32_t first = 0; first < instanceTotal; first += chunk)
                    chunks.push_back(first);
                const glm::vec3 camera = streamCamera;
                std::for_each(std::execution::par, chunks.begin(), chunks.end(), [&](std::uint32_t first) {
                    const std::uint32_t last = std::min(first + chunk, instanceTotal);
                    for (std::uint32_t index = first; index < last; ++index)
                    {
                        const GpuInstance& instance = instanceData[index];
                        if (instance.meshletRange.z == 0)
                            continue;
                        const glm::vec3 center(instance.bounds);
                        const float radius = instance.bounds.w;
                        const glm::vec2 flat(center.x - camera.x, center.z - camera.z);
                        const float flatDistance = glm::length(flat);
                        if (flatDistance + radius < instance.lodRange.x || flatDistance - radius > instance.lodRange.w)
                            continue;
                        const float distance = std::max(glm::length(center - camera) - radius, 0.0f);
                        const std::uint32_t bits = floatBits(distance);
                        for (const std::uint32_t material : materialSets[instanceMaterialSet[index]])
                            if (material < materialTotal)
                                atomicMinBits(materialDistanceBits[material], bits);
                    }
                });
                std::vector<std::uint32_t> materialChunks;
                for (std::uint32_t first = 0; first < materialTotal; first += chunk)
                    materialChunks.push_back(first);
                const std::size_t textureTotal = textureTargets.size();
                std::for_each(std::execution::par, materialChunks.begin(), materialChunks.end(), [&](std::uint32_t first) {
                    const auto last = static_cast<std::uint32_t>(std::min<std::size_t>(first + chunk, materialTotal));
                    for (std::uint32_t material = first; material < last; ++material)
                    {
                        const std::uint32_t bits = materialDistanceBits[material];
                        if (bits == infBits)
                            continue;
                        for (const std::uint32_t texture : materialTextureIds[material])
                            if (texture != 0 && texture < textureTotal)
                                atomicMinBits(textureDistanceBits[texture], bits);
                    }
                });
            }
            // Streamed sparse textures are the adjustable part; everything else is fixed.
            VkDeviceSize adjustable = 0;
            for (const auto& entry : streamingTextures)
            {
                auto& target = textureTargets[entry.index];
                target.distance = std::bit_cast<float>(textureDistanceBits[entry.index]);
                if (target.sparse)
                    adjustable += target.residentBytes;
            }
            const VkDeviceSize committed = rendererStats.textureResidentBytes - retiredPageBytes;
            const VkDeviceSize fixed = committed > adjustable ? committed - adjustable : 0;
            // Headroom for textures still being created/bound.
            const VkDeviceSize goal = textureBudgetBytes / 100 * 92;
            const auto neededAt = [&](float bias) {
                return std::transform_reduce(std::execution::par, streamingTextures.begin(), streamingTextures.end(),
                    VkDeviceSize(0), std::plus<>(), [&](const StreamingTexture& entry) {
                        const auto& target = textureTargets[entry.index];
                        if (!target.sparse)
                            return VkDeviceSize(0);
                        return residentBytesAt(target, distanceMip(target, target.distance, bias));
                    });
            };
            float bias = textureMipUserBias;
            if (fixed + neededAt(bias) > goal)
            {
                float low = bias;
                float high = bias + 16.0f;
                for (int step = 0; step < 10; ++step)
                {
                    const float middle = (low + high) * 0.5f;
                    if (fixed + neededAt(middle) > goal)
                        low = middle;
                    else
                        high = middle;
                }
                bias = high;
            }
            residencyBias = bias - textureMipUserBias;
            rendererStats.textureMipBias = residencyBias;
            // Desired mips, unloads (with half a level of hysteresis), promotion order.
            promotionOrder.clear();
            std::vector<TextureViewRequest> evictViews;
            std::uint32_t pendingPromotions = 0;
            for (std::size_t position = 0; position < streamingTextures.size(); ++position)
            {
                const std::uint32_t index = streamingTextures[position].index;
                auto& target = textureTargets[index];
                target.desiredMip = distanceMip(target, target.distance, bias);
                if (target.firstResidentMip > target.desiredMip)
                {
                    ++pendingPromotions;
                    if (!target.evictPending)
                        promotionOrder.push_back(position);
                    continue;
                }
                if (!target.sparse || target.evictPending || target.mipBindPending)
                    continue;
                // Half a level of hysteresis unless the budget is already under pressure.
                const std::uint32_t unloadTo = committed > goal ? target.desiredMip
                    : distanceMip(target, target.distance, bias - 0.5f);
                if (target.firstResidentMip >= unloadTo || evictViews.size() >= 512)
                    continue;
                target.evictMip = unloadTo;
                target.evictPending = true;
                evictingTextures.push_back(index);
                if (sparseQueue)
                {
                    if (target.nextView)
                        retireView(std::exchange(target.nextView, VK_NULL_HANDLE));
                    evictViews.push_back({index, target.image, target.format, unloadTo, target.mipLevels});
                }
            }
            if (!evictViews.empty())
                queueSparseBinds({}, std::move(evictViews));
            std::sort(std::execution::par, promotionOrder.begin(), promotionOrder.end(),
                [&](std::size_t left, std::size_t right) {
                    return textureTargets[streamingTextures[left].index].distance
                        < textureTargets[streamingTextures[right].index].distance;
                });
            rendererStats.streamingTextures = pendingPromotions;
            rendererStats.residencyMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        }

        // Turns worker-created images into live textures (in creation order once ready).
        void finishCreatedTextures()
        {
            for (auto entry = creatingTextures.begin(); entry != creatingTextures.end();)
            {
                if (!entry->ready && entry->plan.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                {
                    ++entry;
                    continue;
                }
                const auto finishStart = std::chrono::steady_clock::now();
                try
                {
                    if (!entry->ready)
                        entry->ready = entry->plan.get();
                    // Texture budget: the slot keeps sampling the default texture until its
                    // coarse resident range fits (the residency pass frees room meanwhile).
                    if (rendererStats.textureResidentBytes + planResidentBytes(*entry->ready, entry->texture)
                        > textureBudgetBytes)
                    {
                        ++rendererStats.textureBudgetDeferrals;
                        ++entry;
                        continue;
                    }
                    TexturePlan plan = std::move(*entry->ready);
                    entry->ready.reset();
                    textureTargets[entry->index] = finishTextureTarget(std::move(plan), entry->texture);
                    pendingTextures.push_back({entry->index, std::move(entry->texture)});
                    markTextureDirty(entry->index);
                }
                catch (const std::exception& error)
                {
                    std::fprintf(stderr, "Texture %u stays on the default texture: %s\n",
                        entry->index, error.what());
                }
                textureCreateAccumMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - finishStart).count();
                entry = creatingTextures.erase(entry);
            }
        }

        static VkImageCreateInfo sparseImageInfo(const SparseImageKey& key)
        {
            VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imageInfo.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = key.format;
            imageInfo.extent = {key.width, key.height, 1};
            imageInfo.mipLevels = key.mipLevels;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            return imageInfo;
        }

        VkImage createSparseImage(const SparseImageKey& key) const
        {
            const VkImageCreateInfo imageInfo = sparseImageInfo(key);
            VkImage image = VK_NULL_HANDLE;
            return vkCreateImage(device, &imageInfo, nullptr, &image) == VK_SUCCESS ? image : VK_NULL_HANDLE;
        }

        // Residency layout shared by every sparse image of one shape.
        SparseImageShape querySparseShape(VkImage image) const
        {
            SparseImageShape shape;
            vkGetImageMemoryRequirements(device, image, &shape.memoryRequirements);
            std::uint32_t sparseCount = 0;
            vkGetImageSparseMemoryRequirements(device, image, &sparseCount, nullptr);
            std::vector<VkSparseImageMemoryRequirements> sparseRequirements(sparseCount);
            if (sparseCount != 0)
                vkGetImageSparseMemoryRequirements(device, image, &sparseCount, sparseRequirements.data());
            const VkSparseImageMemoryRequirements* color = nullptr;
            for (const auto& requirement : sparseRequirements)
            {
                if (requirement.formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT)
                {
                    color = &requirement;
                    break;
                }
            }
            if (!color || color->formatProperties.imageGranularity.width == 0
                || color->formatProperties.imageGranularity.height == 0)
                return shape;
            shape.valid = true;
            shape.granularity = color->formatProperties.imageGranularity;
            shape.mipTailFirstLod = color->imageMipTailFirstLod;
            if (color->imageMipTailSize != 0)
                shape.opaqueRanges.push_back({color->imageMipTailOffset, color->imageMipTailSize,
                    VK_NULL_HANDLE, 0, 0});
            for (const auto& requirement : sparseRequirements)
            {
                if (!(requirement.formatProperties.aspectMask & VK_IMAGE_ASPECT_METADATA_BIT)
                    || requirement.imageMipTailSize == 0)
                    continue;
                shape.opaqueRanges.push_back({requirement.imageMipTailOffset, requirement.imageMipTailSize,
                    VK_NULL_HANDLE, 0, VK_SPARSE_MEMORY_BIND_METADATA_BIT});
            }
            return shape;
        }

        // Called with sparseImagePoolMutex held: queues a refill once a shape runs low. A cold
        // shape (not prewarmed) gets a small bucket after its second miss.
        void requestSparseImageRefill(const SparseImageKey& key, SparseImageBucket& bucket)
        {
            if (bucket.refilling || !sparseImageRefiller.joinable() || (bucket.shape && !bucket.shape->valid))
                return;
            if (bucket.refillTarget == 0)
            {
                if (bucket.misses < 2)
                    return;
                bucket.refillTarget = 8;
            }
            if (bucket.free.size() > bucket.refillTarget / 4)
                return;
            bucket.refilling = true;
            sparseImageRefills.push_back(key);
            sparseImageRefillWake.notify_one();
        }

        // Worker-safe: a pooled sparse image of this exact shape, or a newly created one on a
        // miss. Returns null (shape.valid false) when the shape cannot be a sparse texture.
        VkImage acquireSparseImage(const SparseImageKey& key, SparseImageShape& shape)
        {
            if (sparseImagePoolEnabled)
            {
                std::lock_guard lock(sparseImagePoolMutex);
                auto& bucket = sparseImagePool[key];
                if (bucket.shape && !bucket.shape->valid)
                    return VK_NULL_HANDLE;
                if (!bucket.free.empty())
                {
                    const VkImage image = bucket.free.back();
                    bucket.free.pop_back();
                    shape = *bucket.shape;
                    sparseImagePoolHits.fetch_add(1, std::memory_order_relaxed);
                    requestSparseImageRefill(key, bucket);
                    return image;
                }
                ++bucket.misses;
                sparseImagePoolMisses.fetch_add(1, std::memory_order_relaxed);
                requestSparseImageRefill(key, bucket);
            }
            const VkImage image = createSparseImage(key);
            if (!image)
                return VK_NULL_HANDLE;
            SparseImageShape created = querySparseShape(image);
            if (sparseImagePoolEnabled)
            {
                std::lock_guard lock(sparseImagePoolMutex);
                auto& bucket = sparseImagePool[key];
                if (!bucket.shape)
                    bucket.shape = created;
            }
            if (!created.valid)
            {
                vkDestroyImage(device, image, nullptr);
                return VK_NULL_HANDLE;
            }
            shape = std::move(created);
            return image;
        }

        // Takes back a sparse image with no memory bound (unused, or unbindSparseImage'd).
        void recycleSparseImage(const SparseImageKey& key, VkImage image)
        {
            if (!image)
                return;
            if (sparseImagePoolEnabled)
            {
                std::lock_guard lock(sparseImagePoolMutex);
                if (const auto found = sparseImagePool.find(key); found != sparseImagePool.end()
                    && found->second.shape && found->second.shape->valid
                    && found->second.free.size() < std::max<std::size_t>(64, found->second.refillTarget * 2u))
                {
                    found->second.free.push_back(image);
                    sparseImagePoolRecycled.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
            vkDestroyImage(device, image, nullptr);
        }

        // Frame thread, GPU idle for this image (preview teardown): unbinds every tile and mip
        // tail page so a recycled image never aliases pages handed to another texture.
        bool unbindSparseImage(const SparseImageKey& key, VkImage image)
        {
            if (!sparseImagePoolEnabled)
                return false;
            SparseImageShape shape;
            {
                std::lock_guard lock(sparseImagePoolMutex);
                const auto found = sparseImagePool.find(key);
                if (found == sparseImagePool.end() || !found->second.shape || !found->second.shape->valid)
                    return false;
                shape = *found->second.shape;
            }
            std::vector<VkSparseImageMemoryBind> tiles;
            for (std::uint32_t mip = 0; mip < std::min(shape.mipTailFirstLod, key.mipLevels); ++mip)
            {
                VkSparseImageMemoryBind bind{};
                bind.subresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0};
                bind.extent = {std::max(1u, key.width >> mip), std::max(1u, key.height >> mip), 1};
                tiles.push_back(bind);
            }
            VkSparseImageMemoryBindInfo tileInfo{image, static_cast<std::uint32_t>(tiles.size()), tiles.data()};
            VkSparseImageOpaqueMemoryBindInfo opaqueInfo{image,
                static_cast<std::uint32_t>(shape.opaqueRanges.size()), shape.opaqueRanges.data()};
            VkBindSparseInfo bindInfo{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
            bindInfo.imageBindCount = tiles.empty() ? 0u : 1u;
            bindInfo.pImageBinds = &tileInfo;
            bindInfo.imageOpaqueBindCount = shape.opaqueRanges.empty() ? 0u : 1u;
            bindInfo.pImageOpaqueBinds = &opaqueInfo;
            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VkFence fence = VK_NULL_HANDLE;
            if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
                return false;
            VkResult result = VK_SUCCESS;
            {
                std::lock_guard queueLock(queueMutex);
                result = vkQueueBindSparse(queue, 1, &bindInfo, fence);
            }
            if (result == VK_SUCCESS)
                result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            vkDestroyFence(device, fence, nullptr);
            return result == VK_SUCCESS;
        }

        void runSparseImageRefiller()
        {
            std::unique_lock lock(sparseImagePoolMutex);
            for (;;)
            {
                sparseImageRefillWake.wait(lock, [&] { return sparseImageRefillStop || !sparseImageRefills.empty(); });
                if (sparseImageRefillStop)
                    return;
                const SparseImageKey key = sparseImageRefills.front();
                sparseImageRefills.pop_front();
                auto& bucket = sparseImagePool[key];
                const std::size_t want = bucket.refillTarget > bucket.free.size()
                    ? bucket.refillTarget - bucket.free.size() : 0;
                lock.unlock();
                // One serial batch; images join the bucket as they are created.
                timedWorker("sparse image refill", [&] {
                    for (std::size_t created = 0; created < want; ++created)
                    {
                        const VkImage image = createSparseImage(key);
                        if (!image)
                            break;
                        std::lock_guard pushLock(sparseImagePoolMutex);
                        sparseImagePool[key].free.push_back(image);
                        sparseImagePoolRefilled.fetch_add(1, std::memory_order_relaxed);
                        if (sparseImageRefillStop)
                            break;
                    }
                });
                lock.lock();
                sparseImagePool[key].refilling = false;
            }
        }

        void stopSparseImageRefiller()
        {
            if (!sparseImageRefiller.joinable())
                return;
            {
                std::lock_guard lock(sparseImagePoolMutex);
                sparseImageRefillStop = true;
            }
            sparseImageRefillWake.notify_all();
            sparseImageRefiller.join();
        }

        // Startup: creates the pooled images for the common texture shapes (Fallout 4 BC1/BC3/
        // BC5/BC7, mostly 1024^2 and 512^2) while no frame is in flight, then starts the refiller.
        void prewarmSparseImages()
        {
            double scale = 1.0;
            if (const char* value = std::getenv("OPENF4_SPARSE_IMAGE_POOL"); value && *value)
            {
                const double parsed = std::strtod(value, nullptr);
                if (std::isfinite(parsed) && parsed >= 0.0 && parsed <= 16.0)
                    scale = parsed;
            }
            if (!sparseTexturesSupported || !textureStreamTuning.enabled || scale <= 0.0)
                return;
            sparseImagePoolEnabled = true;
            struct WarmShape
            {
                std::uint32_t width, height, count;
            };
            static constexpr WarmShape shapes[] = {
                {1024, 1024, 384}, {512, 512, 128}, {2048, 2048, 48}, {1024, 512, 16}, {512, 1024, 16},
                {256, 256, 16}, {2048, 1024, 8}, {1024, 2048, 8}, {512, 256, 8}};
            static constexpr std::pair<VkFormat, double> formats[] = {
                {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 1.0}, {VK_FORMAT_BC3_UNORM_BLOCK, 1.0},
                {VK_FORMAT_BC5_UNORM_BLOCK, 1.0}, {VK_FORMAT_BC7_UNORM_BLOCK, 0.25}};
            const auto start = std::chrono::steady_clock::now();
            std::size_t total = 0;
            for (const auto& [format, formatScale] : formats)
            {
                if (!sparseTextureFormatSupported(format))
                    continue;
                for (const auto& warm : shapes)
                {
                    const std::uint32_t mipLevels = static_cast<std::uint32_t>(
                        std::bit_width(std::max(warm.width, warm.height)));
                    const SparseImageKey key{format, warm.width, warm.height, mipLevels};
                    const auto count = static_cast<std::uint32_t>(std::max(1.0,
                        std::round(warm.count * formatScale * scale)));
                    auto& bucket = sparseImagePool[key];
                    for (std::uint32_t index = 0; index < count; ++index)
                    {
                        const VkImage image = createSparseImage(key);
                        if (!image)
                            break;
                        if (!bucket.shape)
                            bucket.shape = querySparseShape(image);
                        if (!bucket.shape->valid)
                        {
                            vkDestroyImage(device, image, nullptr);
                            break;
                        }
                        bucket.free.push_back(image);
                    }
                    bucket.refillTarget = count;
                    total += bucket.free.size();
                }
            }
            sparseImageRefiller = std::thread([this] { runSparseImageRefiller(); });
            std::printf("Renderer: sparse image pool %zu images, %zu shapes (%.1f ms)\n", total,
                sparseImagePool.size(), std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
        }

        // Creates one sparse image so the page pool (memory type, page size) exists and its
        // first block is allocated on a worker thread before the first streamed texture.
        void warmSparsePool()
        {
            if (!sparseTexturesSupported || !textureStreamTuning.enabled)
                return;
            const VkFormat format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
            if (!sparseTextureFormatSupported(format))
                return;
            VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imageInfo.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = format;
            imageInfo.extent = {1024, 1024, 1};
            imageInfo.mipLevels = 11;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            VkImage image = VK_NULL_HANDLE;
            if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
                return;
            VkMemoryRequirements memoryRequirements{};
            vkGetImageMemoryRequirements(device, image, &memoryRequirements);
            vkDestroyImage(device, image, nullptr);
            auto& pool = sparsePool(findMemoryType(memoryRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), memoryRequirements.alignment);
            prefetchSparseBlock(pool);
            prewarmSparseImages();
        }

        void startSparseBinder()
        {
            vkGetDeviceQueue(device, queueFamily, 1, &sparseQueue);
            VkSemaphoreTypeCreateInfo typeInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            typeInfo.initialValue = 0;
            VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &typeInfo};
            check(vkCreateSemaphore(device, &createInfo, nullptr, &sparseTimeline),
                "vkCreateSemaphore sparse timeline");
            sparseBinder = std::thread([this] { runSparseBinder(); });
        }

        // Joins the binder after it has submitted everything queued; later drains are no-ops.
        void stopSparseBinder()
        {
            if (sparseBinder.joinable())
            {
                {
                    std::lock_guard lock(sparseBindMutex);
                    sparseBinderStop = true;
                }
                sparseBindWake.notify_all();
                sparseBinder.join();
            }
            sparseQueue = VK_NULL_HANDLE;
        }

        void runSparseBinder()
        {
            for (;;)
            {
                std::vector<SparseBindJob> jobs;
                {
                    std::unique_lock lock(sparseBindMutex);
                    sparseBindWake.wait(lock, [&] { return sparseBinderStop || !sparseBindJobs.empty(); });
                    if (sparseBindJobs.empty())
                        return;
                    jobs.assign(std::make_move_iterator(sparseBindJobs.begin()),
                        std::make_move_iterator(sparseBindJobs.end()));
                    sparseBindJobs.clear();
                }
                // Everything queued since the last submit goes out as one batch that signals the
                // newest ticket; timeline waits on older tickets are satisfied by it.
                std::vector<VkSparseImageMemoryBindInfo> imageInfos;
                std::vector<VkSparseImageOpaqueMemoryBindInfo> opaqueInfos;
                std::uint64_t ticket = 0;
                std::vector<TextureViewRequest> viewRequests;
                for (const auto& job : jobs)
                {
                    ticket = std::max(ticket, job.ticket);
                    viewRequests.insert(viewRequests.end(), job.views.begin(), job.views.end());
                    for (const auto& image : job.images)
                    {
                        if (!image.tiles.empty())
                            imageInfos.push_back({image.image,
                                static_cast<std::uint32_t>(image.tiles.size()), image.tiles.data()});
                        if (!image.opaque.empty())
                            opaqueInfos.push_back({image.image,
                                static_cast<std::uint32_t>(image.opaque.size()), image.opaque.data()});
                    }
                }
                VkTimelineSemaphoreSubmitInfo timelineInfo{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
                timelineInfo.signalSemaphoreValueCount = 1;
                timelineInfo.pSignalSemaphoreValues = &ticket;
                VkBindSparseInfo bindInfo{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, &timelineInfo};
                bindInfo.imageBindCount = static_cast<std::uint32_t>(imageInfos.size());
                bindInfo.pImageBinds = imageInfos.data();
                bindInfo.imageOpaqueBindCount = static_cast<std::uint32_t>(opaqueInfos.size());
                bindInfo.pImageOpaqueBinds = opaqueInfos.data();
                bindInfo.signalSemaphoreCount = 1;
                bindInfo.pSignalSemaphores = &sparseTimeline;
                const VkResult result = timedWorker("sparse bind (binder thread)", [&] {
                    return vkQueueBindSparse(sparseQueue, 1, &bindInfo, VK_NULL_HANDLE);
                });
                if (result != VK_SUCCESS)
                    sparseBindResult.store(result);
                std::vector<std::pair<std::uint32_t, VkImageView>> views;
                views.reserve(viewRequests.size());
                for (const auto& request : viewRequests)
                    views.emplace_back(request.texture, tryCreateTextureView(
                        request.image, request.format, request.baseMip, request.mipLevels));
                {
                    std::lock_guard lock(sparseBindMutex);
                    binderViews.insert(binderViews.end(), views.begin(), views.end());
                    sparseTicketSubmitted.store(ticket, std::memory_order_release);
                }
                sparseBindDone.notify_all();
            }
        }

        // Frame thread: hands the targets' unsubmitted binds (and views to pre-create) to the
        // binder thread. Every job's ticket is signalled on the timeline, even without binds.
        void queueSparseBinds(std::span<TextureTarget* const> targets,
            std::vector<TextureViewRequest> views = {})
        {
            SparseBindJob job;
            job.ticket = sparseTicketIssued + 1;
            for (const auto& request : views)
                textureTargets[request.texture].bindTicket = job.ticket;
            job.views = std::move(views);
            for (TextureTarget* target : targets)
            {
                if (!target || !target->sparse
                    || (target->imageBinds.empty() && target->opaqueBinds.empty()))
                    continue;
                job.images.push_back({target->image, std::exchange(target->imageBinds, {}),
                    std::exchange(target->opaqueBinds, {})});
                target->bindTicket = job.ticket;
            }
            if (job.images.empty() && job.views.empty())
                return;
            sparseTicketIssued = job.ticket;
            {
                std::lock_guard lock(sparseBindMutex);
                sparseBindJobs.push_back(std::move(job));
            }
            sparseBindWake.notify_one();
        }

        void takeBinderViews()
        {
            std::vector<std::pair<std::uint32_t, VkImageView>> views;
            {
                std::lock_guard lock(sparseBindMutex);
                views.swap(binderViews);
            }
            for (const auto& [index, view] : views)
            {
                if (index < textureTargets.size() && !textureTargets[index].nextView)
                    textureTargets[index].nextView = view;
                else if (view)
                    retireView(view);
            }
        }

        std::uint64_t submittedSparseTicket()
        {
            const std::uint64_t ticket = sparseTicketSubmitted.load(std::memory_order_acquire);
            if (const VkResult result = sparseBindResult.load(); result != VK_SUCCESS)
                check(result, "vkQueueBindSparse (binder thread)");
            return ticket;
        }

        // Blocks until binds up to ticket are submitted and, if asked, executed on the GPU.
        void drainSparseBinds(std::uint64_t ticket, bool gpuComplete)
        {
            if (!sparseQueue || ticket == 0)
                return;
            {
                std::unique_lock lock(sparseBindMutex);
                sparseBindDone.wait(lock, [&] { return sparseTicketSubmitted.load() >= ticket; });
            }
            if (gpuComplete && sparseBindResult.load() == VK_SUCCESS)
            {
                VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                waitInfo.semaphoreCount = 1;
                waitInfo.pSemaphores = &sparseTimeline;
                waitInfo.pValues = &ticket;
                check(vkWaitSemaphores(device, &waitInfo, UINT64_MAX), "vkWaitSemaphores sparse timeline");
            }
        }

        // Async binds: new sparse textures hand their binds to the binder thread and wait in
        // bindingTextures (descriptor on the default texture); those whose bind has been
        // submitted join this frame's uploads and get their view back.
        void stageAsyncSparseBinds()
        {
            const auto start = std::chrono::steady_clock::now();
            std::vector<TextureTarget*> unbound;
            for (const auto& pending : pendingTextures)
            {
                auto& target = textureTargets[pending.index];
                if (target.sparse && (!target.imageBinds.empty() || !target.opaqueBinds.empty()))
                    unbound.push_back(&target);
            }
            queueSparseBinds(unbound);
            for (auto& pending : pendingTextures)
                bindingTextures.push_back(std::move(pending));
            pendingTextures.clear();
            const std::uint64_t submitted = submittedSparseTicket();
            std::vector<PendingTexture> waiting;
            for (auto& entry : bindingTextures)
            {
                auto& target = textureTargets[entry.index];
                if (target.sparse && target.bindTicket > submitted)
                {
                    if (target.view)
                        target.heldView = std::exchange(target.view, VK_NULL_HANDLE);
                    waiting.push_back(std::move(entry));
                    continue;
                }
                if (target.heldView)
                {
                    target.view = std::exchange(target.heldView, VK_NULL_HANDLE);
                    markTextureDirty(entry.index);
                }
                if (target.sparse)
                    frameSparseWait = std::max(frameSparseWait, target.bindTicket);
                pendingTextures.push_back(std::move(entry));
            }
            bindingTextures = std::move(waiting);
            rendererStats.sparseBindMs += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        }

        bool submitSparseBinds(std::span<TextureTarget*> targets,
            VkSemaphore signalSemaphore = VK_NULL_HANDLE, VkFence fence = VK_NULL_HANDLE)
        {
            std::vector<VkSparseImageMemoryBindInfo> imageInfos;
            std::vector<VkSparseImageOpaqueMemoryBindInfo> opaqueInfos;
            std::vector<TextureTarget*> submitted;
            for (TextureTarget* target : targets)
            {
                if (!target || !target->sparse
                    || (target->imageBinds.empty() && target->opaqueBinds.empty())
                    || std::find(submitted.begin(), submitted.end(), target) != submitted.end())
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
            {
                std::lock_guard queueLock(queueMutex);
                check(vkQueueBindSparse(queue, 1, &bindInfo, fence), "vkQueueBindSparse texture batch");
            }
            for (TextureTarget* target : submitted)
            {
                target->imageBinds.clear();
                target->opaqueBinds.clear();
            }
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
                textureTargets.push_back(createTextureTarget(*texture, false));
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
            {
                std::lock_guard queueLock(queueMutex);
                check(vkQueueSubmit2(queue, 1, &submitInfo, fence), "vkQueueSubmit2 textures");
            }
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
                textureInfos.push_back({textureSampler,
                    texture.view ? texture.view : textureTargets.front().view,
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
            frame.dirtyTextures.clear();
        }

        void updateDescriptorSets()
        {
            for (auto& frame : frames)
                updateDescriptorSet(frame);
        }

        // Brings this frame's set up to date: a full rewrite after buffer/pool changes,
        // otherwise only the bindless slots whose texture view changed (new or promoted).
        void syncTextureDescriptors(Frame& frame)
        {
            if (frame.descriptorGeneration != descriptorGeneration)
            {
                updateDescriptorSet(frame);
                return;
            }
            if (frame.dirtyTextures.empty())
                return;
            std::sort(frame.dirtyTextures.begin(), frame.dirtyTextures.end());
            frame.dirtyTextures.erase(std::unique(frame.dirtyTextures.begin(), frame.dirtyTextures.end()),
                frame.dirtyTextures.end());
            std::vector<VkDescriptorImageInfo> infos;
            std::vector<VkWriteDescriptorSet> writes;
            infos.reserve(frame.dirtyTextures.size());
            writes.reserve(frame.dirtyTextures.size());
            for (const std::uint32_t index : frame.dirtyTextures)
            {
                if (index >= textureTargets.size() || index >= descriptorCapacity)
                    continue;
                const VkImageView view = textureTargets[index].view;
                infos.push_back({textureSampler, view ? view : textureTargets.front().view,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = frame.descriptorSet;
                write.dstBinding = 11;
                write.dstArrayElement = index;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write.pImageInfo = &infos.back();
                writes.push_back(write);
            }
            if (!writes.empty())
                vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            frame.dirtyTextures.clear();
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

        // stream=false (startup scene, editor previews) uploads the whole chain at once so the
        // result is deterministic; world streaming starts low-res and refines over later frames.
        std::uint32_t appendTexture(SceneTexture texture, bool stream = true)
        {
            if (textureTargets.size() >= maxSceneTextures)
                throw std::runtime_error("Scene exceeds the bindless texture capacity");
            if (texture.pixels.empty() || texture.width == 0 || texture.height == 0
                || texture.mipLevels == 0 || texture.mipOffsets.size() < texture.mipLevels)
                throw std::runtime_error("Streamed texture has no image data");
            for (std::uint32_t mip = 0; mip < texture.mipLevels; ++mip)
                if (texture.mipOffsets[mip] >= texture.pixels.size()
                    || (mip + 1 < texture.mipLevels && texture.mipOffsets[mip + 1] < texture.mipOffsets[mip]))
                    stream = false;  // per-mip streaming needs ascending, in-range mip offsets
            const auto index = static_cast<std::uint32_t>(textureTargets.size());
            const auto createStart = std::chrono::steady_clock::now();
            if (stream && textureStreamTuning.enabled && !textureTargets.empty())
            {
                // The slot samples the default texture until the worker has created the image;
                // then its low mips upload and finer mips stream in.
                TextureTarget placeholder;
                placeholder.format = textureFormat(texture.format);
                placeholder.extent = {texture.width, texture.height};
                placeholder.mipLevels = texture.mipLevels;
                textureTargets.push_back(placeholder);
                const VkImageCreateInfo imageInfo = textureImageInfo(texture);
                const bool attemptSparse = shouldAttemptSparse(texture, true);
                const std::uint32_t startDimension = textureStreamTuning.startDimension;
                auto plan = std::async(std::launch::async, [this, imageInfo, attemptSparse, startDimension] {
                    return timedWorker("texture image create", [&] {
                        return planTextureImage(imageInfo, attemptSparse, true, startDimension);
                    });
                });
                creatingTextures.push_back({index, std::move(texture), std::move(plan)});
                markTextureDirty(index);
                textureCreateAccumMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - createStart).count();
                rendererStats.residentTextures = static_cast<std::uint32_t>(textureTargets.size());
                return index;
            }
            textureTargets.push_back(createTextureTarget(texture, false));
            const double createMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - createStart).count();
            textureCreateAccumMs += createMs;
            if (profileUploads && createMs >= 2.0)
                std::printf("    renderer: texture create %.2f ms (%ux%u, %u mips, format %d, %s, first mip %u)\n",
                    createMs, texture.width, texture.height, texture.mipLevels,
                    static_cast<int>(texture.format), textureTargets.back().sparse ? "sparse" : "VMA",
                    textureTargets.back().firstResidentMip);
            pendingTextures.push_back({index, std::move(texture)});
            markTextureDirty(index);
            rendererStats.residentTextures = static_cast<std::uint32_t>(textureTargets.size());
            return index;
        }

        std::uint32_t appendMaterial(const SceneMaterial& material)
        {
            const GpuMaterial gpu = makeGpuMaterial(material);
            reserveSceneGrowth(0, 0, 0, 0, 0, sizeof(GpuMaterial), 0);
            const std::uint32_t index = materialCount++;
            pendingMaterials.push_back(gpu);
            materialTextureIds.resize(materialCount);
            materialTextureIds[index] = materialTextures(material);
            return index;
        }

        void updateMaterial(std::uint32_t index, const SceneMaterial& material)
        {
            if (index >= materialCount) throw std::runtime_error("Material index is out of bounds");
            const auto gpu = makeGpuMaterial(material);
            if (index < materialTextureIds.size())
                materialTextureIds[index] = materialTextures(material);
            const auto pendingBase = materialCount - pendingMaterials.size();
            if (index >= pendingBase) pendingMaterials[index - pendingBase] = gpu;
            else pendingMaterialUpdates[index] = gpu;
        }

        void endPreview()
        {
            if (!preview) return;
            // Preview buffers reuse their tail only after all prior readers have completed.
            waitForFrames();
            {
                std::lock_guard queueLock(queueMutex);
                vkQueueWaitIdle(queue);  // also any sparse binds of preview textures
            }
            drainSparseBinds(sparseTicketIssued, true);
            takeBinderViews();
            const auto saved = *preview;
            const auto trim = [](auto& pending, std::size_t total, std::size_t keep) {
                const auto uploaded = total - pending.size();
                pending.resize(keep > uploaded ? keep - uploaded : 0);
            };
            trim(pendingVertices, vertexCount, saved.vertices);
            trim(pendingIndices, indexCount, saved.indices);
            trim(pendingMeshlets, meshletCount, saved.meshlets);
            trim(pendingInstances, instanceCount, saved.instances);
            trim(pendingSources, candidateCount, saved.candidates);
            trim(pendingMaterials, materialCount, saved.materials);
            std::erase_if(pendingTextures, [&](const PendingTexture& entry) {
                return entry.index >= saved.textures;
            });
            std::erase_if(bindingTextures, [&](const PendingTexture& entry) {
                return entry.index >= saved.textures;
            });
            std::erase_if(boundPromotions, [&](std::uint32_t index) { return index >= saved.textures; });
            for (auto& entry : creatingTextures)
                if (entry.index >= saved.textures)
                {
                    try
                    {
                        TexturePlan plan = entry.ready ? std::move(*entry.ready) : entry.plan.get();
                        destroyPlan(plan);
                    }
                    catch (...)
                    {
                    }
                }
            std::erase_if(creatingTextures, [&](const CreatingTexture& entry) {
                return entry.index >= saved.textures;
            });
            std::erase_if(pendingInstanceUpdates, [&](auto id) { return id >= saved.instances; });
            std::erase_if(pendingMaterialUpdates, [&](const auto& entry) { return entry.first >= saved.materials; });
            std::erase_if(pendingMeshPatches, [&](const auto& patch) { return patch.range.vertexOffset >= saved.vertices; });
            flushRetiredViews();  // both sets are rewritten below
            for (std::size_t i = saved.textures; i < textureTargets.size(); ++i)
                destroyTextureTarget(textureTargets[i]);
            textureTargets.resize(saved.textures);
            std::erase_if(streamingTextures, [&](const StreamingTexture& entry) {
                return entry.index >= saved.textures;
            });
            streamingCursor = 0;
            promotionOrder.clear();
            residencyDirty = true;
            vertexCount = saved.vertices; indexCount = saved.indices; meshletCount = saved.meshlets;
            instanceCount = saved.instances; candidateCount = saved.candidates; materialCount = saved.materials;
            instanceData.resize(instanceCount); meshletData.resize(meshletCount);
            instanceMaterialSet.resize(std::min<std::size_t>(instanceMaterialSet.size(), instanceCount));
            materialTextureIds.resize(std::min<std::size_t>(materialTextureIds.size(), materialCount));
            std::erase_if(materialSetByRange, [&](const auto& entry) {
                return (entry.first >> 32) >= saved.meshlets;
            });
            std::erase_if(evictingTextures, [&](std::uint32_t index) { return index >= saved.textures; });
            sceneCenter = saved.center; sceneRadius = saved.radius;
            rendererStats = saved.stats;
            selectedInstances.clear(); activeSelection.reset();
            preview.reset();
            setSelection(saved.selection, saved.active);
            ++descriptorGeneration;
            updateDescriptorSets();
        }

        std::uint32_t replacePreview(PackedMesh mesh, std::span<const SceneInstance> objects,
            std::span<const SceneMaterial> surfaceMaterials, std::span<const SceneTexture> textures)
        {
            if (mesh.meshlets.empty() || objects.empty() || surfaceMaterials.empty())
                throw std::runtime_error("Preview scene is empty");
            for (const auto& meshlet : mesh.meshlets)
                if (meshlet.materialIndex >= surfaceMaterials.size())
                    throw std::runtime_error("Preview material index is invalid");
            for (const auto& material : surfaceMaterials)
                if (material.diffuseTexture > textures.size() || material.normalTexture > textures.size())
                    throw std::runtime_error("Preview texture index is invalid");
            endPreview();
            preview = PreviewCheckpoint{vertexCount, indexCount, meshletCount, instanceCount,
                candidateCount, materialCount, static_cast<std::uint32_t>(textureTargets.size()),
                sceneCenter, sceneRadius, rendererStats, selectedInstances, activeSelection};
            try
            {
                setSelection({}, std::nullopt);
                const auto materialBase = materialCount;
                const auto textureBase = static_cast<std::uint32_t>(textureTargets.size());
                for (const auto& texture : textures) (void)appendTexture(texture, false);
                for (auto material : surfaceMaterials)
                {
                    if (material.diffuseTexture) material.diffuseTexture += textureBase - 1;
                    if (material.normalTexture) material.normalTexture += textureBase - 1;
                    (void)appendMaterial(material);
                }
                for (auto& meshlet : mesh.meshlets) meshlet.materialIndex += materialBase;
                const auto range = appendPackedMesh(std::move(mesh));
                const auto first = instanceCount;
                sceneCenter = glm::vec3(range.bounds); sceneCenter.x = -sceneCenter.x;
                sceneRadius = std::max(range.bounds.w, 1.0f);
                for (auto object : objects)
                {
                    object.meshletOffset += range.meshletOffset;
                    if (!object.meshletCount) object.meshletCount = range.meshletCount;
                    (void)appendInstance(object);
                }
                return first;
            }
            catch (...) { endPreview(); throw; }
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
            reserveSceneGrowth(mesh.vertices.size() * sizeof(GpuVertex), mesh.indices.size() * sizeof(std::uint32_t),
                mesh.meshlets.size() * sizeof(GpuMeshlet), 0, 0, 0, 0);
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
            for (auto& [key, id] : materialSetByRange)
            {
                if ((key >> 32) != range.meshletOffset)
                    continue;
                auto& set = materialSets[id];
                set.clear();
                const auto count = static_cast<std::uint32_t>(key & 0xFFFFFFFFu);
                for (std::uint32_t meshlet = range.meshletOffset; meshlet < range.meshletOffset + count
                    && meshlet < meshletData.size(); ++meshlet)
                    set.push_back(meshletData[meshlet].materialIndex);
                std::sort(set.begin(), set.end());
                set.erase(std::unique(set.begin(), set.end()), set.end());
            }
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
            reserveSceneGrowth(0, 0, 0, sizeof(GpuInstance), source.meshletCount, 0, 0);
            std::vector<glm::uvec2> candidates;
            candidates.reserve(source.meshletCount);
            for (std::uint32_t meshlet = source.meshletOffset;
                meshlet < source.meshletOffset + source.meshletCount; ++meshlet)
                candidates.emplace_back(instanceCount, meshlet);

            const std::uint32_t index = instanceCount++;
            instanceData.push_back(gpu);
            instanceMaterialSet.resize(instanceCount);
            instanceMaterialSet[index] = materialSetFor(source.meshletOffset, source.meshletCount);
            pendingInstances.push_back(gpu);
            pendingSources.insert(pendingSources.end(), candidates.begin(), candidates.end());
            candidateCount += source.meshletCount;
            rendererStats.sourceCandidates = candidateCount;

            const auto combinedBounds = unionBoundingSpheres(
                glm::vec4(sceneCenter, sceneRadius), source.bounds);
            sceneCenter = glm::vec3(combinedBounds);
            sceneRadius = std::max(combinedBounds.w, 1.0f);
            return index;
        }

        std::uint32_t appendSkinPalette(std::span<const glm::mat4> transforms)
        {
            if (transforms.empty())
                throw std::runtime_error("Skin palette cannot be empty");
            if (skinData.size() + transforms.size() > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("Skin palette index exceeds 32 bits");
            reserveSceneGrowth(0, 0, 0, 0, 0, 0, transforms.size() * sizeof(glm::mat4));
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

        void updateLodRange(std::uint32_t index, glm::vec4 range)
        {
            if (index >= instanceData.size())
                throw std::runtime_error("Streamed instance index is out of bounds");
            if (!std::isfinite(range.x) || !std::isfinite(range.y)
                || !std::isfinite(range.z) || !std::isfinite(range.w)
                || range.x < 0.0f || range.y < range.x || range.z < range.y || range.w < range.z)
                throw std::runtime_error("Invalid instance LOD range");
            if (instanceData[index].lodRange == range) return;
            instanceData[index].lodRange = range;
            const std::size_t pendingBase = instanceData.size() - pendingInstances.size();
            if (index >= pendingBase)
                pendingInstances[index - pendingBase].lodRange = range;
            else
                pendingInstanceUpdates.push_back(index);
        }

        bool recordStreamingUploads(Frame& frame, VkCommandBuffer command)
        {
            frameSparseWait = 0;
            releaseRetiredPages();
            if (!preview)
            {
                applyTextureEvictions();
                finishCreatedTextures();
            }
            if (sparseQueue)
                stageAsyncSparseBinds();
            // Per-frame cull output: grown only for the frame being recorded (its previous use
            // has completed), so the other in-flight frame is never waited on.
            const VkDeviceSize visibleSize = visibleListBytes(candidateCount);
            if (frame.visible.size < visibleSize)
            {
                frame.retiredUploads.push_back(frame.visible);
                VkDeviceSize visibleCapacity = std::max({visibleSize, sources.size * 2,
                    std::max<VkDeviceSize>(frame.visible.size * 2, 256)});
                if (sceneCapBytes != 0)
                    visibleCapacity = std::max(visibleSize, std::min(visibleCapacity, frame.visible.size + sceneCapRoom()));
                frame.visible = createBuffer(visibleCapacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                frame.descriptorGeneration = 0;  // rewrite this frame's set only
            }
            if (!preview && !streamingTextures.empty()
                && (residencyDirty || frameSerial >= residencySerial + residencyInterval
                    || glm::distance(streamCamera, residencyCamera) > residencyMoveDistance))
            {
                updateTextureResidency();
                residencyDirty = false;
                residencySerial = frameSerial;
                residencyCamera = streamCamera;
            }
            const bool streamTextures = !preview && !streamingTextures.empty()
                && (!promotionOrder.empty() || !boundPromotions.empty() || !pendingTextures.empty());
            const bool hasUploads = !pendingVertices.empty() || !pendingIndices.empty()
                || !pendingMeshlets.empty() || !pendingInstances.empty() || !pendingSources.empty()
                || !pendingMaterials.empty() || !pendingMaterialUpdates.empty() || !pendingTextures.empty()
                || !pendingInstanceUpdates.empty() || !pendingMeshPatches.empty() || skinDataDirty
                || streamTextures;
            if (!hasUploads)
            {
                syncTextureDescriptors(frame);
                return false;
            }

            const auto growStart = std::chrono::steady_clock::now();
            ensureDescriptorCapacity(static_cast<std::uint32_t>(textureTargets.size()));
            bool buffersChanged = false;
            bool growReadBarrier = false;
            const auto grow = [&](Buffer& buffer, VkDeviceSize required) {
                buffersChanged |= growSceneBuffer(frame, command, buffer, required, growReadBarrier);
            };
            grow(vertices, static_cast<VkDeviceSize>(vertexCount) * sizeof(GpuVertex));
            grow(indices, static_cast<VkDeviceSize>(indexCount) * sizeof(std::uint32_t));
            grow(meshlets, static_cast<VkDeviceSize>(meshletCount) * sizeof(GpuMeshlet));
            grow(instances, static_cast<VkDeviceSize>(instanceCount) * sizeof(GpuInstance));
            grow(sources, static_cast<VkDeviceSize>(candidateCount) * sizeof(glm::uvec2));
            grow(materials, static_cast<VkDeviceSize>(materialCount) * sizeof(GpuMaterial));
            grow(skinMatrices, static_cast<VkDeviceSize>(skinData.size()) * sizeof(glm::mat4));
            if (buffersChanged)
            {
                // Growth copies must land before this frame's upload copies patch the new buffers.
                VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
                VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                dependency.memoryBarrierCount = 1;
                dependency.pMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(command, &dependency);
                // Each frame's set picks up the new buffers when that frame is next recorded.
                ++descriptorGeneration;
            }
            rendererStats.uploadGrowMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - growStart).count();

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
            for (const auto& [index, material] : pendingMaterialUpdates)
                addUpload(materials, static_cast<VkDeviceSize>(index) * sizeof(GpuMaterial), &material, sizeof(material));
            if (skinDataDirty)
                addUpload(skinMatrices, 0, skinData.data(), skinData.size() * sizeof(glm::mat4));

            std::vector<VkDeviceSize> textureOffsets;
            std::vector<std::size_t> textureSourceOffsets;
            textureOffsets.reserve(pendingTextures.size());
            textureSourceOffsets.reserve(pendingTextures.size());
            std::vector<TextureTarget*> sparseTargets;
            sparseTargets.reserve(pendingTextures.size());
            for (std::size_t index = 0; index < pendingTextures.size(); ++index)
            {
                const auto& texture = pendingTextures[index].texture;
                auto& target = textureTargets[pendingTextures[index].index];
                const std::size_t sourceOffset = texture.mipOffsets[target.firstResidentMip];
                if (sourceOffset >= texture.pixels.size())
                    throw std::runtime_error("Streamed texture resident mip data is truncated");
                stagingSize = (stagingSize + 15) & ~VkDeviceSize(15);
                textureOffsets.push_back(stagingSize);
                textureSourceOffsets.push_back(sourceOffset);
                stagingSize += texture.pixels.size() - sourceOffset;
                sparseTargets.push_back(&target);
                rendererStats.textureUploadBytes += texture.pixels.size() - sourceOffset;
            }

            // Virtual-texture refinement: each visited texture gains its next finer mip (pages
            // bound from the pool, data copied from the retained CPU chain) within the budget.
            struct MipPromotion
            {
                std::uint32_t texture = 0;
                std::uint32_t mip = 0;
                const SceneTexture* source = nullptr;
                VkDeviceSize stagingOffset = 0;
                VkDeviceSize size = 0;
            };
            std::vector<MipPromotion> promotions;
            if (streamTextures)
            {
                const std::size_t count = promotionOrder.size();
                VkDeviceSize spent = 0;
                const auto addPromotion = [&](const StreamingTexture& entry, std::uint32_t mip, VkDeviceSize bytes) {
                    stagingSize = (stagingSize + 15) & ~VkDeviceSize(15);
                    promotions.push_back({entry.index, mip, &entry.source, stagingSize, bytes});
                    stagingSize += bytes;
                    spent += bytes;
                };
                // Async binds: mips bound on an earlier frame upload once the binder submitted them.
                if (sparseQueue && !boundPromotions.empty())
                {
                    const std::uint64_t submitted = submittedSparseTicket();
                    takeBinderViews();
                    std::vector<std::uint32_t> stillBinding;
                    for (const std::uint32_t index : boundPromotions)
                    {
                        auto& target = textureTargets[index];
                        if (target.bindTicket > submitted)
                        {
                            stillBinding.push_back(index);
                            continue;
                        }
                        const auto entry = std::ranges::find(streamingTextures, index, &StreamingTexture::index);
                        if (entry == streamingTextures.end() || target.firstResidentMip <= target.targetMip)
                        {
                            target.mipBindPending = false;
                            continue;
                        }
                        // mipBindPending stays set until the promotion below is applied: cleared
                        // here, the streaming loop further down saw firstResidentMip unchanged and
                        // bound this same mip again to fresh (never uploaded) pages, and the next
                        // frame then "promoted" the following, never-bound mip (black textures).
                        const std::uint32_t mip = target.firstResidentMip - 1;
                        addPromotion(*entry, mip, mipByteSize(entry->source, mip));
                        frameSparseWait = std::max(frameSparseWait, target.bindTicket);
                    }
                    boundPromotions = std::move(stillBinding);
                }
                std::size_t visited = 0;
                std::size_t bindRequests = 0;
                VkDeviceSize bindSpent = 0;
                std::vector<TextureTarget*> bindNow;
                std::vector<TextureViewRequest> viewNow;
                for (; visited < count && promotions.size() + bindRequests < textureStreamTuning.maxPromotions;
                    ++visited)
                {
                    const auto& entry = streamingTextures[promotionOrder[visited]];
                    auto& target = textureTargets[entry.index];
                    if (target.firstResidentMip <= std::max(target.targetMip, target.desiredMip)
                        || target.mipBindPending || target.evictPending)
                        continue;
                    const std::uint32_t mip = target.firstResidentMip - 1;
                    const VkDeviceSize bytes = mipByteSize(entry.source, mip);
                    // Hard texture budget: never bind past it (unloads free room on later frames).
                    if (rendererStats.textureResidentBytes + mipPageBytes(target, mip) > textureBudgetBytes)
                    {
                        ++rendererStats.textureBudgetDeferrals;
                        continue;
                    }
                    const bool deferToBinder = sparseQueue != VK_NULL_HANDLE;
                    // Always make progress on one mip, even one larger than the whole budget.
                    if (deferToBinder ? bindRequests != 0 && bindSpent + bytes > textureStreamTuning.frameBudget
                                      : !promotions.empty() && spent + bytes > textureStreamTuning.frameBudget)
                        break;
                    if (deferToBinder)
                    {
                        // Pages are taken now; the bind is submitted off-thread and the mip is
                        // uploaded on a later frame (above), so this frame never waits on it.
                        if (target.sparse)
                        {
                            bindSparseTiles(target, mip);
                            bindNow.push_back(&target);
                        }
                        viewNow.push_back({entry.index, target.image, target.format, mip, target.mipLevels});
                        target.mipBindPending = true;
                        boundPromotions.push_back(entry.index);
                        ++bindRequests;
                        bindSpent += bytes;
                        continue;
                    }
                    if (target.sparse)
                    {
                        bindSparseTiles(target, mip);
                        sparseTargets.push_back(&target);
                    }
                    addPromotion(entry, mip, bytes);
                }
                if (!bindNow.empty() || !viewNow.empty())
                {
                    const auto queueStart = std::chrono::steady_clock::now();
                    queueSparseBinds(bindNow, std::move(viewNow));
                    rendererStats.sparseBindMs += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - queueStart).count();
                }
                rendererStats.textureUploadBytes += spent;
            }
            const auto bindStart = std::chrono::steady_clock::now();
            const bool waitForSparse = submitSparseBinds(sparseTargets, frame.sparseReady);
            rendererStats.sparseBindMs += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - bindStart).count();
            if (profileUploads && rendererStats.sparseBindMs >= 4.0)
                std::printf("    renderer: slow sparse bind %.2f ms (%zu targets)\n",
                    rendererStats.sparseBindMs, sparseTargets.size());

            const auto stagingStart = std::chrono::steady_clock::now();
            Buffer& staging = frameStaging(frame, stagingSize);
            const auto stagingDone = std::chrono::steady_clock::now();
            // Staging fill runs in parallel: every copy is split into <= 1 MiB pieces.
            {
                struct StagingCopy
                {
                    std::byte* destination = nullptr;
                    const std::byte* source = nullptr;
                    std::size_t size = 0;
                };
                std::vector<StagingCopy> copies;
                auto* base = static_cast<std::byte*>(staging.mapped);
                const auto addCopy = [&](VkDeviceSize offset, const void* data, std::size_t size) {
                    constexpr std::size_t piece = std::size_t(1) << 20;
                    const auto* source = static_cast<const std::byte*>(data);
                    for (std::size_t done = 0; done < size; done += piece)
                        copies.push_back({base + offset + done, source + done, std::min(piece, size - done)});
                };
                for (const auto& upload : uploads)
                    addCopy(upload.stagingOffset, upload.data, static_cast<std::size_t>(upload.size));
                for (std::size_t index = 0; index < pendingTextures.size(); ++index)
                    addCopy(textureOffsets[index],
                        pendingTextures[index].texture.pixels.data() + textureSourceOffsets[index],
                        pendingTextures[index].texture.pixels.size() - textureSourceOffsets[index]);
                for (const auto& promotion : promotions)
                    addCopy(promotion.stagingOffset,
                        promotion.source->pixels.data() + promotion.source->mipOffsets[promotion.mip],
                        static_cast<std::size_t>(promotion.size));
                if (copies.size() > 4)
                    std::for_each(std::execution::par, copies.begin(), copies.end(), [](const StagingCopy& copy) {
                        std::memcpy(copy.destination, copy.source, copy.size);
                    });
                else
                    for (const auto& copy : copies)
                        std::memcpy(copy.destination, copy.source, copy.size);
            }
            if (stagingSize != 0)
                check(vmaFlushAllocation(allocator, staging.allocation, 0, stagingSize),
                    "vmaFlushAllocation streaming batch");

            // Scene buffers are shared by both in-flight frames: wait for earlier
            // shader reads before overwriting existing geometry/instance/skin data.
            if (!pendingMeshPatches.empty() || !pendingInstanceUpdates.empty() || !pendingMaterialUpdates.empty() || skinDataDirty)
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
                const auto& texture = pendingTextures[index].texture;
                auto& target = textureTargets[pendingTextures[index].index];
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
            // The promoted mip is outside every view until the copy below has completed: views
            // are swapped after recording, and the new view is first sampled after this barrier.
            for (const auto& promotion : promotions)
            {
                const auto& target = textureTargets[promotion.texture];
                imageBarrier(command, target.image, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 1, promotion.mip);
                VkBufferImageCopy copy{};
                copy.bufferOffset = promotion.stagingOffset;
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, promotion.mip, 0, 1};
                copy.imageExtent = {std::max(1u, target.extent.width >> promotion.mip),
                    std::max(1u, target.extent.height >> promotion.mip), 1};
                vkCmdCopyBufferToImage(command, staging.handle, target.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
                imageBarrier(command, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 1, promotion.mip);
            }
            if (!uploads.empty() || buffersChanged)
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

            const auto viewsStart = std::chrono::steady_clock::now();
            for (const auto& promotion : promotions)
            {
                auto& target = textureTargets[promotion.texture];
                retireView(target.view);
                target.view = target.nextView ? std::exchange(target.nextView, VK_NULL_HANDLE)
                    : createTextureView(target.image, target.format, promotion.mip, target.mipLevels);
                target.firstResidentMip = promotion.mip;
                target.mipBindPending = false;
                markTextureDirty(promotion.texture);
            }
            // Promotion pointers into streamingTextures are dead from here on.
            if (!promotions.empty())
                residencyDirty = true;
            if (!promotions.empty())
                std::erase_if(streamingTextures, [&](const StreamingTexture& entry) {
                    // Sparse textures keep their CPU chain: unloaded mips can stream back in.
                    const auto& target = textureTargets[entry.index];
                    return !target.sparse && target.firstResidentMip <= target.targetMip;
                });
            for (auto& pending : pendingTextures)
            {
                const auto& target = textureTargets[pending.index];
                if (target.firstResidentMip > target.targetMip)
                {
                    streamingTextures.push_back({pending.index, std::move(pending.texture)});
                    residencyDirty = true;
                }
            }
            pendingVertices.clear();
            pendingIndices.clear();
            pendingMeshlets.clear();
            pendingInstances.clear();
            pendingInstanceUpdates.clear();
            pendingSources.clear();
            pendingMaterials.clear();
            pendingMaterialUpdates.clear();
            pendingTextures.clear();
            pendingMeshPatches.clear();
            skinDataDirty = false;
            prefetchSceneBuffer(vertices, static_cast<VkDeviceSize>(vertexCount) * sizeof(GpuVertex));
            prefetchSceneBuffer(indices, static_cast<VkDeviceSize>(indexCount) * sizeof(std::uint32_t));
            prefetchSceneBuffer(meshlets, static_cast<VkDeviceSize>(meshletCount) * sizeof(GpuMeshlet));
            prefetchSceneBuffer(instances, static_cast<VkDeviceSize>(instanceCount) * sizeof(GpuInstance));
            prefetchSceneBuffer(sources, static_cast<VkDeviceSize>(candidateCount) * sizeof(glm::uvec2));
            prefetchSceneBuffer(materials, static_cast<VkDeviceSize>(materialCount) * sizeof(GpuMaterial));
            prefetchSceneBuffer(skinMatrices, static_cast<VkDeviceSize>(skinData.size()) * sizeof(glm::mat4));
            const auto syncStart = std::chrono::steady_clock::now();
            syncTextureDescriptors(frame);
            if (profileUploads)
            {
                const auto now = std::chrono::steady_clock::now();
                const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                if (ms(stagingStart, now) >= 4.0)
                    std::printf("    renderer: slow upload tail: staging %.2f, copy+record %.2f, views %.2f (%zu), descriptors %.2f\n",
                        ms(stagingStart, stagingDone), ms(stagingDone, viewsStart), ms(viewsStart, syncStart),
                        promotions.size(), ms(syncStart, now));
            }
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
            const auto shadowFragmentCode = compileShader(shadowFragmentShader, shaderc_fragment_shader, "shadow.frag");
            const VkShaderModule computeModule = shaderModule(computeCode);
            const VkShaderModule meshModule = shaderModule(meshCode);
            const VkShaderModule fragmentModule = shaderModule(fragmentCode);
            const VkShaderModule shadowModule = shaderModule(shadowCode);
            const VkShaderModule shadowFragmentModule = shaderModule(shadowFragmentCode);

            VkComputePipelineCreateInfo computeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            computeInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            computeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeInfo.stage.module = computeModule;
            computeInfo.stage.pName = "main";
            computeInfo.layout = pipelineLayout;
            check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, nullptr, &computePipeline),
                "vkCreateComputePipelines");

            // constant 0: decalPass, constant 2: visibleList (0 opaque/front, 1 decal/back).
            const std::array<VkSpecializationMapEntry, 2> passEntries{
                VkSpecializationMapEntry{0, 0, sizeof(std::uint32_t)},
                VkSpecializationMapEntry{2, sizeof(std::uint32_t), sizeof(std::uint32_t)}};
            const std::array<std::uint32_t, 2> regularPassValue{0, 0};
            const std::array<std::uint32_t, 2> decalPassValue{1, 1};
            const VkSpecializationInfo regularPassSpecialization{
                2, passEntries.data(), sizeof(regularPassValue), regularPassValue.data()};
            const VkSpecializationInfo decalPassSpecialization{
                2, passEntries.data(), sizeof(decalPassValue), decalPassValue.data()};
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
            rasterization.cullMode = VK_CULL_MODE_NONE;
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

            const std::array shadowStages{
                VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, shadowModule, "main", nullptr},
                VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, shadowFragmentModule, "main", nullptr}
            };
            VkPipelineRasterizationStateCreateInfo shadowRasterization = rasterization;
            // Two-sided: cutout cards (branches, foliage, fences) are usually single quads and
            // must cast whichever way they face the sun. Normal offset handles the extra acne.
            shadowRasterization.cullMode = VK_CULL_MODE_NONE;
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
            shadowInfo.stageCount = static_cast<std::uint32_t>(shadowStages.size());
            shadowInfo.pStages = shadowStages.data();
            shadowInfo.pRasterizationState = &shadowRasterization;
            shadowInfo.pDepthStencilState = &shadowDepthState;
            shadowInfo.pColorBlendState = &shadowBlend;
            check(vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1, &shadowInfo, nullptr, &shadowPipeline),
                "vkCreateGraphicsPipelines shadow");

            vkDestroyShaderModule(device, shadowFragmentModule, nullptr);
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
            rasterization.cullMode = VK_CULL_MODE_NONE;
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
            // Same mask pass over the decal list (stored from the back of the visible buffer).
            const std::array<VkSpecializationMapEntry, 2> decalMaskEntries{
                VkSpecializationMapEntry{0, 0, sizeof(std::uint32_t)},
                VkSpecializationMapEntry{2, sizeof(std::uint32_t), sizeof(std::uint32_t)}};
            const std::array<std::uint32_t, 2> decalMaskValues{maskPass, 1};
            const VkSpecializationInfo decalMaskSpecialization{
                2, decalMaskEntries.data(), sizeof(decalMaskValues), decalMaskValues.data()};
            stages[0].pSpecializationInfo = &decalMaskSpecialization;
            check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                &selectionDecalPipeline), "vkCreateGraphicsPipelines selection decal");

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

        void startSubmitThread()
        {
            if (environmentFlagCleared("OPENF4_SUBMIT_THREAD"))
            {
                std::printf("Renderer: GPU submit thread disabled (inline submit)\n");
                return;
            }
#if OPENF4_SUBMIT_STDEXEC
            submitContext.emplace();
            submitScope.emplace();
#else
            submitStop = false;
            submitThread = std::thread([this] { runSubmitThread(); });
#endif
            submitThreadEnabled = true;
            std::printf("Renderer: GPU submit thread (%s)\n",
                OPENF4_SUBMIT_STDEXEC ? "stdexec run_loop + counting_scope" : "std::thread");
        }

        // Submits everything already handed off, then joins the thread.
        void stopSubmitThread()
        {
#if OPENF4_SUBMIT_STDEXEC
            if (submitScope)
            {
                [[maybe_unused]] auto joined = stdexec::sync_wait(submitScope->join());
                submitScope.reset();
            }
            submitContext.reset();
#else
            if (submitThread.joinable())
            {
                {
                    std::lock_guard lock(submitMutex);
                    submitStop = true;
                }
                submitWake.notify_all();
                submitThread.join();
            }
#endif
            submitThreadEnabled = false;
        }

#if !OPENF4_SUBMIT_STDEXEC
        void runSubmitThread()
        {
            for (;;)
            {
                SubmitJob job;
                {
                    std::unique_lock lock(submitMutex);
                    submitWake.wait(lock, [&] { return submitStop || !submitJobs.empty(); });
                    if (submitJobs.empty())
                        return;
                    job = submitJobs.front();
                    submitJobs.pop_front();
                }
                runSubmitJob(job);
            }
        }
#endif

        void enqueueSubmit(const SubmitJob& job)
        {
#if OPENF4_SUBMIT_STDEXEC
            stdexec::spawn(stdexec::schedule(submitContext->get_scheduler())
                    | stdexec::then([this, job]() noexcept { runSubmitJob(job); }),
                submitScope->get_token());
#else
            {
                std::lock_guard lock(submitMutex);
                submitJobs.push_back(job);
            }
            submitWake.notify_one();
#endif
        }

        // Caller holds queueMutex.
        VkResult submitFrame(const SubmitJob& job) noexcept
        {
            VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
            commandInfo.commandBuffer = job.commandBuffer;
            VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            signalInfo.semaphore = job.presentSemaphore;
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            submitInfo.waitSemaphoreInfoCount = job.waitCount;
            submitInfo.pWaitSemaphoreInfos = job.waits.data();
            submitInfo.commandBufferInfoCount = 1;
            submitInfo.pCommandBufferInfos = &commandInfo;
            submitInfo.signalSemaphoreInfoCount = 1;
            submitInfo.pSignalSemaphoreInfos = &signalInfo;
            return vkQueueSubmit2(queue, 1, &submitInfo, job.fence);
        }

        // Caller holds queueMutex and swapchainMutex.
        VkResult presentFrame(const SubmitJob& job) noexcept
        {
            VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &job.presentSemaphore;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &job.swapchain;
            presentInfo.pImageIndices = &job.imageIndex;
            return vkQueuePresentKHR(queue, &presentInfo);
        }

        // Submit thread.
        void runSubmitJob(const SubmitJob& job) noexcept
        {
            const auto start = std::chrono::steady_clock::now();
            auto submitted = start;
            VkResult submitResult = VK_SUCCESS;
            VkResult presentResult = VK_SUCCESS;
            {
                std::lock_guard queueLock(queueMutex);
                submitResult = submitFrame(job);
                submitted = std::chrono::steady_clock::now();
                if (submitResult == VK_SUCCESS)
                {
                    std::lock_guard swapchainLock(swapchainMutex);
                    presentResult = presentFrame(job);
                }
            }
            const auto end = std::chrono::steady_clock::now();
            const auto us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            std::uint64_t previous = submitThreadMaxUs.load(std::memory_order_relaxed);
            while (previous < us && !submitThreadMaxUs.compare_exchange_weak(previous, us))
            {
            }
            if (profileUploads)
            {
                const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                if (ms(start, end) >= 4.0)
                    std::printf("    renderer: submit thread frame %llu: submit %.2f / present %.2f ms @%.1f\n",
                        static_cast<unsigned long long>(job.ticket), ms(start, submitted), ms(submitted, end), traceAt(start));
            }
            {
                std::lock_guard lock(submitMutex);
                if (submitResult != VK_SUCCESS)
                {
                    if (submitError == VK_SUCCESS)
                    {
                        submitError = submitResult;
                        submitErrorWhat = "vkQueueSubmit2 (submit thread)";
                    }
                }
                else if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
                    swapchainStale.store(true);
                else if (presentResult != VK_SUCCESS && submitError == VK_SUCCESS)
                {
                    submitError = presentResult;
                    submitErrorWhat = "vkQueuePresentKHR (submit thread)";
                }
                submitTicketDone = std::max(submitTicketDone, job.ticket);
            }
            submitDone.notify_all();
        }

        // Frame thread: blocks until `ticket` is submitted and presented; rethrows a failure
        // there (that frame's fence would never signal). Returns the time spent waiting.
        double waitSubmitted(std::uint64_t ticket)
        {
            if (!submitThreadEnabled || ticket == 0)
                return 0.0;
            const auto start = std::chrono::steady_clock::now();
            VkResult error = VK_SUCCESS;
            const char* what = "";
            {
                std::unique_lock lock(submitMutex);
                submitDone.wait(lock, [&] { return submitTicketDone >= ticket || submitError != VK_SUCCESS; });
                error = submitError;
                what = submitErrorWhat;
            }
            if (error != VK_SUCCESS)
                check(error, what);
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }

        void drainSubmits()
        {
            waitSubmitted(submitTicketIssued);
        }

        void render(const glm::mat4& view, const glm::mat4& projection,
            const glm::vec3& cameraPosition, const glm::vec3& worldOrigin,
            std::optional<RenderViewport> requestedViewport, const UiDrawCallback& drawUi)
        {
            const auto cpuStart = std::chrono::steady_clock::now();
            Frame& frame = frames[currentFrame];
            // This slot's previous frame (two back) must be submitted before its fence is waited
            // on and presented before this acquire. UI frames drain and submit inline.
            if (drawUi)
            {
                const double waited = waitSubmitted(submitTicketIssued);
                rendererStats.submitWaitMs = waited;
            }
            else
                rendererStats.submitWaitMs = waitSubmitted(frame.submitTicket);
            rendererStats.submitThreadMs = static_cast<double>(submitThreadMaxUs.exchange(0)) / 1000.0;
            if (swapchainStale.exchange(false))
                recreateSwapchain();
            const auto fenceStart = std::chrono::steady_clock::now();
            check(vkWaitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX), "frame fence");
            const auto fenceDone = std::chrono::steady_clock::now();
            rendererStats.frameWaitMs = std::chrono::duration<double, std::milli>(fenceDone - fenceStart).count();
            if (frame.submitted)
            {
                std::array<std::uint64_t, timestampsPerFrame> timestamps{};
                if (vkGetQueryPoolResults(device, queryPool, currentFrame * timestampsPerFrame,
                    timestampsPerFrame, sizeof(timestamps), timestamps.data(), sizeof(std::uint64_t),
                    VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
                {
                    const auto span = [&](std::size_t from, std::size_t to) {
                        return timestamps[to] >= timestamps[from]
                            ? static_cast<double>(timestamps[to] - timestamps[from]) * timestampPeriod / 1'000'000.0
                            : 0.0;
                    };
                    rendererStats.gpuFrameMs = span(0, 4);
                    rendererStats.gpuCullMs = span(0, 1);
                    rendererStats.gpuShadowMs = span(1, 2);
                    rendererStats.gpuWorldMs = span(2, 3);
                    rendererStats.gpuOverlayMs = span(3, 4);
                }
                check(vmaInvalidateAllocation(allocator, frame.indirect.allocation, 0, VK_WHOLE_SIZE),
                    "vmaInvalidateAllocation");
                const auto* counters = static_cast<const std::uint32_t*>(frame.indirect.mapped);
                rendererStats.visibleMeshlets = std::min(counters[0] + counters[3], candidateCount);
                rendererStats.shadowMeshlets = std::min(counters[6], candidateCount);
            }
            const auto queryDone = std::chrono::steady_clock::now();
            // Destroy calls on this thread stalled for as long as a worker held the driver's
            // allocation lock (a 30 ms vkCreateImage made vkDestroyImageView take 30 ms), so
            // everything this frame retires is released in one batch on a worker instead.
            std::erase_if(pendingFrees, [](const std::future<void>& done) {
                return done.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            });
            std::vector<Buffer> releaseBuffers;
            releaseBuffers.reserve(frame.retiredUploads.size());
            for (auto& upload : frame.retiredUploads)
                if (upload.handle)
                    releaseBuffers.push_back(std::exchange(upload, {}));
            frame.retiredUploads.clear();
            std::vector<VkImageView> releaseViews;
            std::erase_if(retiredViews, [&](const RetiredView& retired) {
                if (frameSerial < retired.serial + frameCount)
                    return false;
                releaseViews.push_back(retired.view);
                return true;
            });
            const std::size_t releasedBuffers = releaseBuffers.size();
            const std::size_t releasedViews = releaseViews.size();
            if (!releaseBuffers.empty() || !releaseViews.empty())
                pendingFrees.push_back(std::async(std::launch::async,
                    [this, buffers = std::move(releaseBuffers), views = std::move(releaseViews)]() mutable {
                        timedWorker("release", [&] {
                            for (VkImageView view : views)
                                vkDestroyImageView(device, view, nullptr);
                            for (auto& buffer : buffers)
                                destroyBuffer(buffer);
                        });
                    }));
            rendererStats.retireMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - fenceDone).count();
            if (profileUploads && rendererStats.retireMs >= 4.0)
            {
                const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                std::printf("    renderer: slow retire %.2f ms: query %.2f, release queue %.2f (%zu buffers, %zu views, %zu pending)\n",
                    rendererStats.retireMs, ms(fenceDone, queryDone), ms(queryDone, std::chrono::steady_clock::now()),
                    releasedBuffers, releasedViews, pendingFrees.size());
            }

            std::uint32_t imageIndex = 0;
            const auto acquireStart = std::chrono::steady_clock::now();
            VkResult acquire = VK_SUCCESS;
            {
                std::lock_guard swapchainLock(swapchainMutex);  // vs. the submit thread's present
                acquire = vkAcquireNextImageKHR(
                    device, swapchain, UINT64_MAX, frame.acquired, VK_NULL_HANDLE, &imageIndex);
            }
            const auto acquireDone = std::chrono::steady_clock::now();
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
            // The shadow map used to stretch over the whole loaded scene (a 9x9-cell world put
            // ~20 world units on every texel and rendered every visible meshlet into it). Small
            // scenes (asset preview, cubes) keep that fit; large worlds fit a texel-snapped box
            // around the camera and only draw casters inside it.
            const float shadowHalfExtent = preview ? sceneRadius : std::min(sceneRadius, cullTuning.shadowDistance);
            const bool fitToCamera = shadowHalfExtent < sceneRadius;
            const glm::mat4 lightRotation = glm::lookAtRH(lightDirection, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            const float shadowTileTexels = static_cast<float>(cullTuning.shadowTile);
            const float shadowDepthRange = fitToCamera ? std::max(cullTuning.shadowDistance * 4.0f, 32768.0f) : sceneRadius * 2.5f;
            // Cascaded shadow maps. Large worlds split the view frustum into up to four slices
            // (practical log/uniform split out to 1.5x OPENF4_SHADOW_DISTANCE); each slice gets a
            // bounding sphere (rotation invariant, so the texel grid does not swim while looking
            // around) whose centre snaps to whole texels (no shimmer while moving). Small scenes
            // (asset preview, cubes) keep a single map fitted to the whole scene.
            struct ShadowCascade { glm::vec3 center; float radius; };
            std::array<ShadowCascade, maxShadowCascades> shadowCascades{};
            std::uint32_t shadowCascadeCount = 1;
            float firstCascadeFar = 0.0f;
            if (fitToCamera)
            {
                shadowCascadeCount = cullTuning.shadowCascades;
                const glm::vec3 forward = -glm::normalize(glm::vec3(glm::inverse(view)[2]));
                const float tanX = 1.0f / std::max(std::abs(projection[0][0]), 1.0e-4f);
                const float tanY = 1.0f / std::max(std::abs(projection[1][1]), 1.0e-4f);
                const float corner2 = tanX * tanX + tanY * tanY;
                const float shadowFar = 1.5f * cullTuning.shadowDistance;
                constexpr float splitNear = 16.0f;
                constexpr float logWeight = 0.85f;
                // Fraction of each slice cross-faded into the next cascade (shader cascadeBlend).
                constexpr float cascadeBlend = 0.2f;
                float sliceNear = 0.0f;
                float previousNear = 0.0f;
                for (std::uint32_t cascade = 0; cascade < shadowCascadeCount; ++cascade)
                {
                    const float t = static_cast<float>(cascade + 1) / static_cast<float>(shadowCascadeCount);
                    const float sliceFar = cascade + 1 == shadowCascadeCount ? shadowFar
                        : logWeight * splitNear * std::pow(shadowFar / splitNear, t)
                            + (1.0f - logWeight) * (splitNear + (shadowFar - splitNear) * t);
                    // The previous slice's blend band also samples this cascade, so cover it too.
                    const float coverNear = cascade == 0 ? 0.0f
                        : sliceNear - cascadeBlend * (sliceNear - previousNear);
                    const float centerDistance = std::min(sliceFar, 0.5f * (coverNear + sliceFar) * (1.0f + corner2));
                    const float radius = std::max(
                        std::sqrt((sliceFar - centerDistance) * (sliceFar - centerDistance) + sliceFar * sliceFar * corner2),
                        std::sqrt((centerDistance - coverNear) * (centerDistance - coverNear) + coverNear * coverNear * corner2));
                    // Quantised so FOV/aspect jitter does not rescale the grid every frame.
                    shadowCascades[cascade] = {cameraPosition + forward * centerDistance,
                        std::ceil(radius / 64.0f) * 64.0f};
                    camera.cascadeSplits[static_cast<glm::length_t>(cascade)] = sliceFar;
                    if (cascade == 0)
                        firstCascadeFar = sliceFar;
                    previousNear = sliceNear;
                    sliceNear = sliceFar;
                }
                camera.cameraForward = glm::vec4(forward, 1.0f);
            }
            else
                shadowCascades[0] = {localSceneCenter, shadowHalfExtent};
            glm::vec2 cullMinimum(std::numeric_limits<float>::max());
            glm::vec2 cullMaximum(std::numeric_limits<float>::lowest());
            float cullNear = std::numeric_limits<float>::max();
            float cullFar = std::numeric_limits<float>::lowest();
            for (std::uint32_t cascade = 0; cascade < shadowCascadeCount; ++cascade)
            {
                const auto& [center, radius] = shadowCascades[cascade];
                glm::vec3 centerInLight = glm::vec3(lightRotation * glm::vec4(center, 1.0f));
                const float texel = 2.0f * radius / shadowTileTexels;
                centerInLight.x = std::floor(centerInLight.x / texel) * texel;
                centerInLight.y = std::floor(centerInLight.y / texel) * texel;
                const float nearDistance = -centerInLight.z - shadowDepthRange;
                const float farDistance = -centerInLight.z + shadowDepthRange;
                glm::mat4 cascadeProjection = glm::orthoRH_ZO(
                    centerInLight.x - radius, centerInLight.x + radius,
                    centerInLight.y - radius, centerInLight.y + radius, nearDistance, farDistance);
                cascadeProjection[1][1] *= -1.0f;
                camera.cascadeViewProjection[cascade] = cascadeProjection * lightRotation;
                camera.cascadeTexel[static_cast<glm::length_t>(cascade)] = texel;
                cullMinimum = glm::min(cullMinimum, glm::vec2(centerInLight) - radius);
                cullMaximum = glm::max(cullMaximum, glm::vec2(centerInLight) + radius);
                cullNear = std::min(cullNear, nearDistance);
                cullFar = std::max(cullFar, farDistance);
            }
            camera.shadowParams = glm::vec4(static_cast<float>(shadowCascadeCount),
                1.0f / (2.0f * shadowDepthRange), cullTuning.shadowSoftness, shadowTileTexels);
            // Caster culling uses one box around every cascade.
            glm::mat4 lightProjection = glm::orthoRH_ZO(cullMinimum.x, cullMaximum.x,
                cullMinimum.y, cullMaximum.y, cullNear, cullFar);
            lightProjection[1][1] *= -1.0f;
            camera.lightViewProjection = lightProjection * lightRotation;
            camera.planes = extractFrustum(camera.viewProjection);
            camera.lightPlanes = extractFrustum(camera.lightViewProjection);
            {
                const float viewportPixels = static_cast<float>(requestedViewport
                    ? std::max(viewportHeight, 1u) : std::max(extent.height, 1u));
                // Projected radius in pixels = radius * (0.5 * height * |P[1][1]|) / distance.
                camera.cull.x = cullTuning.minRadiusPixels > 0.0f && !preview
                    ? 0.5f * viewportPixels * std::abs(projection[1][1]) : 0.0f;
                camera.cull.y = cullTuning.minRadiusPixels;
                camera.cull.z = std::numeric_limits<float>::max(); // >0 enables the shadow list
                // Casters smaller than half a shadow texel cannot change a lit/shadowed sample.
                // Half a texel of the finest cascade, scaled with distance in the cull shader.
                camera.cull.w = fitToCamera ? 0.5f * camera.cascadeTexel.x : 0.0f;
                camera.cullFlags.z = fitToCamera ? firstCascadeFar : 0.0f;
                camera.cullFlags.x = cullTuning.coneCulling ? 1.0f : 0.0f;
                camera.cullFlags.y = cullTuning.debugNoAlphaTest ? 1.0f : 0.0f;
            }
            camera.cameraPosition = glm::vec4(cameraPosition, preview ? static_cast<float>(preview->instances) : -1.0f);
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
            const std::uint32_t firstQuery = currentFrame * timestampsPerFrame;
            vkCmdResetQueryPool(frame.commandBuffer, queryPool, firstQuery, timestampsPerFrame);
            vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                queryPool, firstQuery);
            rendererStats.textureCreateMs = textureCreateAccumMs;
            textureCreateAccumMs = 0.0;
            rendererStats.textureUploadBytes = 0;
            rendererStats.uploadGrowMs = 0.0;
            rendererStats.sparseBindMs = 0.0;
            const auto uploadStart = std::chrono::steady_clock::now();
            streamCamera = cameraPosition + worldOrigin;
            streamCameraValid = !preview;
            rendererStats.residencyMs = 0.0;
            refreshMemoryBudget();
            const bool waitForSparse = recordStreamingUploads(frame, frame.commandBuffer);
            rendererStats.streamUploadMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - uploadStart).count();
            if (streamingTextures.empty())
                rendererStats.streamingTextures = 0;
            rendererStats.texturePeakResidentBytes = std::max<std::uint64_t>(
                rendererStats.texturePeakResidentBytes, rendererStats.textureResidentBytes);
            rendererStats.sparsePoolStallMs = sparsePoolStallAccumMs;
            sparsePoolStallAccumMs = 0.0;
            const std::array<VkDrawMeshTasksIndirectCommandEXT, indirectCommandCount> indirectCommands{
                VkDrawMeshTasksIndirectCommandEXT{0, 1, 1}, VkDrawMeshTasksIndirectCommandEXT{0, 1, 1},
                VkDrawMeshTasksIndirectCommandEXT{0, 1, 1}};
            vkCmdUpdateBuffer(frame.commandBuffer, frame.indirect.handle, 0,
                sizeof(indirectCommands), indirectCommands.data());
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
            vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                queryPool, firstQuery + 1);

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
            vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
            vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout, 0, 1, &frame.descriptorSet, 0, nullptr);
            const std::uint32_t shadowTile = cullTuning.shadowTile;
            for (std::uint32_t cascade = 0; cascade < shadowCascadeCount; ++cascade)
            {
                const std::uint32_t tileX = (cascade & 1u) * shadowTile;
                const std::uint32_t tileY = (cascade >> 1u) * shadowTile;
                const VkViewport shadowViewport{static_cast<float>(tileX), static_cast<float>(tileY),
                    static_cast<float>(shadowTile), static_cast<float>(shadowTile), 0.0f, 1.0f};
                const VkRect2D shadowScissor{{static_cast<std::int32_t>(tileX), static_cast<std::int32_t>(tileY)},
                    {shadowTile, shadowTile}};
                vkCmdSetViewport(frame.commandBuffer, 0, 1, &shadowViewport);
                vkCmdSetScissor(frame.commandBuffer, 0, 1, &shadowScissor);
                vkCmdPushConstants(frame.commandBuffer, pipelineLayout, VK_SHADER_STAGE_MESH_BIT_EXT,
                    0, sizeof(cascade), &cascade);
                drawMeshTasksIndirect(frame.commandBuffer, frame.indirect.handle, shadowCommandOffset, 1,
                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
            }
            vkCmdEndRendering(frame.commandBuffer);
            vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                queryPool, firstQuery + 2);
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
                drawMeshTasksIndirect(frame.commandBuffer, frame.indirect.handle, opaqueCommandOffset, 1,
                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
                vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, decalPipeline);
                drawMeshTasksIndirect(frame.commandBuffer, frame.indirect.handle, decalCommandOffset, 1,
                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
            }
            vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                queryPool, firstQuery + 3);
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
                drawMeshTasksIndirect(frame.commandBuffer, frame.indirect.handle, opaqueCommandOffset, 1,
                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
                vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, selectionDecalPipeline);
                drawMeshTasksIndirect(frame.commandBuffer, frame.indirect.handle, decalCommandOffset, 1,
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
                queryPool, firstQuery + 4);
            check(vkEndCommandBuffer(frame.commandBuffer), "vkEndCommandBuffer");
            check(vkResetFences(device, 1, &frame.fence), "vkResetFences");

            std::array<VkSemaphoreSubmitInfo, 3> waitInfos{};
            std::uint32_t waitCount = 0;
            const auto addWait = [&](VkSemaphore semaphore, VkPipelineStageFlags2 stage, std::uint64_t value) {
                auto& info = waitInfos[waitCount++];
                info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
                info.semaphore = semaphore;
                info.stageMask = stage;
                info.value = value;
            };
            addWait(frame.acquired, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0);
            if (waitForSparse)
                addWait(frame.sparseReady, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0);
            // Async binds: the binder thread already submitted this signal (no wait-before-signal).
            if (frameSparseWait != 0)
                addWait(sparseTimeline, VK_PIPELINE_STAGE_2_TRANSFER_BIT, frameSparseWait);
            SubmitJob job;
            job.waits = waitInfos;
            job.waitCount = waitCount;
            job.commandBuffer = frame.commandBuffer;
            job.presentSemaphore = presentSemaphores[imageIndex];
            job.fence = frame.fence;
            job.swapchain = swapchain;
            job.imageIndex = imageIndex;
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            if (submitThreadEnabled && !drawUi)
            {
                // Hand-off only: submit + present (and any driver-lock stall in them) run on the
                // submit thread while this thread moves on to the next frame.
                job.ticket = ++submitTicketIssued;
                frame.submitTicket = job.ticket;
                enqueueSubmit(job);
                frame.submitted = true;
                if (profileUploads && ms(acquireStart, acquireDone) >= 4.0)
                    std::printf("    renderer: slow acquire %.2f ms\n", ms(acquireStart, acquireDone));
            }
            else
            {
                const auto submitStart = std::chrono::steady_clock::now();
                auto submittedAt = submitStart;
                VkResult present = VK_SUCCESS;
                {
                    std::lock_guard queueLock(queueMutex);
                    check(submitFrame(job), "vkQueueSubmit2");
                    frame.submitted = true;
                    submittedAt = std::chrono::steady_clock::now();
                    std::lock_guard swapchainLock(swapchainMutex);
                    present = presentFrame(job);
                }
                if (profileUploads)
                {
                    const auto presentDone = std::chrono::steady_clock::now();
                    if (ms(acquireStart, acquireDone) >= 4.0 || ms(submitStart, submittedAt) >= 4.0
                        || ms(submittedAt, presentDone) >= 4.0)
                        std::printf("    renderer: slow acquire %.2f / submit %.2f / present %.2f ms\n",
                            ms(acquireStart, acquireDone), ms(submitStart, submittedAt), ms(submittedAt, presentDone));
                }
                if (present == VK_ERROR_OUT_OF_DATE_KHR || present == VK_SUBOPTIMAL_KHR)
                    recreateSwapchain();
                else
                    check(present, "vkQueuePresentKHR");
            }

            currentFrame = (currentFrame + 1) % frameCount;
            ++frameSerial;
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

    void VulkanRenderer::updateMaterial(std::uint32_t index, const SceneMaterial& material)
    {
        mImpl->updateMaterial(index, material);
    }

    std::uint32_t VulkanRenderer::replacePreviewScene(PackedMesh mesh,
        std::span<const SceneInstance> instances, std::span<const SceneMaterial> materials,
        std::span<const SceneTexture> textures)
    {
        return mImpl->replacePreview(std::move(mesh), instances, materials, textures);
    }

    void VulkanRenderer::endPreview() { mImpl->endPreview(); }
    std::uint32_t VulkanRenderer::previewMaterialOffset() const noexcept
    { return mImpl->preview ? mImpl->preview->materials : 0; }
    std::uint32_t VulkanRenderer::previewTextureOffset() const noexcept
    { return mImpl->preview ? mImpl->preview->textures : 0; }

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

    void VulkanRenderer::updateInstanceLodRange(std::uint32_t instance, glm::vec4 lodRange)
    {
        mImpl->updateLodRange(instance, lodRange);
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
