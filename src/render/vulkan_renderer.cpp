#include "render/vulkan_renderer.hpp"

#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <glm/gtc/type_ptr.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace empire::render
{
    namespace
    {
        constexpr std::uint32_t frameCount = 2;

        const char* computeShader = R"(
#version 460
layout(local_size_x = 64) in;

struct Vertex { vec4 position; vec4 normalAndColor; vec4 texCoord; };
struct Meshlet { uint vertexOffset; uint indexOffset; uint vertexCount; uint triangleCount; vec4 sphere; uint materialIndex; uint padding0; uint padding1; uint padding2; };
struct Instance { mat4 transform; vec4 bounds; vec4 lodRange; uvec4 meshletRange; };

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
    uvec4 counts;
} camera;

void main()
{
    uint sourceIndex = gl_GlobalInvocationID.x;
    if (sourceIndex >= camera.counts.x)
        return;

    uvec2 source = sources[sourceIndex];
    Instance instance = instances[source.x];
    Meshlet meshlet = meshlets[source.y];
    vec3 center = (instance.transform * vec4(meshlet.sphere.xyz, 1.0)).xyz;
    center.x = -center.x;
    float scale = max(length(instance.transform[0].xyz),
        max(length(instance.transform[1].xyz), length(instance.transform[2].xyz)));
    float radius = meshlet.sphere.w * scale;
    for (uint plane = 0; plane < 6; ++plane) {
        if (dot(camera.planes[plane].xyz, center) + camera.planes[plane].w < -radius)
            return;
    }

    float distanceToCamera = distance(camera.cameraPosition.xyz, center);
    if (distanceToCamera < instance.lodRange.x || distanceToCamera > instance.lodRange.w)
        return;

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

struct Vertex { vec4 position; vec4 normalAndColor; vec4 texCoord; };
struct Meshlet { uint vertexOffset; uint indexOffset; uint vertexCount; uint triangleCount; vec4 sphere; uint materialIndex; uint padding0; uint padding1; uint padding2; };
struct Instance { mat4 transform; vec4 bounds; vec4 lodRange; uvec4 meshletRange; };

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
    uvec4 counts;
} camera;

layout(location = 0) out vec3 fragmentNormal[];
layout(location = 1) out flat uint fragmentInstance[];
layout(location = 2) out vec3 fragmentWorldPosition[];
layout(location = 3) out vec2 fragmentTexCoord[];
layout(location = 4) out flat uint fragmentMaterial[];
layout(location = 5) out vec4 fragmentBlendWeights[];

void main()
{
    uvec2 candidate = visible[gl_WorkGroupID.x];
    Instance instance = instances[candidate.x];
    Meshlet meshlet = meshlets[candidate.y];
    SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);

    for (uint vertexIndex = gl_LocalInvocationIndex;
         vertexIndex < meshlet.vertexCount; vertexIndex += gl_WorkGroupSize.x) {
        Vertex vertex = vertices[meshlet.vertexOffset + vertexIndex];
        vec4 world = instance.transform * vertex.position;
        world.x = -world.x;
        gl_MeshVerticesEXT[vertexIndex].gl_Position = camera.viewProjection * world;
        vec3 worldNormal = mat3(instance.transform) * vertex.normalAndColor.xyz;
        worldNormal.x = -worldNormal.x;
        fragmentNormal[vertexIndex] = normalize(worldNormal);
        fragmentInstance[vertexIndex] = candidate.x;
        fragmentWorldPosition[vertexIndex] = world.xyz;
        fragmentTexCoord[vertexIndex] = vertex.texCoord.xy;
        fragmentMaterial[vertexIndex] = meshlet.materialIndex;
        vec3 blend = vec3(vertex.normalAndColor.w, vertex.texCoord.zw);
        fragmentBlendWeights[vertexIndex] = vec4(blend, max(0.0, 1.0 - dot(blend, vec3(1.0))));
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

struct Instance { mat4 transform; vec4 bounds; vec4 lodRange; uvec4 meshletRange; };
struct Material { vec4 baseColor; uvec4 textureIndices; vec4 parameters; };
layout(std430, binding = 3) readonly buffer Instances { Instance instances[]; };
layout(std430, binding = 9) readonly buffer Materials { Material materials[]; };
layout(std140, binding = 7) uniform Camera {
    mat4 viewProjection;
    mat4 lightViewProjection;
    vec4 planes[6];
    vec4 cameraPosition;
    uvec4 counts;
} camera;
layout(binding = 8) uniform sampler2DShadow shadowMap;
layout(binding = 10) uniform sampler2D diffuseTextures[4096];

layout(location = 0) in vec3 fragmentNormal;
layout(location = 1) in flat uint fragmentInstance;
layout(location = 2) in vec3 fragmentWorldPosition;
layout(location = 3) in vec2 fragmentTexCoord;
layout(location = 4) in flat uint fragmentMaterial;
layout(location = 5) in vec4 fragmentBlendWeights;
layout(location = 0) out vec4 outputColor;

float hash12(vec2 point)
{
    vec3 p3 = fract(vec3(point.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main()
{
    Instance instance = instances[fragmentInstance];
    float distanceToCamera = distance(camera.cameraPosition.xyz, instance.bounds.xyz);
    float fadeIn = instance.lodRange.y <= instance.lodRange.x
        ? step(instance.lodRange.x, distanceToCamera)
        : smoothstep(instance.lodRange.x, instance.lodRange.y, distanceToCamera);
    float fadeOut = instance.lodRange.w <= instance.lodRange.z
        ? 1.0 - step(instance.lodRange.z, distanceToCamera)
        : 1.0 - smoothstep(instance.lodRange.z, instance.lodRange.w, distanceToCamera);
    if (hash12(gl_FragCoord.xy) > fadeIn * fadeOut)
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
    Material material = materials[fragmentMaterial];
    vec4 textureSample;
    if (material.parameters.z > 0.5) {
        vec4 weights = max(fragmentBlendWeights, vec4(0.0));
        weights /= max(dot(weights, vec4(1.0)), 0.0001);
        textureSample =
            texture(diffuseTextures[nonuniformEXT(material.textureIndices.x)], fragmentTexCoord) * weights.x
            + texture(diffuseTextures[nonuniformEXT(material.textureIndices.y)], fragmentTexCoord) * weights.y
            + texture(diffuseTextures[nonuniformEXT(material.textureIndices.z)], fragmentTexCoord) * weights.z
            + texture(diffuseTextures[nonuniformEXT(material.textureIndices.w)], fragmentTexCoord) * weights.w;
    } else {
        textureSample = texture(
            diffuseTextures[nonuniformEXT(material.textureIndices.x)], fragmentTexCoord);
    }
    if (material.parameters.y > 0.0 && textureSample.a < material.parameters.y)
        discard;
    vec3 sampled = textureSample.rgb;
    if (material.textureIndices.w != 0u && material.textureIndices.z != 0u) {
        float gray = dot(sampled, vec3(0.333333));
        sampled = textureLod(diffuseTextures[nonuniformEXT(material.textureIndices.z)],
            vec2(gray, 1.0 - material.parameters.x), 0.0).rgb;
    }
    vec3 base = sampled * material.baseColor.rgb;
    outputColor = vec4(base * (0.18 + diffuse * 0.82), 1.0);
}
)";

        const char* shadowMeshShader = R"(
#version 460
#extension GL_EXT_mesh_shader : require

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

struct Vertex { vec4 position; vec4 normalAndColor; vec4 texCoord; };
struct Meshlet { uint vertexOffset; uint indexOffset; uint vertexCount; uint triangleCount; vec4 sphere; uint materialIndex; uint padding0; uint padding1; uint padding2; };
struct Instance { mat4 transform; vec4 bounds; vec4 lodRange; uvec4 meshletRange; };
layout(std430, binding = 0) readonly buffer Vertices { Vertex vertices[]; };
layout(std430, binding = 1) readonly buffer Indices { uint indices[]; };
layout(std430, binding = 2) readonly buffer Meshlets { Meshlet meshlets[]; };
layout(std430, binding = 3) readonly buffer Instances { Instance instances[]; };
layout(std430, binding = 5) readonly buffer Visible { uvec2 visible[]; };
layout(push_constant) uniform ShadowPush { mat4 lightViewProjection; } shadow;

void main()
{
    uvec2 candidate = visible[gl_WorkGroupID.x];
    Instance instance = instances[candidate.x];
    Meshlet meshlet = meshlets[candidate.y];
    SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);
    for (uint vertexIndex = gl_LocalInvocationIndex;
         vertexIndex < meshlet.vertexCount; vertexIndex += gl_WorkGroupSize.x) {
        Vertex vertex = vertices[meshlet.vertexOffset + vertexIndex];
        vec4 world = instance.transform * vertex.position;
        world.x = -world.x;
        gl_MeshVerticesEXT[vertexIndex].gl_Position = shadow.lightViewProjection * world;
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
            bool sparse = false;
        };

        struct alignas(16) GpuInstance
        {
            glm::mat4 transform{1.0f};
            glm::vec4 bounds{};
            glm::vec4 lodRange{};
            glm::uvec4 meshletRange{};
        };

        struct alignas(16) GpuMaterial
        {
            glm::vec4 baseColor{1.0f};
            glm::uvec4 textureIndices{};
            glm::vec4 parameters{};
        };

        struct alignas(16) CameraData
        {
            glm::mat4 viewProjection{1.0f};
            glm::mat4 lightViewProjection{1.0f};
            std::array<glm::vec4, 6> planes{};
            glm::vec4 cameraPosition{};
            glm::uvec4 counts{};
        };

        static_assert(sizeof(GpuVertex) == 48);
        static_assert(sizeof(GpuMeshlet) == 48);
        static_assert(sizeof(GpuInstance) == 112);
        static_assert(sizeof(GpuMaterial) == 48);
        static_assert(sizeof(CameraData) == 256);

        static GpuMaterial makeGpuMaterial(const SceneMaterial& material)
        {
            if (material.terrain)
                return {material.baseColor, material.terrainTextures,
                    glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)};
            return {material.baseColor,
                glm::uvec4(material.diffuseTexture, material.normalTexture,
                    material.grayscaleTexture, material.grayscaleToPalette ? 1u : 0u),
                glm::vec4(material.grayscaleToPaletteScale, material.alphaCutoff, 0.0f, 0.0f)};
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
            VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
            Buffer uniform;
            Buffer visible;
            Buffer indirect;
            bool submitted = false;
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

        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        std::uint32_t descriptorCapacity = 0;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline computePipeline = VK_NULL_HANDLE;
        VkPipeline graphicsPipeline = VK_NULL_HANDLE;
        VkPipeline shadowPipeline = VK_NULL_HANDLE;
        VkQueryPool queryPool = VK_NULL_HANDLE;
        PFN_vkCmdDrawMeshTasksIndirectEXT drawMeshTasksIndirect = nullptr;

        Buffer vertices;
        Buffer indices;
        Buffer meshlets;
        Buffer instances;
        Buffer sources;
        Buffer materials;
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
            if (graphicsPipeline)
                vkDestroyPipeline(device, graphicsPipeline, nullptr);
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
                destroyBuffer(frame.indirect);
                destroyBuffer(frame.visible);
                destroyBuffer(frame.uniform);
                if (frame.acquired)
                    vkDestroySemaphore(device, frame.acquired, nullptr);
                if (frame.fence)
                    vkDestroyFence(device, frame.fence, nullptr);
            }
            destroyBuffer(sources);
            destroyBuffer(instances);
            destroyBuffer(materials);
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
            application.pApplicationName = "EMPIRE";
            application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
            application.pEngineName = "EMPIRE";
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
                        _dupenv_s(&sparseSetting, &sparseSettingSize, "EMPIRE_DISABLE_SPARSE");
                        sparseDisabled = sparseSetting != nullptr;
                        std::free(sparseSetting);
#else
                        sparseDisabled = std::getenv("EMPIRE_DISABLE_SPARSE") != nullptr;
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
            constexpr std::array extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME};
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
        }

        void destroySwapchain()
        {
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
                VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frameCount * 8},
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
            std::array<VkDescriptorSetLayoutBinding, 11> bindings{};
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
            bindings[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[10].binding = 10;
            bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[10].descriptorCount = maxSceneTextures;
            bindings[10].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            std::array<VkDescriptorBindingFlags, bindings.size()> bindingFlags{};
            bindingFlags[10] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
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
            VkPushConstantRange pushConstant{};
            pushConstant.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
            pushConstant.size = sizeof(glm::mat4);
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstant;
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
            std::vector<GpuInstance> gpuInstances;
            gpuInstances.reserve(sceneInstances.size());
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
                gpuInstances.push_back({source.transform, worldBounds, source.lodRange,
                    glm::uvec4(meshletOffset, rangeMeshletCount, 0, 0)});
                sceneMinimum = glm::min(sceneMinimum, glm::vec3(worldBounds) - worldBounds.w);
                sceneMaximum = glm::max(sceneMaximum, glm::vec3(worldBounds) + worldBounds.w);
            }
            sceneCenter = (sceneMinimum + sceneMaximum) * 0.5f;
            sceneRadius = std::max(glm::distance(sceneCenter, sceneMaximum), 1.0f);
            std::vector<glm::uvec2> candidateSources;
            candidateSources.reserve(sceneInstances.size() * mesh.meshlets.size());
            for (std::uint32_t instanceIndex = 0; instanceIndex < sceneInstances.size(); ++instanceIndex)
            {
                const auto& range = gpuInstances[instanceIndex].meshletRange;
                for (std::uint32_t meshletIndex = range.x; meshletIndex < range.x + range.y; ++meshletIndex)
                    candidateSources.emplace_back(instanceIndex, meshletIndex);
            }
            candidateCount = static_cast<std::uint32_t>(candidateSources.size());
            vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
            indexCount = static_cast<std::uint32_t>(mesh.indices.size());
            meshletCount = static_cast<std::uint32_t>(mesh.meshlets.size());
            instanceCount = static_cast<std::uint32_t>(gpuInstances.size());
            materialCount = static_cast<std::uint32_t>(gpuMaterials.size());
            rendererStats.sourceCandidates = candidateCount;

            vertices = createBuffer(byteSize(std::span(mesh.vertices)),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            indices = createBuffer(byteSize(std::span(mesh.indices)),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            meshlets = createBuffer(byteSize(std::span(mesh.meshlets)),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            instances = createBuffer(byteSize(std::span(gpuInstances)),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            sources = createBuffer(byteSize(std::span(candidateSources)),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            materials = createBuffer(byteSize(std::span(gpuMaterials)),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

            const std::array destinations{&vertices, &indices, &meshlets, &instances, &sources, &materials};
            const std::array sizes{
                byteSize(std::span(mesh.vertices)), byteSize(std::span(mesh.indices)),
                byteSize(std::span(mesh.meshlets)), byteSize(std::span(gpuInstances)),
                byteSize(std::span(candidateSources)), byteSize(std::span(gpuMaterials))
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
                static_cast<const void*>(mesh.meshlets.data()), static_cast<const void*>(gpuInstances.data()),
                static_cast<const void*>(candidateSources.data()), static_cast<const void*>(gpuMaterials.data())
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
                    std::max<VkDeviceSize>(candidateCount * sizeof(glm::uvec2), sizeof(glm::uvec2)),
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

        TextureTarget createTextureTarget(const SceneTexture& texture)
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

            std::uint32_t sparseFormatCount = 0;
            if (sparseTexturesSupported && texture.width >= 1024 && texture.height >= 1024)
                vkGetPhysicalDeviceSparseImageFormatProperties(physicalDevice, format,
                    VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT, imageInfo.usage,
                    VK_IMAGE_TILING_OPTIMAL, &sparseFormatCount, nullptr);
            target.sparse = sparseFormatCount > 0;
            if (target.sparse)
            {
                ++rendererStats.sparseTextures;
                imageInfo.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
                check(vkCreateImage(device, &imageInfo, nullptr, &target.image), "vkCreateImage sparse texture");

                std::uint32_t requirementCount = 0;
                vkGetImageSparseMemoryRequirements(device, target.image, &requirementCount, nullptr);
                std::vector<VkSparseImageMemoryRequirements> sparseRequirements(requirementCount);
                vkGetImageSparseMemoryRequirements(
                    device, target.image, &requirementCount, sparseRequirements.data());
                const auto requirement = std::find_if(sparseRequirements.begin(), sparseRequirements.end(),
                    [](const auto& value) {
                        return value.formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT;
                    });
                if (requirement == sparseRequirements.end())
                    throw std::runtime_error("Sparse texture has no color memory requirement");

                VkMemoryRequirements memoryRequirements{};
                vkGetImageMemoryRequirements(device, target.image, &memoryRequirements);
                std::vector<VkSparseImageMemoryBind> imageBinds;
                VkDeviceSize memorySize = 0;
                const VkExtent3D granularity = requirement->formatProperties.imageGranularity;
                for (std::uint32_t mip = 0; mip < requirement->imageMipTailFirstLod; ++mip)
                {
                    const VkExtent3D mipExtent{
                        std::max(1u, texture.width >> mip), std::max(1u, texture.height >> mip), 1};
                    const VkDeviceSize blockCount =
                        ((mipExtent.width + granularity.width - 1) / granularity.width)
                        * ((mipExtent.height + granularity.height - 1) / granularity.height);
                    memorySize = (memorySize + memoryRequirements.alignment - 1)
                        & ~(memoryRequirements.alignment - 1);
                    VkSparseImageMemoryBind bind{};
                    bind.subresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0};
                    bind.extent = mipExtent;
                    bind.memoryOffset = memorySize;
                    imageBinds.push_back(bind);
                    memorySize += blockCount * memoryRequirements.alignment;
                }
                memorySize = (memorySize + memoryRequirements.alignment - 1)
                    & ~(memoryRequirements.alignment - 1);
                VkSparseMemoryBind tailBind{};
                const bool hasTail = requirement->imageMipTailFirstLod < texture.mipLevels;
                if (hasTail)
                {
                    tailBind.resourceOffset = requirement->imageMipTailOffset;
                    tailBind.size = requirement->imageMipTailSize;
                    tailBind.memoryOffset = memorySize;
                    memorySize += requirement->imageMipTailSize;
                }

                VkMemoryAllocateInfo allocationInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocationInfo.allocationSize = memorySize;
                allocationInfo.memoryTypeIndex = findMemoryType(
                    memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                check(vkAllocateMemory(device, &allocationInfo, nullptr, &target.sparseMemory),
                    "vkAllocateMemory sparse texture");
                for (auto& bind : imageBinds)
                    bind.memory = target.sparseMemory;
                tailBind.memory = target.sparseMemory;

                VkSparseImageMemoryBindInfo imageBindInfo{};
                imageBindInfo.image = target.image;
                imageBindInfo.bindCount = static_cast<std::uint32_t>(imageBinds.size());
                imageBindInfo.pBinds = imageBinds.data();
                VkSparseImageOpaqueMemoryBindInfo opaqueBindInfo{};
                opaqueBindInfo.image = target.image;
                opaqueBindInfo.bindCount = hasTail ? 1u : 0u;
                opaqueBindInfo.pBinds = hasTail ? &tailBind : nullptr;
                VkBindSparseInfo bindInfo{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
                bindInfo.imageBindCount = imageBinds.empty() ? 0u : 1u;
                bindInfo.pImageBinds = imageBinds.empty() ? nullptr : &imageBindInfo;
                bindInfo.imageOpaqueBindCount = hasTail ? 1u : 0u;
                bindInfo.pImageOpaqueBinds = hasTail ? &opaqueBindInfo : nullptr;
                VkFence fence = VK_NULL_HANDLE;
                VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                check(vkCreateFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence sparse texture");
                check(vkQueueBindSparse(queue, 1, &bindInfo, fence), "vkQueueBindSparse");
                check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "sparse texture fence");
                vkDestroyFence(device, fence, nullptr);
            }
            else
            {
                VmaAllocationCreateInfo allocationInfo{};
                allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                check(vmaCreateImage(allocator, &imageInfo, &allocationInfo,
                    &target.image, &target.allocation, nullptr), "vmaCreateImage texture");
            }

            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = target.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = texture.mipLevels;
            viewInfo.subresourceRange.layerCount = 1;
            check(vkCreateImageView(device, &viewInfo, nullptr, &target.view), "vkCreateImageView texture");
            return target;
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
            stagingOffsets.reserve(textures.size());
            textureTargets.reserve(textures.size());
            rendererStats.residentTextures = static_cast<std::uint32_t>(textures.size());
            for (const SceneTexture* texture : textures)
            {
                if (texture->pixels.empty() || texture->width == 0 || texture->height == 0
                    || texture->mipLevels == 0 || texture->mipOffsets.size() < texture->mipLevels)
                    throw std::runtime_error("Scene texture has no image data");
                stagingSize = (stagingSize + 15) & ~VkDeviceSize(15);
                stagingOffsets.push_back(stagingSize);
                stagingSize += texture->pixels.size();
                textureTargets.push_back(createTextureTarget(*texture));
            }
            Buffer staging = createBuffer(stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            for (std::size_t index = 0; index < textures.size(); ++index)
                std::memcpy(static_cast<std::byte*>(staging.mapped) + stagingOffsets[index],
                    textures[index]->pixels.data(), textures[index]->pixels.size());
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
                imageBarrier(command, textureTargets[index].image, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                    texture.mipLevels);
                for (std::uint32_t mip = 0; mip < texture.mipLevels; ++mip)
                {
                    const std::uint32_t offset = texture.mipOffsets[mip];
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
                    texture.mipLevels);
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

        void updateDescriptorSets()
        {
            for (auto& frame : frames)
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
                std::array<VkWriteDescriptorSet, 11> writes{};
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
                writes[10].descriptorCount = static_cast<std::uint32_t>(textureInfos.size());
                writes[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[10].pImageInfo = textureInfos.data();
                vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            }
        }

        void ensureDescriptorCapacity(std::uint32_t required)
        {
            if (required <= descriptorCapacity)
                return;
            replaceDescriptorPool(grownDescriptorCapacity(required));
            updateDescriptorSets();
        }

        std::uint32_t appendTexture(const SceneTexture& texture)
        {
            if (textureTargets.size() >= maxSceneTextures)
                throw std::runtime_error("Scene exceeds the bindless texture capacity");
            if (texture.pixels.empty() || texture.width == 0 || texture.height == 0
                || texture.mipLevels == 0 || texture.mipOffsets.size() < texture.mipLevels)
                throw std::runtime_error("Streamed texture has no image data");
            waitForFrames();
            ensureDescriptorCapacity(static_cast<std::uint32_t>(textureTargets.size() + 1));
            TextureTarget target = createTextureTarget(texture);
            Buffer staging = createBuffer(texture.pixels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            std::memcpy(staging.mapped, texture.pixels.data(), texture.pixels.size());
            check(vmaFlushAllocation(allocator, staging.allocation, 0, VK_WHOLE_SIZE),
                "vmaFlushAllocation streamed texture");

            VkCommandBuffer command = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocateInfo.commandPool = commandPool;
            allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocateInfo.commandBufferCount = 1;
            check(vkAllocateCommandBuffers(device, &allocateInfo, &command),
                "vkAllocateCommandBuffers streamed texture");
            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(command, &beginInfo), "vkBeginCommandBuffer streamed texture");
            imageBarrier(command, target.image, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, texture.mipLevels);
            for (std::uint32_t mip = 0; mip < texture.mipLevels; ++mip)
            {
                VkBufferImageCopy copy{};
                copy.bufferOffset = texture.mipOffsets[mip];
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
                copy.imageExtent = {
                    std::max(1u, texture.width >> mip), std::max(1u, texture.height >> mip), 1};
                vkCmdCopyBufferToImage(command, staging.handle, target.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            }
            imageBarrier(command, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, texture.mipLevels);
            check(vkEndCommandBuffer(command), "vkEndCommandBuffer streamed texture");
            VkFence fence = VK_NULL_HANDLE;
            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            check(vkCreateFence(device, &fenceInfo, nullptr, &fence),
                "vkCreateFence streamed texture");
            VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
            commandInfo.commandBuffer = command;
            VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            submitInfo.commandBufferInfoCount = 1;
            submitInfo.pCommandBufferInfos = &commandInfo;
            check(vkQueueSubmit2(queue, 1, &submitInfo, fence), "vkQueueSubmit2 streamed texture");
            check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
                "streamed texture fence");
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, commandPool, 1, &command);
            destroyBuffer(staging);

            const auto index = static_cast<std::uint32_t>(textureTargets.size());
            textureTargets.push_back(target);
            rendererStats.residentTextures = static_cast<std::uint32_t>(textureTargets.size());
            updateDescriptorSets();
            return index;
        }

        std::uint32_t appendMaterial(const SceneMaterial& material)
        {
            waitForFrames();
            const GpuMaterial gpu = makeGpuMaterial(material);
            const VkDeviceSize offset = static_cast<VkDeviceSize>(materialCount) * sizeof(GpuMaterial);
            ensureBuffer(materials, offset + sizeof(GpuMaterial), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            uploadBytes(materials, offset, &gpu, sizeof(gpu));
            const std::uint32_t index = materialCount++;
            updateDescriptorSets();
            return index;
        }

        MeshRange appendPackedMesh(PackedMesh mesh)
        {
            if (mesh.vertices.empty() || mesh.indices.empty() || mesh.meshlets.empty())
                throw std::runtime_error("Streamed mesh is empty");
            waitForFrames();
            MeshRange range;
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
            const VkDeviceSize vertexOffset = static_cast<VkDeviceSize>(vertexCount) * sizeof(GpuVertex);
            const VkDeviceSize indexOffset = static_cast<VkDeviceSize>(indexCount) * sizeof(std::uint32_t);
            const VkDeviceSize meshletOffset = static_cast<VkDeviceSize>(meshletCount) * sizeof(GpuMeshlet);
            ensureBuffer(vertices, vertexOffset + mesh.vertices.size() * sizeof(GpuVertex),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            ensureBuffer(indices, indexOffset + mesh.indices.size() * sizeof(std::uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            ensureBuffer(meshlets, meshletOffset + mesh.meshlets.size() * sizeof(GpuMeshlet),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            uploadBytes(vertices, vertexOffset, mesh.vertices.data(), mesh.vertices.size() * sizeof(GpuVertex));
            uploadBytes(indices, indexOffset, mesh.indices.data(), mesh.indices.size() * sizeof(std::uint32_t));
            uploadBytes(meshlets, meshletOffset, mesh.meshlets.data(), mesh.meshlets.size() * sizeof(GpuMeshlet));
            vertexCount += static_cast<std::uint32_t>(mesh.vertices.size());
            indexCount += static_cast<std::uint32_t>(mesh.indices.size());
            meshletCount += static_cast<std::uint32_t>(mesh.meshlets.size());
            updateDescriptorSets();
            return range;
        }

        void appendInstance(SceneInstance source)
        {
            if (source.meshletCount == 0 || source.meshletOffset + source.meshletCount > meshletCount)
                throw std::runtime_error("Streamed instance has an invalid meshlet range");
            waitForFrames();
            source.bounds.x = -source.bounds.x;
            const GpuInstance gpu{source.transform, source.bounds, source.lodRange,
                glm::uvec4(source.meshletOffset, source.meshletCount, 0, 0)};
            std::vector<glm::uvec2> candidates;
            candidates.reserve(source.meshletCount);
            for (std::uint32_t meshlet = source.meshletOffset;
                meshlet < source.meshletOffset + source.meshletCount; ++meshlet)
                candidates.emplace_back(instanceCount, meshlet);

            const VkDeviceSize instanceOffset = static_cast<VkDeviceSize>(instanceCount) * sizeof(GpuInstance);
            const VkDeviceSize sourceOffset = static_cast<VkDeviceSize>(candidateCount) * sizeof(glm::uvec2);
            ensureBuffer(instances, instanceOffset + sizeof(GpuInstance), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            ensureBuffer(sources, sourceOffset + candidates.size() * sizeof(glm::uvec2),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            const VkDeviceSize visibleSize = sourceOffset + candidates.size() * sizeof(glm::uvec2);
            for (auto& frame : frames)
                ensureBuffer(frame.visible, visibleSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
            uploadBytes(instances, instanceOffset, &gpu, sizeof(gpu));
            uploadBytes(sources, sourceOffset, candidates.data(), candidates.size() * sizeof(glm::uvec2));
            ++instanceCount;
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
            updateDescriptorSets();
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

            const std::array stages{
                VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule, "main", nullptr},
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
            depth.depthCompareOp = VK_COMPARE_OP_LESS;
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
            VkPipelineRenderingCreateInfo shadowRendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            shadowRendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
            VkGraphicsPipelineCreateInfo shadowInfo = graphicsInfo;
            shadowInfo.pNext = &shadowRendering;
            shadowInfo.stageCount = 1;
            shadowInfo.pStages = &shadowStage;
            shadowInfo.pRasterizationState = &shadowRasterization;
            shadowInfo.pColorBlendState = &shadowBlend;
            check(vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1, &shadowInfo, nullptr, &shadowPipeline),
                "vkCreateGraphicsPipelines shadow");

            vkDestroyShaderModule(device, shadowModule, nullptr);
            vkDestroyShaderModule(device, fragmentModule, nullptr);
            vkDestroyShaderModule(device, meshModule, nullptr);
            vkDestroyShaderModule(device, computeModule, nullptr);
        }

        void imageBarrier(VkCommandBuffer command, VkImage image, VkImageLayout oldLayout,
            VkImageLayout newLayout, VkImageAspectFlags aspect, std::uint32_t levelCount = 1)
        {
            VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            barrier.srcStageMask = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
                ? VK_PIPELINE_STAGE_2_NONE : oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                    : oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                        ? VK_PIPELINE_STAGE_2_TRANSFER_BIT : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
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
                        | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            barrier.dstAccessMask = newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
                : newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                    ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
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
            barrier.subresourceRange.levelCount = levelCount;
            barrier.subresourceRange.layerCount = 1;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.imageMemoryBarrierCount = 1;
            dependency.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command, &dependency);
        }

        void render(const glm::mat4& view, const glm::mat4& projection, const glm::vec3& cameraPosition)
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
            check(vkResetFences(device, 1, &frame.fence), "vkResetFences");
            check(vkResetCommandBuffer(frame.commandBuffer, 0), "vkResetCommandBuffer");

            CameraData camera;
            camera.viewProjection = projection * view;
            const glm::vec3 lightDirection = glm::normalize(glm::vec3(0.35f, 0.8f, 0.25f));
            const glm::mat4 lightView = glm::lookAtRH(
                sceneCenter + lightDirection * sceneRadius * 2.5f,
                sceneCenter, glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 lightProjection = glm::orthoRH_ZO(-sceneRadius, sceneRadius,
                -sceneRadius, sceneRadius, 0.1f, sceneRadius * 5.0f);
            lightProjection[1][1] *= -1.0f;
            camera.lightViewProjection = lightProjection * lightView;
            camera.planes = extractFrustum(camera.viewProjection);
            camera.cameraPosition = glm::vec4(cameraPosition, 1.0f);
            camera.counts = glm::uvec4(candidateCount, candidateCount, 0, 0);
            std::memcpy(frame.uniform.mapped, &camera, sizeof(camera));
            check(vmaFlushAllocation(allocator, frame.uniform.allocation, 0, sizeof(camera)),
                "vmaFlushAllocation");

            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(frame.commandBuffer, &beginInfo), "vkBeginCommandBuffer");
            vkCmdResetQueryPool(frame.commandBuffer, queryPool, currentFrame * 2, 2);
            vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                queryPool, currentFrame * 2);
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
            vkCmdPushConstants(frame.commandBuffer, pipelineLayout, VK_SHADER_STAGE_MESH_BIT_EXT,
                0, sizeof(camera.lightViewProjection), glm::value_ptr(camera.lightViewProjection));
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
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.clearValue.depthStencil = {1.0f, 0};
            VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
            rendering.renderArea.extent = extent;
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color;
            rendering.pDepthAttachment = &depthAttachment;
            vkCmdBeginRendering(frame.commandBuffer, &rendering);
            const VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width),
                static_cast<float>(extent.height), 0.0f, 1.0f};
            const VkRect2D scissor{{0, 0}, extent};
            vkCmdSetViewport(frame.commandBuffer, 0, 1, &viewport);
            vkCmdSetScissor(frame.commandBuffer, 0, 1, &scissor);
            vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
            vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout, 0, 1, &frame.descriptorSet, 0, nullptr);
            drawMeshTasksIndirect(frame.commandBuffer, frame.indirect.handle, 0, 1,
                sizeof(VkDrawMeshTasksIndirectCommandEXT));
            vkCmdEndRendering(frame.commandBuffer);

            imageBarrier(frame.commandBuffer, swapchainImages[imageIndex],
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_ASPECT_COLOR_BIT);
            swapchainLayouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                queryPool, currentFrame * 2 + 1);
            check(vkEndCommandBuffer(frame.commandBuffer), "vkEndCommandBuffer");

            VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            waitInfo.semaphore = frame.acquired;
            waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
            commandInfo.commandBuffer = frame.commandBuffer;
            VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            signalInfo.semaphore = presentSemaphores[imageIndex];
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            submitInfo.waitSemaphoreInfoCount = 1;
            submitInfo.pWaitSemaphoreInfos = &waitInfo;
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
        const glm::mat4& view, const glm::mat4& projection, const glm::vec3& cameraPosition)
    {
        mImpl->render(view, projection, cameraPosition);
    }

    std::uint32_t VulkanRenderer::addTexture(const SceneTexture& texture)
    {
        return mImpl->appendTexture(texture);
    }

    std::uint32_t VulkanRenderer::addMaterial(const SceneMaterial& material)
    {
        return mImpl->appendMaterial(material);
    }

    MeshRange VulkanRenderer::addMesh(PackedMesh mesh)
    {
        return mImpl->appendPackedMesh(std::move(mesh));
    }

    void VulkanRenderer::addInstance(SceneInstance instance)
    {
        mImpl->appendInstance(instance);
    }

    const RendererStats& VulkanRenderer::stats() const noexcept
    {
        return mImpl->rendererStats;
    }

    const std::string& VulkanRenderer::deviceName() const noexcept
    {
        return mImpl->gpuName;
    }
}
