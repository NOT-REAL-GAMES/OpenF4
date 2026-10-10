#include "assets/havok_packfile.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string_view>

#include <glm/gtc/matrix_transform.hpp>

namespace openf4::assets
{
    namespace
    {
        // Layouts: Smithbox HKX2 hk2014 reflection and BethesdaLibrary's FO4 packfile notes.
        constexpr std::uint32_t invalid = 0xffffffffu;
        constexpr std::uint32_t noMotion = 0x7fffffffu;

        void require(bool condition, std::string_view message)
        {
            if (!condition) throw std::runtime_error(std::string(message));
        }

        bool finite(glm::vec3 v)
        {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        }

        bool unit(glm::quat q)
        {
            return std::isfinite(glm::dot(q, q)) && std::abs(glm::dot(q, q) - 1.0f) < 0.001f;
        }

        struct Bytes
        {
            std::vector<std::byte> data;

            std::uint32_t size() const
            {
                require(data.size() < 0x7fffffffu, "Havok export exceeds the packfile size limit");
                return static_cast<std::uint32_t>(data.size());
            }

            std::uint32_t allocate(std::size_t count, std::uint32_t alignment = 16)
            {
                require(count < 0x7fffffffu && data.size() + count + alignment < 0x7fffffffu,
                    "Havok export exceeds the packfile size limit");
                data.resize((data.size() + alignment - 1) & ~(std::size_t(alignment) - 1));
                const auto offset = size();
                data.resize(data.size() + count);
                return offset;
            }

            template <class T> void put(std::uint32_t offset, T value)
            {
                require(offset <= data.size() && sizeof(T) <= data.size() - offset,
                    "Havok writer attempted an out-of-range write");
                std::memcpy(data.data() + offset, &value, sizeof(T));
            }

            void vector(std::uint32_t offset, glm::vec3 value, float w = 0)
            {
                put(offset, value.x); put(offset + 4, value.y); put(offset + 8, value.z);
                put(offset + 12, w);
            }

            void quaternion(std::uint32_t offset, glm::quat value)
            {
                vector(offset, {value.x, value.y, value.z}, value.w);
            }

            void string(std::uint32_t offset, std::string_view text)
            {
                require(offset < data.size() && text.size() < data.size() - offset,
                    "Havok writer string exceeds its buffer");
                std::memcpy(data.data() + offset, text.data(), text.size());
                data[offset + text.size()] = std::byte{};
            }
        };

        struct Bounds
        {
            glm::vec3 minimum{std::numeric_limits<float>::max()};
            glm::vec3 maximum{-std::numeric_limits<float>::max()};
            void add(glm::vec3 point)
            {
                require(finite(point) && glm::all(glm::lessThanEqual(glm::abs(point), glm::vec3(1000000))),
                    "Havok export contains a non-finite or unbounded vertex");
                minimum = glm::min(minimum, point); maximum = glm::max(maximum, point);
            }
            void add(const Bounds& other) { add(other.minimum); add(other.maximum); }
        };

        struct Writer
        {
            Bytes bytes;
            Bytes names;
            std::vector<std::array<std::uint32_t, 2>> local;
            std::vector<std::array<std::uint32_t, 3>> global;
            std::vector<std::array<std::uint32_t, 3>> reflected;
            std::map<std::string_view, std::uint32_t> classes;

            std::uint32_t className(std::string_view name, std::uint32_t signature)
            {
                if (const auto found = classes.find(name); found != classes.end()) return found->second;
                const auto offset = names.allocate(name.size() + 6, 1);
                names.put(offset, signature); names.put<std::uint8_t>(offset + 4, 9);
                names.string(offset + 5, name);
                classes.emplace(name, offset + 5);
                return offset + 5;
            }

            std::uint32_t object(std::uint32_t count, std::string_view name, std::uint32_t signature)
            {
                const auto offset = bytes.allocate(count);
                reflected.push_back({offset, 0, className(name, signature)});
                return offset;
            }

            void pointer(std::uint32_t slot, std::uint32_t value) { global.push_back({slot, 2, value}); }

            void array(std::uint32_t slot, std::uint32_t offset, std::uint32_t count)
            {
                bytes.put(slot + 8, count); bytes.put(slot + 12, count | 0x80000000u);
                if (count) local.push_back({slot, offset});
            }

            void shapeHeader(std::uint32_t offset, std::uint16_t flags, std::uint8_t keyBits,
                std::uint8_t dispatch, float radius, std::uint32_t material)
            {
                bytes.put(offset + 0x10, flags); bytes.put(offset + 0x12, keyBits);
                bytes.put(offset + 0x13, dispatch); bytes.put(offset + 0x14, radius);
                bytes.put<std::uint64_t>(offset + 0x18, material);
                if (dispatch == 2)
                {
                    bytes.put(offset + 0x30, invalid); bytes.put(offset + 0x58, invalid);
                    array(offset + 0x38, 0, 0); array(offset + 0x48, 0, 0);
                }
            }

            std::vector<std::byte> finish()
            {
                const auto nameCount = names.size();
                names.allocate(0);
                std::fill(names.data.begin() + nameCount, names.data.end(), std::byte{0xff});
                bytes.allocate(0);
                const auto localStart = bytes.size();
                for (auto entry : local) bytes.put(bytes.allocate(sizeof(entry), 1), entry);
                bytes.put(bytes.allocate(8, 1), std::array{invalid, invalid});
                const auto globalStart = bytes.size();
                for (auto entry : global) bytes.put(bytes.allocate(sizeof(entry), 1), entry);
                bytes.put(bytes.allocate(12, 1), std::array{invalid, invalid, invalid});
                const auto virtualStart = bytes.size();
                for (auto entry : reflected) bytes.put(bytes.allocate(sizeof(entry), 1), entry);
                bytes.put(bytes.allocate(12, 1), std::array{invalid, invalid, invalid});

                Bytes file;
                file.allocate(0x100);
                file.put<std::uint32_t>(0, 0x57e0e057); file.put<std::uint32_t>(4, 0x10c0c010);
                file.put<std::uint32_t>(12, 11); file.put<std::uint32_t>(16, 0x01000108);
                file.put<std::uint32_t>(20, 3); file.put<std::uint32_t>(24, 2);
                file.put<std::uint32_t>(36, classes.at("hknpPhysicsSystemData"));
                file.string(40, "hk_2014.1.0-r1"); file.put<std::uint8_t>(55, 0xff);
                file.put<std::uint32_t>(60, 21);
                const auto section = [&](std::uint32_t slot, std::string_view name, std::uint32_t start,
                    std::uint32_t l, std::uint32_t g, std::uint32_t v, std::uint32_t end) {
                    std::fill(file.data.begin() + slot, file.data.begin() + slot + 0x40, std::byte{0xff});
                    file.string(slot, name); file.put(slot + 20, start);
                    file.put(slot + 24, l); file.put(slot + 28, g); file.put(slot + 32, v);
                    file.put(slot + 36, end); file.put(slot + 40, end); file.put(slot + 44, end);
                };
                section(0x40, "__classnames__", 0x100, names.size(), names.size(), names.size(), names.size());
                section(0x80, "__types__", 0x100 + names.size(), 0, 0, 0, 0);
                section(0xc0, "__data__", 0x100 + names.size(), localStart, globalStart, virtualStart, bytes.size());
                file.data.insert(file.data.end(), names.data.begin(), names.data.end());
                file.data.insert(file.data.end(), bytes.data.begin(), bytes.data.end());
                return std::move(file.data);
            }
        };

        glm::mat4 placement(glm::mat4 transform, glm::vec3 scale, float& uniformScale,
            bool allowNonuniform = false)
        {
            require(finite(scale) && scale.x > 0 && scale.y > 0 && scale.z > 0,
                "Havok export requires positive shape scale");
            require(allowNonuniform || (std::abs(scale.x - scale.y) < 0.00001f
                && std::abs(scale.x - scale.z) < 0.00001f),
                "Havok export cannot nonuniformly scale rounded collision");
            for (int column = 0; column != 4; ++column)
                for (int row = 0; row != 4; ++row)
                    require(std::isfinite(transform[column][row]), "Havok export has an invalid shape transform");
            require(std::abs(transform[0].w) < 0.00001f && std::abs(transform[1].w) < 0.00001f
                && std::abs(transform[2].w) < 0.00001f && std::abs(transform[3].w - 1) < 0.00001f,
                "Havok export does not support projective shape transforms");
            const glm::mat3 rotation(transform);
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                    require(std::abs(glm::dot(rotation[i], rotation[j]) - (i == j ? 1.0f : 0.0f)) < 0.001f,
                        "Havok export shape transform must be a rigid rotation");
            require(glm::determinant(rotation) > 0.0f, "Havok export does not support reflected shapes");
            uniformScale = scale.x;
            return glm::scale(transform, scale);
        }

        struct Shape { std::uint32_t offset; Bounds bounds; std::uint32_t byteSize; };

        Shape convex(Writer& writer, const HavokConvexPart& part, std::uint32_t material)
        {
            const auto& mesh = part.polytope;
            require(mesh.vertices.size() >= 4 && mesh.vertices.size() <= 252
                && !mesh.indices.empty() && mesh.indices.size() % 3 == 0,
                "Havok convex export requires 4-252 vertices and a closed triangle hull");
            require(std::isfinite(mesh.convexRadius) && mesh.convexRadius >= 0,
                "Havok convex export has an invalid radius");
            float scale;
            const auto transform = placement(part.transform, part.scale, scale, mesh.convexRadius == 0);
            std::vector<glm::vec3> vertices;
            Bounds bounds;
            for (auto vertex : mesh.vertices)
            {
                vertex = glm::vec3(transform * glm::vec4(vertex, 1));
                bounds.add(vertex); vertices.push_back(vertex);
            }
            const float tolerance = std::max(1.0f, glm::length(bounds.maximum - bounds.minimum)) * 0.00001f;
            std::map<std::pair<std::uint32_t, std::uint32_t>, unsigned> edges;
            std::vector<glm::vec4> planes;
            double volume = 0;
            for (std::size_t index = 0; index < mesh.indices.size(); index += 3)
            {
                const auto a = mesh.indices[index], b = mesh.indices[index + 1], c = mesh.indices[index + 2];
                require(a < vertices.size() && b < vertices.size() && c < vertices.size()
                    && a != b && b != c && a != c, "Havok convex export has invalid triangle indices");
                auto normal = glm::cross(vertices[b] - vertices[a], vertices[c] - vertices[a]);
                require(glm::length(normal) > 1.0e-8f, "Havok convex export contains degenerate triangles");
                normal = glm::normalize(normal);
                const auto plane = glm::vec4(normal, -glm::dot(normal, vertices[a]));
                for (auto vertex : vertices)
                    require(glm::dot(normal, vertex) + plane.w <= tolerance,
                        "Havok convex export requires outward winding and convex geometry");
                planes.push_back(plane);
                volume += glm::dot(glm::dvec3(vertices[a]), glm::cross(glm::dvec3(vertices[b]), glm::dvec3(vertices[c]))) / 6;
                ++edges[{a,b}]; ++edges[{b,c}]; ++edges[{c,a}];
            }
            for (const auto& [edge, count] : edges)
                require(count == 1 && edges.contains({edge.second, edge.first}),
                    "Havok convex export requires a closed manifold hull");
            require(volume > 1.0e-12, "Havok convex export has zero volume");
            const auto vertexCount = static_cast<std::uint32_t>((vertices.size() + 3) & ~std::size_t(3));
            const auto faceCount = static_cast<std::uint32_t>(planes.size());
            const auto planeCount = (faceCount + 3) & ~3u;
            const auto vertexOffset = 0x50u;
            const auto planeOffset = vertexOffset + vertexCount * 16;
            const auto faceOffset = planeOffset + planeCount * 16;
            const auto indexOffset = faceOffset + faceCount * 4;
            require(indexOffset + mesh.indices.size() < 65535, "Havok convex shape exceeds relative-array limits");
            const auto byteSize = (indexOffset + static_cast<std::uint32_t>(mesh.indices.size()) + 15) & ~15u;
            const auto offset = writer.object(byteSize,
                "hknpConvexPolytopeShape", 0x3ce9b3e3);
            writer.shapeHeader(offset, 0x143, 0, 1, mesh.convexRadius * scale, material);
            const auto relative = [&](std::uint32_t slot, std::uint32_t target, std::uint32_t count) {
                writer.bytes.put<std::uint16_t>(offset + slot, static_cast<std::uint16_t>(count));
                writer.bytes.put<std::uint16_t>(offset + slot + 2, static_cast<std::uint16_t>(target - slot));
            };
            relative(0x30, vertexOffset, vertexCount); relative(0x40, planeOffset, planeCount);
            relative(0x44, faceOffset, faceCount); relative(0x48, indexOffset, static_cast<std::uint32_t>(mesh.indices.size()));
            for (std::uint32_t index = 0; index < vertexCount; ++index)
            {
                writer.bytes.vector(offset + vertexOffset + index * 16, vertices[std::min<std::size_t>(index, vertices.size() - 1)]);
                writer.bytes.put(offset + vertexOffset + index * 16 + 12, 0x3f000000u | std::min(index, static_cast<std::uint32_t>(vertices.size() - 1)));
            }
            for (std::uint32_t index = 0; index < planeCount; ++index)
                writer.bytes.put(offset + planeOffset + index * 16, planes[std::min(index, faceCount - 1)]);
            for (std::uint32_t index = 0; index < faceCount; ++index)
            {
                writer.bytes.put<std::uint16_t>(offset + faceOffset + index * 4, static_cast<std::uint16_t>(index * 3));
                writer.bytes.put<std::uint8_t>(offset + faceOffset + index * 4 + 2, 3);
            }
            for (std::uint32_t index = 0; index < mesh.indices.size(); ++index)
                writer.bytes.put<std::uint8_t>(offset + indexOffset + index, static_cast<std::uint8_t>(mesh.indices[index]));
            bounds.minimum -= mesh.convexRadius * scale; bounds.maximum += mesh.convexRadius * scale;
            return {offset, bounds, byteSize};
        }

        Shape capsule(Writer& writer, const HavokCapsulePart& part, std::uint32_t material)
        {
            require(finite(part.capsule.pointA) && finite(part.capsule.pointB)
                && std::isfinite(part.capsule.radius) && part.capsule.radius > 0 && part.capsule.radius <= 1000000,
                "Havok capsule export has invalid endpoints or radius");
            float scale;
            const auto transform = placement(part.transform, part.scale, scale);
            const auto a = glm::vec3(transform * glm::vec4(part.capsule.pointA, 1));
            const auto b = glm::vec3(transform * glm::vec4(part.capsule.pointB, 1));
            const auto radius = part.capsule.radius * scale;
            require(glm::length(b - a) > 1.0e-6f, "Havok capsule export requires distinct endpoints");
            const auto offset = writer.object(0x1b0, "hknpCapsuleShape", 0x60a75f4c);
            writer.shapeHeader(offset, 0x1c3, 0, 1, radius, material);
            writer.bytes.vector(offset + 0x50, a, 1); writer.bytes.vector(offset + 0x60, b, 1);
            writer.bytes.put<std::uint16_t>(offset + 0x30, 8);
            writer.bytes.put<std::uint16_t>(offset + 0x32, 0x40);
            writer.bytes.put<std::uint16_t>(offset + 0x40, 8);
            writer.bytes.put<std::uint16_t>(offset + 0x42, 0xb0);
            writer.bytes.put<std::uint16_t>(offset + 0x44, 6);
            writer.bytes.put<std::uint16_t>(offset + 0x46, 0x12c);
            writer.bytes.put<std::uint16_t>(offset + 0x48, 24);
            writer.bytes.put<std::uint16_t>(offset + 0x4a, 0x148);
            const auto minimum = glm::min(a,b) - radius / 99;
            const auto maximum = glm::max(a,b) + radius / 99;
            for (unsigned i = 0; i < 8; ++i)
            {
                const glm::vec3 point((i & 2) ? minimum.x : maximum.x,
                    (i & 4) ? maximum.y : minimum.y, (i & 1) ? maximum.z : minimum.z);
                writer.bytes.vector(offset + 0x70 + i * 16, point);
                writer.bytes.put(offset + 0x7c + i * 16, 0x3f000000u | i);
            }
            const std::array<glm::vec4, 8> planes{{{-1,0,0,minimum.x}, {0,-1,0,minimum.y},
                {0,1,0,-maximum.y}, {1,0,0,-maximum.x}, {0,0,1,-maximum.z}, {0,0,-1,minimum.z},
                {0,0,-1,minimum.z}, {0,0,-1,minimum.z}}};
            constexpr std::array<std::uint8_t,24> indices{7,6,2,3, 3,2,0,1, 7,5,4,6,
                1,0,4,5, 1,5,7,3, 2,6,4,0};
            for (unsigned i = 0; i < planes.size(); ++i) writer.bytes.put(offset + 0xf0 + i * 16, planes[i]);
            for (unsigned i = 0; i < 6; ++i)
            {
                writer.bytes.put<std::uint16_t>(offset + 0x170 + i * 4, static_cast<std::uint16_t>(i * 4));
                writer.bytes.put<std::uint8_t>(offset + 0x172 + i * 4, 4);
                writer.bytes.put<std::uint8_t>(offset + 0x173 + i * 4, 4);
            }
            writer.bytes.put(offset + 0x190, indices);
            Bounds bounds; bounds.add(a); bounds.add(b); bounds.minimum -= radius; bounds.maximum += radius;
            return {offset, bounds, 0x1b0};
        }

        struct TreeNode { Bounds bounds; std::uint32_t left = 0, right = 0, value = 0; };

        std::vector<TreeNode> tree(const std::vector<Bounds>& bounds)
        {
            std::vector<TreeNode> result;
            std::vector<std::uint32_t> order(bounds.size()); std::iota(order.begin(), order.end(), 0);
            const std::function<std::uint32_t(std::size_t, std::size_t)> build = [&](std::size_t begin, std::size_t end) {
                const auto node = static_cast<std::uint32_t>(result.size()); result.emplace_back();
                Bounds combined;
                for (auto i = begin; i < end; ++i) combined.add(bounds[order[i]]);
                result[node].bounds = combined;
                if (end - begin == 1) { result[node].value = order[begin]; return node; }
                const auto extent = combined.maximum - combined.minimum;
                const int axis = extent.x >= extent.y && extent.x >= extent.z ? 0 : (extent.y >= extent.z ? 1 : 2);
                const auto middle = begin + (end - begin) / 2;
                std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
                    [&](auto a, auto b) { return bounds[a].minimum[axis] + bounds[a].maximum[axis]
                        < bounds[b].minimum[axis] + bounds[b].maximum[axis]; });
                const auto left = build(begin, middle), right = build(middle, end);
                result[node].left = left; result[node].right = right;
                return node;
            };
            build(0, order.size()); return result;
        }

        std::uint32_t compressedTree(Writer& writer, const std::vector<TreeNode>& nodes,
            bool top, std::vector<std::uint16_t>* leaves = nullptr)
        {
            const auto stride = top ? 5u : 4u;
            const auto start = writer.bytes.allocate(nodes.size() * stride);
            const std::function<void(std::uint32_t, Bounds)> emit = [&](std::uint32_t index, Bounds parent) {
                const auto& node = nodes[index];
                Bounds decoded = parent;
                for (int axis = 0; axis < 3; ++axis)
                {
                    const float span = parent.maximum[axis] - parent.minimum[axis];
                    const auto quantize = [&](float delta) {
                        if (span <= 0) return 0;
                        return std::clamp(static_cast<int>(std::sqrt(std::max(0.0f, delta / span) * 226.0f)), 0, 15);
                    };
                    int lo = index ? quantize(node.bounds.minimum[axis] - parent.minimum[axis]) : 0;
                    int hi = index ? quantize(parent.maximum[axis] - node.bounds.maximum[axis]) : 0;
                    while (lo && parent.minimum[axis] + float(lo * lo) / 226 * span > node.bounds.minimum[axis]) --lo;
                    while (hi && parent.maximum[axis] - float(hi * hi) / 226 * span < node.bounds.maximum[axis]) --hi;
                    decoded.minimum[axis] += float(lo * lo) / 226 * span;
                    decoded.maximum[axis] -= float(hi * hi) / 226 * span;
                    writer.bytes.put<std::uint8_t>(start + index * stride + axis, static_cast<std::uint8_t>(lo << 4 | hi));
                }
                std::uint32_t value = node.value * (top ? 1 : 2);
                if (node.left)
                {
                    const auto delta = node.right - index;
                    require(delta % 2 == 0 && delta < (top ? 65536u : 256u), "Havok BVH link exceeds its codec");
                    value = top ? 0x8000u | delta / 2 : delta | 1;
                    emit(node.left, decoded); emit(node.right, decoded);
                }
                else if (leaves) (*leaves)[node.value] = static_cast<std::uint16_t>(index);
                if (top) writer.bytes.put<std::uint8_t>(start + index * stride + 3, static_cast<std::uint8_t>(value >> 8));
                writer.bytes.put<std::uint8_t>(start + index * stride + stride - 1, static_cast<std::uint8_t>(value));
            };
            emit(0, nodes.front().bounds); return start;
        }

        Shape compound(Writer& writer, const std::vector<Shape>& children, std::uint32_t material)
        {
            require(children.size() > 1 && children.size() <= 4096, "Havok compound export supports 2-4096 parts");
            const auto count = static_cast<std::uint32_t>(children.size());
            const auto offset = writer.object(0xd0, "hknpDynamicCompoundShape", 0x4620d11c);
            writer.shapeHeader(offset, 4, static_cast<std::uint8_t>(std::bit_width(count)), 2, 0, material);
            writer.bytes.put(offset + 0x70, invalid); writer.bytes.put(offset + 0xa0, 1u);
            const auto instances = writer.bytes.allocate(children.size() * 0x80);
            writer.array(offset + 0x60, instances, count);
            const auto data = writer.object(0x40, "hknpDynamicCompoundShapeData", 0xf33dc3cc);
            writer.pointer(offset + 0xc0, data);
            std::vector<Bounds> bounds; for (auto child : children) bounds.push_back(child.bounds);
            const auto nodes = tree(bounds);
            const auto storage = writer.bytes.allocate((nodes.size() + 2) * 32);
            writer.array(data + 0x10, storage, static_cast<std::uint32_t>(nodes.size() + 2));
            writer.bytes.put<std::uint16_t>(data + 0x20, static_cast<std::uint16_t>(nodes.size() + 1));
            writer.bytes.put(data + 0x28, count); writer.bytes.put<std::uint16_t>(data + 0x30, 1);
            std::vector<std::uint32_t> parents(nodes.size());
            std::vector<std::uint32_t> leafIndices(children.size());
            for (std::uint32_t i = 0; i < nodes.size(); ++i)
            {
                const auto& node = nodes[i]; const auto destination = storage + (i + 1) * 32;
                writer.bytes.vector(destination, node.bounds.minimum);
                writer.bytes.put(destination + 12, 0x3f000000u | parents[i]);
                writer.bytes.vector(destination + 16, node.bounds.maximum);
                if (node.left)
                {
                    parents[node.left] = i + 1; parents[node.right] = i + 1;
                    writer.bytes.put<std::uint16_t>(destination + 28, static_cast<std::uint16_t>(node.left + 1));
                    writer.bytes.put<std::uint16_t>(destination + 30, static_cast<std::uint16_t>(node.right + 1));
                }
                else
                {
                    writer.bytes.put<std::uint16_t>(destination + 30, static_cast<std::uint16_t>(node.value));
                    leafIndices[node.value] = i + 1;
                }
            }
            for (std::uint32_t i = 0; i < count; ++i)
            {
                const auto instance = instances + i * 0x80;
                writer.bytes.put(instance, glm::mat4(1));
                writer.bytes.put(instance + 0x0c, 0x3f000040u);
                writer.bytes.put(instance + 0x2c, 0x3f000000u | children[i].byteSize);
                writer.bytes.put(instance + 0x3c, 0x3f000000u | leafIndices[i]);
                writer.bytes.vector(instance + 0x40, glm::vec3(1), 1);
                writer.pointer(instance + 0x50, children[i].offset);
                writer.bytes.put(instance + 0x58, invalid);
            }
            writer.bytes.vector(offset + 0x80, nodes.front().bounds.minimum);
            writer.bytes.put(offset + 0x8c, 0x3f000001u);
            writer.bytes.vector(offset + 0x90, nodes.front().bounds.maximum);
            return {offset, nodes.front().bounds, 0xd0};
        }

        Shape mesh(Writer& writer, const HavokTriangleMesh& source, std::uint32_t material, std::uint32_t filter)
        {
            require(!source.vertices.empty() && !source.indices.empty() && source.indices.size() % 3 == 0,
                "Havok mesh export requires triangles");
            require(std::ranges::all_of(source.primitiveDataRuns, [](const auto& run) { return run.value == 0xffff; }),
                "Rebuilding imported per-primitive Havok materials is not supported; preserve the original collision");
            struct Section
            {
                Bounds bounds;
                std::vector<glm::vec3> vertices;
                std::vector<std::array<std::uint8_t, 4>> triangles;
                std::vector<std::uint32_t> packed;
                std::vector<Bounds> triangleBounds;
                glm::vec3 scale{};
            };
            std::vector<Section> sections(1);
            std::map<std::uint32_t, std::uint8_t> remap;
            for (std::size_t i = 0; i < source.indices.size(); i += 3)
            {
                std::array<std::uint32_t,3> triangle{source.indices[i], source.indices[i + 1], source.indices[i + 2]};
                for (auto index : triangle) require(index < source.vertices.size(), "Havok mesh export has an invalid vertex index");
                require(glm::length(glm::cross(source.vertices[triangle[1]] - source.vertices[triangle[0]],
                    source.vertices[triangle[2]] - source.vertices[triangle[0]])) > 1.0e-8f,
                    "Havok mesh export contains a degenerate triangle");
                std::size_t added = 0; for (auto index : triangle) added += !remap.contains(index);
                if (sections.back().triangles.size() == 127 || remap.size() + added > 255)
                { sections.emplace_back(); remap.clear(); }
                auto& section = sections.back();
                std::array<std::uint8_t,4> encoded{};
                for (std::size_t j = 0; j < 3; ++j)
                {
                    const auto [entry, inserted] = remap.emplace(triangle[j], static_cast<std::uint8_t>(section.vertices.size()));
                    if (inserted) { section.vertices.push_back(source.vertices[triangle[j]]); section.bounds.add(source.vertices[triangle[j]]); }
                    encoded[j] = entry->second;
                }
                encoded[3] = encoded[2]; section.triangles.push_back(encoded);
            }
            require(sections.size() <= 32768 && source.indices.size() / 3 <= 65535,
                "Havok mesh export exceeds the section/primitive addressing limit");
            std::vector<Bounds> sectionBounds;
            Bounds bounds;
            for (auto& section : sections)
            {
                section.scale = (section.bounds.maximum - section.bounds.minimum) / glm::vec3(2047,2047,1023);
                for (auto& vertex : section.vertices)
                {
                    glm::uvec3 quantized{};
                    for (int axis = 0; axis != 3; ++axis)
                        if (section.scale[axis] > 0) quantized[axis] = static_cast<unsigned>(std::lround(
                            (vertex[axis] - section.bounds.minimum[axis]) / section.scale[axis]));
                    quantized = glm::min(quantized, glm::uvec3(2047,2047,1023));
                    section.packed.push_back(quantized.x | quantized.y << 11 | quantized.z << 22);
                    vertex = section.bounds.minimum + glm::vec3(quantized) * section.scale;
                    section.bounds.add(vertex);
                }
                for (auto triangle : section.triangles)
                {
                    Bounds triangleBounds;
                    for (int i = 0; i < 3; ++i) triangleBounds.add(section.vertices[triangle[i]]);
                    require(glm::length(glm::cross(section.vertices[triangle[1]] - section.vertices[triangle[0]],
                        section.vertices[triangle[2]] - section.vertices[triangle[0]])) > 1.0e-8f,
                        "Havok mesh quantization collapsed a triangle; split or rescale the source mesh");
                    section.triangleBounds.push_back(triangleBounds);
                }
                sectionBounds.push_back(section.bounds); bounds.add(section.bounds);
            }
            const auto last = static_cast<std::uint32_t>(sections.size() - 1);
            const auto maxKey = last << 8 | static_cast<std::uint32_t>(sections.back().triangles.size() - 1) << 1;
            const auto bits = std::bit_width(maxKey + 1);
            const auto offset = writer.object(0xc0, "hknpCompressedMeshShape", 0x5f60d536);
            writer.shapeHeader(offset, 0x204, static_cast<std::uint8_t>(bits), 2, 0, material);
            const auto properties = writer.object(0x20, "hkRefCountedProperties", 0x7c574867);
            writer.pointer(offset + 0x20, properties); writer.array(properties, properties + 0x10, 1);
            writer.bytes.put<std::uint16_t>(properties + 0x18, 0xf601);
            const auto materialProperties = writer.object(0x40, "hknpBSMaterialProperties", 0xa3e47a9a);
            writer.pointer(properties + 0x10, materialProperties);
            writer.array(materialProperties + 0x10, materialProperties + 0x20, 1);
            writer.bytes.put(materialProperties + 0x30, filter);
            writer.bytes.put(materialProperties + 0x34, material);
            const auto data = writer.object(0xd0, "hknpCompressedMeshShapeData", 0xa2bdfc59);
            writer.pointer(offset + 0x60, data);
            const auto topNodes = tree(sectionBounds);
            std::vector<std::uint16_t> leaves(sections.size());
            const auto nodes = compressedTree(writer, topNodes, true, &leaves);
            writer.array(data + 0x10, nodes, static_cast<std::uint32_t>(topNodes.size()));
            writer.bytes.vector(data + 0x20, bounds.minimum); writer.bytes.vector(data + 0x30, bounds.maximum);
            writer.bytes.put(data + 0x40, static_cast<std::uint32_t>(source.indices.size() / 3));
            writer.bytes.put(data + 0x44, bits); writer.bytes.put(data + 0x48, maxKey);
            const auto table = writer.bytes.allocate(sections.size() * 0x60);
            writer.array(data + 0x50, table, static_cast<std::uint32_t>(sections.size()));
            const auto primitiveOffset = writer.bytes.allocate(source.indices.size() / 3 * 4);
            writer.array(data + 0x60, primitiveOffset, static_cast<std::uint32_t>(source.indices.size() / 3));
            std::size_t vertexCount = 0; for (const auto& section : sections) vertexCount += section.vertices.size();
            const auto packedOffset = writer.bytes.allocate(vertexCount * 4);
            writer.array(data + 0x80, packedOffset, static_cast<std::uint32_t>(vertexCount));
            writer.array(data + 0x70, 0, 0); writer.array(data + 0x90, 0, 0);
            const auto runs = writer.bytes.allocate(sections.size() * 4);
            writer.array(data + 0xa0, runs, static_cast<std::uint32_t>(sections.size()));
            std::uint32_t firstVertex = 0, firstTriangle = 0;
            for (std::uint32_t i = 0; i < sections.size(); ++i)
            {
                const auto& section = sections[i]; const auto entry = table + i * 0x60;
                const auto sectionTree = tree(section.triangleBounds);
                writer.array(entry, compressedTree(writer, sectionTree, false), static_cast<std::uint32_t>(sectionTree.size()));
                writer.bytes.vector(entry + 0x10, section.bounds.minimum); writer.bytes.vector(entry + 0x20, section.bounds.maximum);
                for (unsigned axis = 0; axis != 3; ++axis)
                { writer.bytes.put(entry + 0x30 + axis * 4, section.bounds.minimum[axis]); writer.bytes.put(entry + 0x3c + axis * 4, section.scale[axis]); }
                writer.bytes.put(entry + 0x48, firstVertex);
                writer.bytes.put(entry + 0x4c, static_cast<std::uint32_t>(section.vertices.size()));
                writer.bytes.put(entry + 0x50, firstTriangle << 8 | static_cast<std::uint32_t>(section.triangles.size()));
                writer.bytes.put(entry + 0x54, i << 8 | 1u);
                writer.bytes.put<std::uint8_t>(entry + 0x58, static_cast<std::uint8_t>(section.vertices.size()));
                writer.bytes.put(entry + 0x5a, leaves[i]);
                for (auto triangle : section.triangles) writer.bytes.put(primitiveOffset + firstTriangle++ * 4, triangle);
                for (auto vertex : section.packed) writer.bytes.put(packedOffset + firstVertex++ * 4, vertex);
                writer.bytes.put<std::uint16_t>(runs + i * 4, 0);
                writer.bytes.put<std::uint8_t>(runs + i * 4 + 3, static_cast<std::uint8_t>(section.triangles.size()));
            }
            const auto simd = writer.bytes.allocate(224);
            writer.array(data + 0xb8, simd, 2);
            for (std::uint32_t node = 0; node < 2; ++node)
                for (std::uint32_t axis = 0; axis < 3; ++axis)
                    for (std::uint32_t lane = 0; lane < 4; ++lane)
                    {
                        writer.bytes.put(simd + node * 112 + axis * 32 + lane * 4, 0x7f7fffeeu);
                        writer.bytes.put(simd + node * 112 + axis * 32 + 16 + lane * 4, 0xff7fffeeu);
                    }
            for (const auto [slot, count] : {std::pair{0x68u, (maxKey + 2) / 2}, std::pair{0x80u, maxKey + 1}})
            {
                const auto words = (count + 31) / 32;
                writer.array(offset + slot, writer.bytes.allocate(words * 4), words);
                writer.bytes.put(offset + slot + 16, count);
            }
            return {offset, bounds, 0xc0};
        }
    }

    std::vector<std::byte> serializeHavokPhysicsSystem(const HavokPhysicsSystem& system)
    {
        require(system.nativeRewriteBlockReason.empty(), system.nativeRewriteBlockReason);
        require(!system.bodies.empty() && system.bodies.size() < 65535, "Havok export requires 1-65534 bodies");
        Writer writer;
        writer.className("hkClass", 0x33d42383); writer.className("hkClassMember", 0xb0efa719);
        writer.className("hkClassEnum", 0x8a3609cf); writer.className("hkClassEnumItem", 0xce6f8a6c);
        const auto root = writer.object(0x80, "hknpPhysicsSystemData", 0xb857718b);
        const auto count = static_cast<std::uint32_t>(system.bodies.size());
        const auto dynamics = static_cast<std::uint32_t>(std::ranges::count_if(system.bodies,
            [](const auto& body) { return body.dynamic; }));
        const auto materials = writer.bytes.allocate(count * 0x50);
        const auto properties = writer.bytes.allocate(dynamics * 0x40);
        const auto motions = writer.bytes.allocate(dynamics * 0x70);
        const auto bodies = writer.bytes.allocate(count * 0x60);
        const auto references = writer.bytes.allocate(count * 8);
        writer.array(root + 0x10, materials, count); writer.array(root + 0x20, properties, dynamics);
        writer.array(root + 0x30, motions, dynamics); writer.array(root + 0x40, bodies, count);
        writer.array(root + 0x50, 0, 0); writer.array(root + 0x60, references, count);
        unsigned motionIndex = 0;
        for (unsigned index = 0; index < count; ++index)
        {
            const auto& body = system.bodies[index];
            require(body.bodyIndex == index, "Havok export body indices must match the NIF body array order");
            require(finite(body.position) && glm::all(glm::lessThanEqual(glm::abs(body.position), glm::vec3(1000000)))
                && unit(body.rotation), "Havok export body transform is invalid");
            require(std::isfinite(body.friction) && body.friction >= 0 && body.friction <= 255
                && std::isfinite(body.restitution) && body.restitution >= 0 && body.restitution <= 1,
                "Havok export contact properties are invalid");
            require(body.parts.size() + body.capsules.size() + bool(body.triangleMesh) != 0,
                "Havok export requires a collision shape");
            require(!body.dynamic || !body.triangleMesh, "Dynamic Havok triangle mesh export is not supported");
            std::vector<Shape> shapes;
            for (const auto& part : body.parts) shapes.push_back(convex(writer, part, body.materialId));
            for (const auto& part : body.capsules) shapes.push_back(capsule(writer, part, body.materialId));
            if (body.triangleMesh) shapes.push_back(mesh(writer, *body.triangleMesh, body.materialId, body.collisionFilterInfo));
            const auto shape = shapes.size() == 1 ? shapes.front() : compound(writer, shapes, body.materialId);
            const auto destination = bodies + index * 0x60;
            writer.pointer(destination, shape.offset); writer.pointer(references + index * 8, shape.offset);
            writer.bytes.put(destination + 8, noMotion);
            writer.bytes.put(destination + 12, body.dynamic ? motionIndex : noMotion);
            writer.bytes.put<std::uint8_t>(destination + 0x10, 0xff);
            writer.bytes.put<std::uint16_t>(destination + 0x12, static_cast<std::uint16_t>(index));
            writer.bytes.put(destination + 0x14, body.collisionFilterInfo);
            writer.bytes.vector(destination + 0x30, body.position); writer.bytes.quaternion(destination + 0x40, body.rotation);
            const auto material = materials + index * 0x50;
            writer.bytes.put<std::uint8_t>(material + 0x11, 0xff);
            const auto half = [](float value) { return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16); };
            writer.bytes.put(material + 0x12, half(body.friction)); writer.bytes.put(material + 0x14, half(body.friction));
            writer.bytes.put(material + 0x16, half(body.restitution));
            writer.bytes.put<std::uint8_t>(material + 0x18, 1); writer.bytes.put<std::uint8_t>(material + 0x19, 2);
            writer.bytes.put(material + 0x1a, half(0.05f)); writer.bytes.put(material + 0x1c, 0x7f7fffeeu);
            writer.bytes.put(material + 0x20, 1.0f); writer.bytes.put(material + 0x26, half(1.0f));
            writer.bytes.put(material + 0x38, half(5.0f));
            if (body.dynamic)
            {
                require(std::isfinite(body.mass) && body.mass > 0 && finite(body.inertiaDiagonal)
                    && glm::all(glm::greaterThan(body.inertiaDiagonal, glm::vec3(0)))
                    && std::isfinite(1.0f / body.mass) && finite(glm::vec3(1) / body.inertiaDiagonal)
                    && finite(body.centerOfMass) && unit(body.inertiaOrientation),
                    "Dynamic Havok export requires positive mass and computed principal inertia");
                for (float value : {body.linearDamping, body.angularDamping, body.gravityFactor,
                    body.maxLinearVelocity, body.maxAngularVelocity})
                    require(std::isfinite(value) && value >= 0, "Havok export motion properties are invalid");
                require(body.motionProfile == HavokMotionProfile::fallout4
                    || body.motionProfile == HavokMotionProfile::legacyOpenF4, "Unsupported Havok motion profile");
                const bool legacy = body.motionProfile == HavokMotionProfile::legacyOpenF4;
                const auto property = properties + motionIndex * 0x40;
                writer.bytes.put(property + 4, (legacy ? 2u : 0u) | (body.gravityFactor != 1 ? 0x20000000u : 0));
                writer.bytes.put(property + 8, body.gravityFactor); writer.bytes.put(property + 12, 1.0f);
                writer.bytes.put(property + 16, body.maxLinearVelocity); writer.bytes.put(property + 20, body.maxAngularVelocity);
                writer.bytes.put(property + 24, body.linearDamping); writer.bytes.put(property + 28, body.angularDamping);
                writer.bytes.put(property + 32, 0.17f);
                writer.bytes.put(property + 36, legacy ? 0x3efb22d1u : 0x3efb22d2u);
                writer.bytes.put(property + 40, legacy ? 0x3b23d70au : 0x3b23d70bu);
                writer.bytes.put(property + 44, legacy ? 0x3b23d70au : 0x3b23d70bu);
                writer.bytes.put(property + 48, 1.0f);
                if (!legacy)
                {
                    // Native FO4 motion/deactivation defaults verified by the traffic-cone comparison.
                    writer.bytes.put(property + 52, 0x999b67ffu);
                    writer.bytes.put(property + 56, 0x06757304u);
                    writer.bytes.put(property + 60, 0x00000073u);
                }
                const auto motion = motions + motionIndex * 0x70;
                writer.bytes.put<std::uint16_t>(motion, static_cast<std::uint16_t>(motionIndex));
                writer.bytes.put<std::uint8_t>(motion + 2, 1);
                writer.bytes.put(motion + 4, 1.0f / body.mass); writer.bytes.put(motion + 8, 1.0f);
                writer.bytes.put(motion + 12, 0x5f7ffff0u); writer.bytes.put(motion + 16, 0x5f7ffff0u);
                writer.bytes.vector(motion + 0x20, glm::vec3(1) / body.inertiaDiagonal);
                writer.bytes.vector(motion + 0x30, body.position + body.rotation * body.centerOfMass);
                writer.bytes.quaternion(motion + 0x40, body.rotation * body.inertiaOrientation);
                ++motionIndex;
            }
        }
        return writer.finish();
    }
}
