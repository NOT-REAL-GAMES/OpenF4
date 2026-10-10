#include "assets/fo4_asset_package.hpp"
#include "assets/fo4_export.hpp"
#include "assets/esm_editor.hpp"
#include "assets/coordinates.hpp"
#include "assets/native_model_export.hpp"

#include <ExtraData.hpp>
#include <NifFile.hpp>
#include <Animation.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>

namespace
{
    using namespace openf4::assets;
    void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
    template<class F> void rejects(F&& operation, const char* message)
    {
        try { operation(); } catch (const std::exception&) { return; }
        throw std::runtime_error(message);
    }
    std::vector<std::byte> read(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate); require(bool(stream), "Cannot read fixture");
        std::vector<std::byte> bytes(static_cast<std::size_t>(stream.tellg())); stream.seekg(0);
        require(bool(stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))), "Cannot read fixture bytes");
        return bytes;
    }
    void write(const std::filesystem::path& path, std::span<const std::byte> bytes)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        require(bool(stream), "Cannot write fixture");
    }
    void archive(const std::filesystem::path& path, const std::map<std::string, std::vector<std::byte>>& files)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary);
        const auto number = [&]<class T>(T value) { output.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
        std::uint64_t names = 24 + files.size() * 36;
        for (const auto& [name, bytes] : files) names += bytes.size();
        number(0x58445442u); number(1u); number(0x4c524e47u); number(static_cast<std::uint32_t>(files.size())); number(names);
        std::uint64_t offset = 24 + files.size() * 36;
        for (const auto& [name, bytes] : files)
        {
            number(0u); number(0u); number(0u); number(0u); number(offset);
            number(0u); number(static_cast<std::uint32_t>(bytes.size())); number(0xbaadf00du); offset += bytes.size();
        }
        for (const auto& [name, bytes] : files) output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        for (const auto& [name, bytes] : files)
        {
            number(static_cast<std::uint16_t>(name.size())); output.write(name.data(), static_cast<std::streamsize>(name.size()));
        }
        require(bool(output), "Cannot write archive fixture");
    }
    constexpr std::array<std::uint32_t, 36> triangles{0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
        1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7};
    const std::array<glm::vec3, 8> vertices{{{-140,35,-70},{70,35,-70},{70,210,-70},{-140,210,-70},
        {-140,35,140},{70,35,140},{70,210,140},{-140,210,140}}};
    ModelGeometry geometry()
    {
        ModelGeometry result; auto& surface = result.surfaces.emplace_back();
        for (const auto p : vertices)
        {
            openf4::render::SourceVertex vertex; vertex.position = p; vertex.normal = {0,1,0};
            surface.vertices.push_back(vertex);
        }
        surface.indices.assign(triangles.begin(), triangles.end());
        surface.metallicFactor = 0; surface.roughnessFactor = 1;
        return result;
    }
    HavokPhysicsSystem collision(bool dynamic)
    {
        HavokPhysicsSystem result; auto& body = result.bodies.emplace_back();
        body.dynamic = dynamic; body.mass = dynamic ? 3 : 0;
        if (dynamic) body.inertiaDiagonal = {2,3,4};
        auto& hull = body.parts.emplace_back().polytope;
        for (const auto p : vertices) hull.vertices.push_back(engineToHavokPosition(p));
        hull.indices.assign(triangles.begin(), triangles.end());
        for (std::size_t i = 0; i < hull.indices.size(); i += 3) std::swap(hull.indices[i + 1], hull.indices[i + 2]);
        body.position = {4, 0, 0}; // Collision extends beyond render bounds; arrival must account for it.
        return result;
    }
    ESMModel model(const std::filesystem::path& world, const NativeModelPackage& package, bool dynamic = false)
    {
        ESMModel result; result.formId = 0x01000800; result.path = package.modelPath;
        result.sourceEsm = world.string(); result.kind = dynamic ? ESMModelKind::movableStatic : ESMModelKind::staticObject;
        return result;
    }
    void run(const std::filesystem::path& root)
    {
        const auto world = root / "owner.esm";
        const auto staticBody = collision(false);
        const auto package = exportNativeModelAsset(world, 0x01000800, geometry(), &staticBody);
        const auto sourceModel = model(world, package);
        const auto before = read(package.assetRoot / package.modelPath);
        const auto inspected = inspectFo4ModelAsset(sourceModel);
        require(inspected.bounds == std::array<std::int16_t, 6>{-140,-70,35,70,140,210}, "OBND axes, units, or order are wrong");
        require(inspected.bethesdaMaximum.x > 349 && inspected.bethesdaMinimum.x == -140,
            "Arrival clearance did not union render and authored collision bounds");
        require(!inspected.modelPath.starts_with("meshes") && inspected.modelPath.find('/') == std::string::npos,
            "MODL path is not meshes-relative Bethesda syntax");
        require(inspected.collision.size() == 1 && inspected.dependencies.size() >= 4 && inspected.stockDependencies.empty(),
            "Preflight omitted native dependencies/collision or misidentified provenance");
        require(!std::filesystem::exists(root / "Data"), "Preflight wrote package files");
        const auto staged = stageFo4ModelAsset(sourceModel, root / "Data");
        for (const auto& dependency : staged.dependencies)
            require(read(root / "Data" / dependency.dataPath) == read(dependency.resolvedPath), "Staged native bytes changed");
        (void)stageFo4ModelAsset(sourceModel, root / "Data");
        require(read(package.assetRoot / package.modelPath) == before, "Native source was rewritten");

        auto inherited = sourceModel; inherited.sourceEsm = (root / "archive-owner.esm").string();
        rejects([&] { (void)inspectFo4ModelAsset(inherited, package.assetRoot); },
            "Arbitrary loose Data was mislabeled as a Fallout4.esm dependency");
        const auto archivePath = root / "stock-data/Fallout4 - Meshes.ba2";
        std::map<std::string, std::vector<std::byte>> archiveFiles;
        for (const auto& dependency : staged.dependencies) archiveFiles.emplace(dependency.dataPath, read(dependency.resolvedPath));
        archive(archivePath, archiveFiles);
        const auto stock = stageFo4ModelAsset(inherited, root / "stock-output", archivePath.parent_path());
        require(stock.copiedFiles.empty() && stock.stockDependencies.size() == archiveFiles.size()
            && !std::filesystem::exists(root / "stock-output"), "Stock archive assets were redistributed");
        for (const auto& dependency : stock.dependencies)
            require(dependency.source == AssetSource::archive && std::filesystem::equivalent(dependency.resolvedPath, archivePath),
                "Stock dependency did not retain its exact source archive for fingerprinting");

#ifdef _WIN32
        // A physical archive fingerprint is not the identity of one logical texture payload.
        const auto proofWorld = root / "payload-proof.esm";
        saveEditableLandscape(proofWorld, createEditableLandscape());
        EsmDocument proof(proofWorld);
        std::vector<ESMModel> proofModels;
        std::vector<ESMReference> proofReferences;
        for (int i = 0; i < 2; ++i)
        {
            auto base = sourceModel;
            base.formId = proof.allocateFormId(); base.editorId = "PayloadBase" + std::to_string(i);
            base.sourceEsm = proofWorld.string(); proofModels.push_back(base);
            auto& ref = proofReferences.emplace_back();
            ref.placement.formId = proof.allocateFormId(); ref.placement.baseObject = base.formId;
            ref.placement.cell = {0, 0}; ref.placement.scale = 1;
        }
        proof.setModels(proofModels); proof.setReferences(proofReferences); proof.save();
        const auto logicalTexture = std::ranges::find_if(stock.dependencies, [](const auto& d) { return d.dataPath.ends_with(".dds"); });
        require(logicalTexture != stock.dependencies.end(), "Stock fixture has no logical texture");
        const auto projectTexture = AssetResolver::projectAssetRoot(proofWorld) / logicalTexture->dataPath;
        const auto& payload = archiveFiles.at(logicalTexture->dataPath);
        Fo4ExportOptions proofOptions; proofOptions.dataDirectory = archivePath.parent_path();
        proofOptions.progress = [&](const auto& progress) {
            if (progress.phase == "Inspecting native assets" && progress.completed == 1) write(projectTexture, payload);
        };
        const auto equalPayload = inspectFo4Export(proofWorld, proofOptions);
        require(equalPayload.canWriteWorldProof(), "Identical stock/project logical asset bytes conflicted with their container fingerprints");
        std::filesystem::remove(projectTexture);
        auto differentPayload = payload; differentPayload.back() ^= std::byte{1};
        proofOptions.progress = [&](const auto& progress) {
            if (progress.phase == "Inspecting native assets" && progress.completed == 1) write(projectTexture, differentPayload);
        };
        const auto different = inspectFo4Export(proofWorld, proofOptions);
        require(!different.canWriteWorldProof() && different.blockers.front().find("Conflicting Data dependency") != std::string::npos,
            "Different stock/project bytes at the same logical path were accepted");
        write(projectTexture, payload);
        proofOptions.progress = [&](const auto& progress) {
            if (progress.phase == "Inspecting native assets" && progress.completed == 2) write(projectTexture, differentPayload);
        };
        const auto changed = inspectFo4Export(proofWorld, proofOptions);
        require(!changed.canWriteWorldProof() && changed.blockers.front().find("Export source changed") != std::string::npos,
            "Changing native asset bytes after inspection bypassed source fingerprint verification");
        write(projectTexture, payload);
#endif

        auto invalid = sourceModel; invalid.path = "meshes/../outside.nif";
        rejects([&] { (void)stageFo4ModelAsset(invalid, root / "bad-path"); }, "Traversal path was accepted");
        invalid.path = "C:\\outside.nif";
        rejects([&] { (void)inspectFo4ModelAsset(invalid); }, "Absolute model path was accepted");
        invalid.path = "models/source.glb";
        rejects([&] { (void)inspectFo4ModelAsset(invalid); }, "Imported model silently acquired generated collision");
        invalid = sourceModel; invalid.sourceEsm.clear();
        rejects([&] { (void)inspectFo4ModelAsset(invalid); }, "Missing source provenance accepted");
        require(!std::filesystem::exists(root / "bad-path"), "Failed path preflight wrote files");

        const auto texture = std::ranges::find_if(staged.dependencies, [](const auto& d) { return d.dataPath.ends_with(".dds"); });
        require(texture != staged.dependencies.end(), "Native fixture has no texture");
        const auto textureBytes = read(texture->resolvedPath);
        const auto absent = texture->resolvedPath.string() + ".held";
        std::filesystem::rename(texture->resolvedPath, absent);
        rejects([&] { (void)stageFo4ModelAsset(sourceModel, root / "missing"); }, "Missing native texture did not block package");
        require(!std::filesystem::exists(root / "missing"), "Missing dependency published partial files");
        std::filesystem::rename(absent, texture->resolvedPath);
        const std::array<std::byte, 3> corrupt{std::byte{1},std::byte{2},std::byte{3}};
        write(texture->resolvedPath, corrupt);
        rejects([&] { (void)inspectFo4ModelAsset(sourceModel); }, "Malformed texture was accepted");
        write(texture->resolvedPath, textureBytes);

        auto uppercase = texture->dataPath;
        std::ranges::transform(uppercase, uppercase.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        write(root / "conflict" / uppercase, corrupt);
        rejects([&] { (void)stageFo4ModelAsset(sourceModel, root / "conflict"); }, "Case-folded dependency conflict was overwritten");
        require(read(root / "conflict" / uppercase) == std::vector<std::byte>(corrupt.begin(), corrupt.end())
            && !std::filesystem::exists(root / "conflict" / package.modelPath), "Conflict preflight wrote or replaced files");

        const auto noCollision = exportNativeModelAsset(world, 0x01000801, geometry());
        rejects([&] { (void)inspectFo4ModelAsset(model(world, noCollision)); }, "Missing STAT collision silently became playable");
        auto movable = sourceModel; movable.kind = ESMModelKind::movableStatic;
        rejects([&] { (void)inspectFo4ModelAsset(movable); }, "Static collision was accepted for MSTT");
        const auto dynamicBody = collision(true);
        const auto dynamicPackage = exportNativeModelAsset(world, 0x01000802, geometry(), &dynamicBody);
        require(inspectFo4ModelAsset(model(world, dynamicPackage, true)).collision[0].bodies[0].mass == 3,
            "Valid MSTT collision did not pass preflight");
        rejects([&] { (void)inspectFo4ModelAsset(model(world, dynamicPackage)); }, "Dynamic native collision was silently coerced to STAT");

        // Populate the optional specular texture slot, which the editor material projection does not expose.
        const auto material = std::ranges::find_if(staged.dependencies, [](const auto& d) { return d.dataPath.ends_with(".bgsm"); });
        require(material != staged.dependencies.end(), "Native fixture has no material");
        auto bytes = read(material->resolvedPath);
        std::size_t offset = 63;
        for (int slot = 0; slot < 2; ++slot)
        {
            std::uint32_t length; std::memcpy(&length, bytes.data() + offset, 4); offset += 4 + length;
        }
        std::uint32_t previous; std::memcpy(&previous, bytes.data() + offset, 4);
        const std::string extra = "openf4/optional-specular.dds";
        const auto size = static_cast<std::uint32_t>(extra.size() + 1);
        std::vector<std::byte> encoded(4 + size); std::memcpy(encoded.data(), &size, 4);
        std::memcpy(encoded.data() + 4, extra.c_str(), size);
        bytes.erase(bytes.begin() + offset, bytes.begin() + offset + 4 + previous);
        bytes.insert(bytes.begin() + offset, encoded.begin(), encoded.end()); write(material->resolvedPath, bytes);
        write(package.assetRoot / "textures" / extra, textureBytes);
        const auto optional = stageFo4ModelAsset(sourceModel, root / "optional");
        require(read(root / "optional/textures" / extra) == textureBytes
            && read(root / "optional" / material->dataPath) == bytes, "Optional material dependencies or unknown bytes were lost");
        require(optional.dependencies.size() == staged.dependencies.size() + 1, "Optional texture missing from source fingerprint list");

#ifdef _WIN32
        const auto proofAssets = AssetResolver::projectAssetRoot(proofWorld);
        const auto projectMaterial = proofAssets / material->dataPath;
        bool addedDependency = false;
        proofOptions.progress = {};
        proofOptions.cancelled = [&] {
            if (!addedDependency)
                for (const auto& entry : std::filesystem::directory_iterator(root))
                    if (entry.path().filename().string().starts_with("closure-race.incomplete-")
                        && std::filesystem::is_directory(entry.path() / "Data"))
                    {
                        write(projectMaterial, bytes);
                        write(proofAssets / "textures" / extra, textureBytes);
                        addedDependency = true;
                        break;
                    }
            return false;
        };
        std::string raceError;
        try { (void)exportFo4WorldProof(proofWorld, root / "closure-race", proofOptions); }
        catch (const std::exception& error) { raceError = error.what(); }
        require(addedDependency && raceError.find("Model package changed after preflight") != std::string::npos
            && !std::filesystem::exists(root / "closure-race"),
            "Staging did not reject an added dependency against the exact inspected closure");
        std::filesystem::remove(projectMaterial);
        std::filesystem::remove(proofAssets / "textures" / extra);
        proofOptions.cancelled = {};

        const auto proofNif = proofAssets / package.modelPath;
        std::filesystem::create_directories(proofNif.parent_path());
        nifly::NifFile withController(package.assetRoot / package.modelPath);
        auto controller = std::make_unique<nifly::NiTransformController>();
        controller->targetRef.index = withController.GetBlockID(withController.GetRootNode());
        withController.GetRootNode()->controllerRef.index = withController.GetHeader().AddBlock(std::move(controller));
        require(withController.Save(proofNif) == 0, "Cannot save controller-only fixture");
        const auto controllerBytes = read(proofNif);
        const auto controllerGeometry = loadNifModel(controllerBytes);
        require(inspectNativeModel(controllerBytes).hasAnimation && controllerGeometry.animations.empty()
            && controllerGeometry.skinBones.empty() && controllerGeometry.attachmentTargets.empty(),
            "Native controller detection depends on the renderer's animation projection");
        const auto controlled = inspectFo4Export(proofWorld, proofOptions);
        require(!controlled.canWriteWorldProof() && controlled.blockers.front().find("dependency closure is not supported") != std::string::npos,
            "Controller-only native model passed the complete-dependency gate");
        rejects([&] { (void)exportFo4WorldProof(proofWorld, root / "controller-package", proofOptions); },
            "Controller-only model published an incomplete package");
        require(!std::filesystem::exists(root / "controller-package") && read(proofNif) == controllerBytes,
            "Rejected controller export changed source or published a package");

        nifly::NifFile withBehavior(package.assetRoot / package.modelPath);
        auto behavior = std::make_unique<nifly::BSBehaviorGraphExtraData>();
        behavior->behaviorGraphFile.get() = "actors/proof/behavior.hkx";
        withBehavior.GetRootNode()->extraDataRefs.AddBlockRef(withBehavior.GetHeader().AddBlock(std::move(behavior)));
        require(withBehavior.Save(proofNif) == 0 && inspectNativeModel(read(proofNif)).hasAnimation,
            "External behavior graph was not detected as an animation dependency");
        require(!inspectFo4Export(proofWorld, proofOptions).canWriteWorldProof(),
            "Unresolved external HKX behavior dependency passed proof preflight");

        nifly::NifFile withAttachment(package.assetRoot / package.modelPath);
        auto attachment = std::make_unique<nifly::BSConnectPointChildren>();
        attachment->targets.resize(1); attachment->targets[0].get() = "UnverifiedAttachment";
        withAttachment.GetRootNode()->extraDataRefs.AddBlockRef(withAttachment.GetHeader().AddBlock(std::move(attachment)));
        require(withAttachment.Save(proofNif) == 0, "Cannot save attachment diagnostic fixture");
        const auto attached = inspectFo4Export(proofWorld, proofOptions);
        require(!attached.canWriteWorldProof() && attached.blockers.front().find("dependency closure is not supported") != std::string::npos,
            "Native attachments passed the complete-dependency gate");
        rejects([&] { (void)exportFo4WorldProof(proofWorld, root / "attachment-package", proofOptions); },
            "Attached model published an incomplete package");
        require(!std::filesystem::exists(root / "attachment-package"), "Rejected attached model published a package");
#endif
    }
}

int main()
{
    const auto root = std::filesystem::temp_directory_path() / ("openf4-asset-package-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try
    {
        run(root); std::filesystem::remove_all(root);
        std::cout << "Fallout 4 native asset package tests passed\n"; return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << " (fixtures: " << root.string() << ")\n"; return 1;
    }
}
