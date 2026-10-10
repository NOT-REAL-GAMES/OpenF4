#include "editor/asset_document.hpp"
#include "assets/asset_resolver.hpp"
#include "assets/coordinates.hpp"
#include "physics/jolt_collision.hpp"
#include "compatibility_fixtures.hpp"

#include <NifFile.hpp>
#include <ExtraData.hpp>
#include <bhk.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <glm/ext/matrix_transform.hpp>

namespace
{
    using namespace openf4;
    void require(bool value, const char* message)
    {
        if (!value) throw std::runtime_error(message);
    }
    template<class F> void rejects(F&& operation, const char* message)
    {
        try { operation(); }
        catch (const std::exception&) { return; }
        throw std::runtime_error(message);
    }
    std::vector<std::byte> read(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        require(bool(input), "Cannot read fixture");
        std::vector<std::byte> bytes(static_cast<std::size_t>(input.tellg()));
        input.seekg(0); input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        require(bool(input), "Cannot read fixture bytes"); return bytes;
    }
    void write(const std::filesystem::path& path, std::span<const std::byte> bytes)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        require(bool(output), "Cannot write fixture");
    }
    assets::ModelGeometry cube()
    {
        assets::ModelGeometry result;
        auto& surface = result.surfaces.emplace_back();
        for (unsigned i = 0; i < 8; ++i)
        {
            render::SourceVertex vertex;
            vertex.position = glm::vec3(i & 1 ? 70 : -70, i & 2 ? 70 : -70, i & 4 ? 70 : -70);
            vertex.normal = glm::normalize(vertex.position);
            surface.vertices.push_back(vertex);
        }
        surface.indices = {0,1,3,0,3,2,4,6,7,4,7,5,0,4,5,0,5,1,
            2,3,7,2,7,6,0,2,6,0,6,4,1,5,7,1,7,3};
        surface.metallicFactor = 0; surface.roughnessFactor = 1;
        assets::ModelImage image;
        image.width = 1; image.height = 1; image.srgb = true; image.mipOffsets = {0};
        image.pixels = {std::byte{64}, std::byte{128}, std::byte{192}, std::byte{255}};
        surface.diffuseImage = std::move(image);
        return result;
    }
    void testHistoryAndMaterials()
    {
        editor::AssetDocument asset(cube());
        require(!asset.dirty() && !asset.movable(), "New asset is dirty or movable");
        const auto pixels = asset.materials()[0].diffuseImage;
        const auto* geometry = &asset.sourceGeometry();
        asset.beginTransaction();
        for (int i = 0; i < 5; ++i)
        {
            auto material = asset.materials()[0]; material.baseColor.r = 0.1f * i;
            asset.setMaterial(0, material);
        }
        asset.endTransaction();
        require(asset.dirty() && asset.materials()[0].diffuseImage == pixels
            && &asset.sourceGeometry() == geometry, "History copied immutable geometry or texture data");
        require(asset.undo() && !asset.dirty() && !asset.undo(), "Gesture did not create exactly one undo entry");
        asset.beginTransaction();
        auto material = asset.materials()[0]; material.doubleSided = true;
        asset.setMaterial(0, material); asset.cancelTransaction();
        require(asset.redo() && asset.dirty(), "Cancelled gesture destroyed redo history");
        asset.markSaved();
        require(!asset.dirty() && asset.undo() && asset.dirty() && asset.redo() && !asset.dirty(),
            "Saved-state identity did not survive undo/redo");
        material = asset.materials()[0]; material.alphaCutoff = -1;
        rejects([&] { asset.setMaterial(0, material); }, "Invalid alpha cutoff accepted");
        require(!asset.dirty(), "Failed edit changed asset state");
        for (int i = 0; i < 80; ++i)
        {
            material = asset.materials()[0]; material.baseColor.g = (i & 1) ? .1f : .9f;
            asset.setMaterial(0, material);
        }
        int count = 0; while (asset.undo()) ++count;
        require(count == 64, "Asset history is not bounded to 64 transactions");
    }

    void testCollision()
    {
        editor::AssetDocument asset(cube());
        asset.addBox({0, 70, 0}, {35, 70, 35});
        asset.addCapsule({70, 0, 0}, {70, 100, 0}, 20);
        asset.setMovable(true, 12);
        const auto& body = asset.collisionSystems()[0].physics.bodies[0];
        require(body.mass == 12 && glm::all(glm::greaterThan(body.inertiaDiagonal, glm::vec3(0))),
            "Movable collision has no derived mass properties");
        auto transform = glm::translate(glm::mat4(1), glm::vec3(35, 140, -70));
        transform = glm::rotate(transform, 0.4f, glm::vec3(0, 1, 0));
        transform = glm::scale(transform, glm::vec3(2, 1, .5f));
        asset.setPartTransform(editor::AssetCollisionPart::convex, 0, transform);
        const auto roundTrip = asset.partTransform(editor::AssetCollisionPart::convex, 0);
        for (int i = 0; i < 4; ++i)
            require(glm::length(roundTrip[i] - transform[i]) < .001f, "Part matrix axes/units did not round trip");
        rejects([&] { asset.setPartTransform(editor::AssetCollisionPart::capsule, 0, transform); },
            "Nonuniform capsule scale accepted");
        const std::array<std::size_t, 1> surfaces{0};
        rejects([&] { asset.generateCollision(surfaces, false); }, "Movable triangle collision accepted");
        asset.setMovable(false); asset.generateCollision(surfaces, false);
        require(asset.collisionSystems()[0].physics.bodies[0].triangleMesh.has_value(), "Triangle generation failed");
        rejects([&] { asset.setMovable(true); }, "Triangle body changed to movable");
        asset.generateCollision(surfaces, true);
        require(asset.collisionSystems()[0].physics.bodies[0].parts.size() == 1, "Convex generation failed");
        asset.erasePart(editor::AssetCollisionPart::convex, 0);
        require(asset.collisionSystems()[0].physics.bodies[0].parts.empty(), "Collision deletion failed");
        require(asset.undo() && asset.collisionSystems()[0].physics.bodies[0].parts.size() == 1,
            "Collision deletion undo failed");
    }

    void testAuthoredMassProperties(const std::filesystem::path& root)
    {
        const auto world = root / "mass-frame.esm";
        editor::AssetDocument shapes(cube());
        shapes.addBox({}, {70, 35, 50});
        shapes.addBox({210, 70, -35}, {35, 70, 20});
        shapes.addCapsule({-100, 0, 0}, {-100, 100, 0}, 20);
        auto collision = shapes.collisionSystems()[0].physics;
        auto& authored = collision.bodies[0];
        authored.dynamic = true; authored.mass = 7;
        authored.centerOfMass = {.2f, -.1f, .3f};
        authored.inertiaDiagonal = {11, 17, 29};
        authored.inertiaOrientation = glm::angleAxis(.35f, glm::normalize(glm::vec3(1, 2, 3)));
        const auto package = assets::exportNativeModelAsset(world, 0x1000800, cube(), &collision);
        const auto sourceBytes = read(package.assetRoot / package.modelPath);
        const auto original = assets::loadNifPhysicsSystems(sourceBytes).at(0).bodies.at(0);
        const auto exactFrame = [](const auto& a, const auto& b) {
            return a.dynamic == b.dynamic && a.mass == b.mass && a.centerOfMass == b.centerOfMass
                && a.inertiaDiagonal == b.inertiaDiagonal && a.inertiaOrientation == b.inertiaOrientation;
        };
        const auto closeFrame = [](const auto& a, const auto& b) {
            const auto ra = glm::mat3_cast(a.inertiaOrientation), rb = glm::mat3_cast(b.inertiaOrientation);
            glm::mat3 da(0), db(0);
            for (int i = 0; i < 3; ++i) { da[i][i] = a.inertiaDiagonal[i]; db[i][i] = b.inertiaDiagonal[i]; }
            const auto difference = ra * da * glm::transpose(ra) - rb * db * glm::transpose(rb);
            return a.dynamic == b.dynamic && std::abs(a.mass - b.mass) < 1e-4f
                && glm::length(a.centerOfMass - b.centerOfMass) < 1e-4f
                && glm::length(difference[0]) < 1e-4f && glm::length(difference[1]) < 1e-4f
                && glm::length(difference[2]) < 1e-4f;
        };
        editor::AssetDocument asset(world, package.modelPath, std::nullopt, true);
        const auto current = [&] { return asset.collisionSystems()[0].physics.bodies[0]; };
        require(exactFrame(current(), original) && !asset.dirty() && !asset.undo(),
            "Matching MSTT hint replaced authored COM/inertia or created hidden history");
        editor::AssetDocument noHint(world, package.modelPath);
        require(exactFrame(noHint.collisionSystems()[0].physics.bodies[0], original),
            "Loading an asset replaced its authored mass frame");
        const auto untouched = asset.saveNative(world, 0x1000800);
        const auto roundTrip = assets::loadNifPhysicsSystems(read(untouched.assetRoot / untouched.modelPath)).at(0).bodies.at(0);
        require(closeFrame(roundTrip, original) && !asset.dirty() && !asset.undo(),
            "Untouched native save replaced authored COM/inertia or edited the document");

        const auto revision = asset.revision();
        asset.setMovable(true); asset.setBodyProperties(0, 0, asset.bodyProperties(0, 0));
        asset.setPartTransform(editor::AssetCollisionPart::convex, 0,
            asset.partTransform(editor::AssetCollisionPart::convex, 0));
        require(asset.revision() == revision && !asset.dirty() && !asset.undo() && exactFrame(current(), original),
            "Repeated mode/properties/transform changed the authored mass frame or history");
        const std::array<float editor::AssetBodyProperties::*, 5> contactFields{
            &editor::AssetBodyProperties::friction, &editor::AssetBodyProperties::restitution,
            &editor::AssetBodyProperties::linearDamping, &editor::AssetBodyProperties::angularDamping,
            &editor::AssetBodyProperties::gravityFactor};
        for (const auto field : contactFields)
        {
            auto properties = asset.bodyProperties(0, 0); properties.*field += .125f;
            asset.setBodyProperties(0, 0, properties);
            require(exactFrame(current(), original), "Contact/damping/gravity edit replaced authored COM/inertia");
            require(asset.undo() && !asset.dirty() && exactFrame(current(), original), "Property undo lost authored mass frame");
            require(asset.redo() && exactFrame(current(), original), "Property redo recalculated authored mass frame");
            require(asset.undo(), "Cannot restore mass fixture baseline");
            asset.beginTransaction(); asset.setBodyProperties(0, 0, properties); asset.cancelTransaction();
            require(!asset.dirty() && exactFrame(current(), original), "Cancelled property gesture lost authored mass frame");
        }
        auto material = asset.materials()[0]; material.baseColor.g = .5f;
        asset.setMaterial(0, material);
        require(exactFrame(current(), original) && asset.undo(), "Material edit replaced authored mass frame");

        auto properties = asset.bodyProperties(0, 0); properties.mass = 14;
        auto expected = original; expected.mass = 14; physics::updateHavokMassProperties(expected);
        asset.setBodyProperties(0, 0, properties);
        require(closeFrame(current(), expected) && glm::length(current().centerOfMass - original.centerOfMass) > .1f,
            "Mass-only edit did not derive uniform-density COM/inertia from geometry");
        require(asset.undo() && exactFrame(current(), original), "Mass edit undo failed to restore authored mass frame");
        require(asset.redo() && closeFrame(current(), expected), "Mass edit redo changed derived mass frame");
        require(asset.undo(), "Cannot undo derived mass frame");
        for (const float invalid : {0.f, -1.f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
        {
            properties = asset.bodyProperties(0, 0); properties.mass = invalid;
            const auto before = asset.revision();
            rejects([&] { asset.setBodyProperties(0, 0, properties); }, "Invalid dynamic mass accepted");
            require(asset.revision() == before && !asset.dirty() && exactFrame(current(), original),
                "Invalid mass changed the authored document");
        }
        asset.beginTransaction();
        properties = asset.bodyProperties(0, 0); properties.mass = 21; asset.setBodyProperties(0, 0, properties);
        asset.cancelTransaction();
        require(!asset.dirty() && exactFrame(current(), original), "Cancelled mass gesture failed to restore authored frame");

        const auto checkDerivedShape = [&] {
            auto expectedShape = current(); physics::updateHavokMassProperties(expectedShape);
            require(closeFrame(current(), expectedShape) && current().mass == original.mass,
                "Shape edit failed to derive mass properties or changed the authored mass");
            require(asset.undo() && exactFrame(current(), original), "Shape undo failed to restore authored mass frame");
        };
        asset.addBox({350, 0, 0}, {35, 35, 35}); checkDerivedShape();
        asset.setPartTransform(editor::AssetCollisionPart::convex, 1,
            glm::translate(glm::mat4(1), glm::vec3(280, -70, 140))); checkDerivedShape();
        asset.setCapsuleGeometry(0, {-140, -35, 0}, {-140, 210, 0}, 35); checkDerivedShape();
        asset.erasePart(editor::AssetCollisionPart::convex, 1); checkDerivedShape();
        asset.generateCollision(std::array<std::size_t, 1>{0}, true); checkDerivedShape();
        const auto beforeBadCapsule = asset.revision();
        rejects([&] { asset.setCapsuleGeometry(0, {}, {}, -1); }, "Negative capsule radius accepted");
        require(asset.revision() == beforeBadCapsule && exactFrame(current(), original), "Failed capsule edit changed mass properties");

        asset.setMovable(false);
        require(!current().dynamic && current().mass == 0 && current().centerOfMass == glm::vec3(0)
            && current().inertiaDiagonal == glm::vec3(0) && current().inertiaOrientation == glm::quat(1, 0, 0, 0),
            "Dynamic-to-static transition retained dynamic mass properties");
        const auto staticRevision = asset.revision(); asset.setMovable(false);
        require(asset.revision() == staticRevision, "Repeated static mode created an undo entry");
        properties = asset.bodyProperties(0, 0); properties.mass = 1;
        rejects([&] { asset.setBodyProperties(0, 0, properties); }, "Static mass was editable");
        require(asset.undo() && !asset.dirty() && exactFrame(current(), original), "Mode undo lost authored mass frame");
        require(asset.redo(), "Mode redo failed");
        const auto staticPackage = asset.saveNative(world, 0x1000801);
        editor::AssetDocument matchingStatic(world, staticPackage.modelPath, std::nullopt, false);
        require(!matchingStatic.dirty() && !matchingStatic.undo() && !matchingStatic.movable(),
            "Matching static hint created hidden history");
        editor::AssetDocument promoteHint(world, staticPackage.modelPath, std::nullopt, true);
        expected = original; expected.mass = 10; physics::updateHavokMassProperties(expected);
        require(closeFrame(promoteHint.collisionSystems()[0].physics.bodies[0], expected)
            && !promoteHint.dirty() && !promoteHint.undo(), "Static-to-movable hint did not derive a clean baseline");
        asset.setMovable(true, 12); expected.mass = 12; physics::updateHavokMassProperties(expected);
        require(closeFrame(current(), expected), "Static-to-dynamic transition did not derive the requested mass");
        require(read(package.assetRoot / package.modelPath) == sourceBytes, "Mass authoring modified the immutable source package");
    }

    void testNativeSave(const std::filesystem::path& root)
    {
        const auto world = root / "world.esm";
        editor::AssetDocument asset(cube());
        asset.addBox({}, glm::vec3(70)); asset.setMovable(true);
        auto material = asset.materials()[0]; material.alphaCutoff = .4f; material.doubleSided = true;
        asset.setMaterial(0, material);
        const auto package = asset.saveNative(world, 0x1000800);
        require(asset.dirty(), "Export marked asset saved before world assignment succeeded");
        const auto bytes = read(package.assetRoot / package.modelPath);
        editor::AssetDocument loaded(world, package.modelPath);
        editor::AssetDocument modlReload(world, package.modelPath.substr(7));
        require(modlReload.capabilities().materials, "Data-relative Bethesda MODL path did not resolve");
        editor::AssetDocument staticRecord(world, package.modelPath, std::nullopt, false);
        const auto& staticBody = staticRecord.collisionSystems()[0].physics.bodies[0];
        require(!staticRecord.movable() && staticBody.mass == 0 && staticBody.inertiaDiagonal == glm::vec3(0)
            && !staticRecord.dirty() && !staticRecord.undo(),
            "STAT motion hint created hidden history or kept dynamic mass properties");
        editor::AssetDocument movableRecord(world, package.modelPath, std::nullopt, true);
        require(movableRecord.movable() && movableRecord.collisionSystems()[0].physics.bodies[0].mass > 0
            && !movableRecord.dirty() && !movableRecord.undo(),
            "MSTT motion hint did not initialize a clean dynamic baseline");
        require(loaded.capabilities().materials && loaded.capabilities().collision && loaded.movable(),
            "Own native export is not editable/movable after reload");
        require(loaded.materials()[0].doubleSided && std::abs(loaded.materials()[0].alphaCutoff - .4f) < .005f,
            "Material factors were lost on native reload");
        auto properties = loaded.bodyProperties(0, 0); properties.friction = .7f;
        loaded.setBodyProperties(0, 0, properties);
        const auto second = loaded.saveNative(world, 0x1000800);
        require(second.modelPath != package.modelPath && read(package.assetRoot / package.modelPath) == bytes,
            "Native export overwrote its source");
        loaded.markSaved();
        rejects([&] { (void)loaded.saveNative(world, 0); }, "Invalid native FormID accepted");
        require(!loaded.dirty(), "Failed export changed dirty state");

        loaded.setTexture(0, {}, false); loaded.setTexture(0, {}, true);
        const auto cleared = loaded.geometryForExport();
        require(!cleared.surfaces[0].diffuseImage && !cleared.surfaces[0].normalImage
            && cleared.surfaces[0].diffuseTexture.empty() && cleared.surfaces[0].normalTexture.empty(),
            "Clearing texture restored the immutable source image");
        material = loaded.materials()[0]; material.alphaBlend = true; material.alphaCutoff = 0;
        loaded.setMaterial(0, material);
        const auto clearPackage = loaded.saveNative(world, 0x1000800);
        editor::AssetDocument clearReload(world, clearPackage.modelPath);
        require(clearReload.materials()[0].alphaBlend && clearReload.materials()[0].alphaCutoff == 0,
            "Blend-only material acquired alpha testing during reload");
        const auto& fallback = *clearReload.materials()[0].diffuseImage;
        require(fallback.pixels.size() >= 4 && fallback.pixels[0] == std::byte{255}
            && fallback.pixels[1] == std::byte{255} && fallback.pixels[2] == std::byte{255},
            "Cleared diffuse image was not exported as the neutral fallback");

        // An unknown source block must survive the collision-only path, not be flattened away.
        nifly::NifFile nif(package.assetRoot / package.modelPath);
        nif.GetRootNode()->name.get() = "Third Party Native Model";
        auto note = std::make_unique<nifly::NiStringExtraData>();
        note->name.get() = "UnrelatedAuthorData"; note->stringData.get() = "preserve this data exactly";
        nif.GetRootNode()->extraDataRefs.AddBlockRef(nif.GetHeader().AddBlock(std::move(note)));
        const auto foreignPath = "meshes/foreign.nif";
        require(nif.Save(package.assetRoot / foreignPath) == 0, "Cannot save native preservation fixture");
        editor::AssetDocument foreign(world, foreignPath);
        editor::AssetDocument foreignStaticHint(world, foreignPath, std::nullopt, false);
        require(foreignStaticHint.movable() && !foreignStaticHint.dirty() && !foreignStaticHint.undo(),
            "Record hint changed preserved native collision motion flags");
        require(!foreign.capabilities().materials && foreign.capabilities().collision
            && !foreign.capabilities().motion, "Foreign native edit capabilities are unsafe");
        rejects([&] { foreign.setMaterial(0, foreign.materials()[0]); }, "Foreign material edit accepted");
        rejects([&] { foreign.setMovable(false); }, "Foreign motion flags can change");
        properties = foreign.bodyProperties(0, 0); properties.friction = .8f;
        foreign.setBodyProperties(0, 0, properties);
        const auto otherWorld = root / "copy.esm";
        const auto copied = foreign.saveNative(otherWorld, 0x1000800);
        editor::AssetDocument roundTrip(otherWorld, copied.modelPath);
        const auto nativeFriction = std::bit_cast<float>(std::bit_cast<std::uint32_t>(.8f) & 0xffff0000u);
        require(roundTrip.collisionSystems()[0].physics.bodies[0].friction == nativeFriction,
            "Collision-only save lost authored properties");
        nifly::NifFile preserved(copied.assetRoot / copied.modelPath);
        bool found = false;
        for (std::uint32_t i = 0; i < preserved.GetHeader().GetNumBlocks(); ++i)
            if (auto* value = preserved.GetHeader().GetBlock<nifly::NiStringExtraData>(i))
                if (value->name.get() == "UnrelatedAuthorData")
                    found = value->stringData.get() == "preserve this data exactly";
        require(found, "Collision-only publication erased unrelated native data");
        require(roundTrip.materials()[0].diffuseImage != nullptr, "Project dependency closure was not copied");

        const auto dependency = assets::inspectNativeModel(read(package.assetRoot / foreignPath)).dependencies.front();
        const auto conflictRoot = assets::AssetResolver::projectAssetRoot(root / "conflict.esm");
        const std::array<std::byte, 3> conflict{std::byte{1}, std::byte{2}, std::byte{3}};
        write(conflictRoot / dependency, conflict);
        rejects([&] { (void)foreign.saveNative(root / "conflict.esm", 0x1000800); },
            "Native dependency conflict was overwritten");
        require(read(conflictRoot / dependency) == std::vector<std::byte>(conflict.begin(), conflict.end()),
            "Failed export damaged an existing dependency");
        require(!std::filesystem::exists(conflictRoot / "meshes"), "Failed preflight published a native model");

        // A recognizable own NIF must not erase an externally edited unsupported BGSM field.
        std::string materialPath = loaded.sourceGeometry().surfaces[0].materialPath;
        std::ranges::replace(materialPath, '\\', '/');
        auto materialBytes = read(package.assetRoot / materialPath);
        require(assets::nativeMaterialIsEditable(materialBytes), "Own BGSM was not recognized");
        materialBytes[8] = std::byte{0}; // UV clamp mode is not an authored asset-editor field.
        write(package.assetRoot / materialPath, materialBytes);
        editor::AssetDocument unsupported(world, package.modelPath);
        require(!unsupported.capabilities().materials && unsupported.capabilities().collision,
            "Externally edited unsupported material fields could be discarded");
        (void)unsupported.saveNative(world, 0x1000800);
        require(read(package.assetRoot / materialPath) == materialBytes,
            "Collision-only save rewrote unsupported source material fields");

        auto ragdoll = std::make_unique<nifly::bhkRagdollSystem>();
        nif.GetRootNode()->extraDataRefs.AddBlockRef(nif.GetHeader().AddBlock(std::move(ragdoll)));
        require(nif.Save(package.assetRoot / "meshes/ragdoll.nif") == 0, "Cannot save unsupported ragdoll fixture");
        editor::AssetDocument unsupportedRagdoll(world, "ragdoll.nif");
        require(!unsupportedRagdoll.capabilities().collision && !unsupportedRagdoll.capabilities().materials,
            "Ragdoll collateral did not make the native asset read-only");
        rejects([&] { (void)unsupportedRagdoll.saveNative(world, 0x1000801); }, "Unsupported ragdoll export accepted");

        editor::AssetDocument noCollision(cube());
        const auto noCollisionPackage = noCollision.saveNative(world, 0x1000802);
        editor::AssetDocument unshapedMovable(world, noCollisionPackage.modelPath, std::nullopt, true);
        require(unshapedMovable.movable() && !unshapedMovable.dirty() && !unshapedMovable.undo(),
            "Unshaped movable source did not retain its record kind");
        rejects([&] { (void)unshapedMovable.saveNative(world, 0x1000802); }, "Unshaped movable native asset was published");
        unshapedMovable.addBox({}, glm::vec3(35));
        require(unshapedMovable.collisionSystems()[0].physics.bodies[0].mass > 0,
            "Authoring first movable shape did not derive mass properties");
        editor::AssetDocument triangle(cube());
        triangle.generateCollision(std::array<std::size_t, 1>{0}, false);
        const auto trianglePackage = triangle.saveNative(world, 0x1000803);
        editor::AssetDocument incompatible(world, trianglePackage.modelPath, std::nullopt, true);
        require(!incompatible.capabilities().collision && !incompatible.capabilities().materials
            && !incompatible.dirty() && !incompatible.undo(),
            "MSTT triangle source was not kept inspect-only without history");

        const auto missingPackage = asset.saveNative(world, 0x1000804);
        const auto missingGeometry = assets::loadNifModel(read(missingPackage.assetRoot / missingPackage.modelPath));
        std::string missingMaterial = missingGeometry.surfaces[0].materialPath;
        std::ranges::replace(missingMaterial, '\\', '/');
        std::filesystem::remove(missingPackage.assetRoot / missingMaterial);
        editor::AssetDocument missing(world, missingPackage.modelPath);
        require(!missing.capabilities().materials && !missing.capabilities().motion && missing.capabilities().collision,
            "Missing native material enabled lossy conversion to defaults");
        rejects([&] { (void)missing.saveNative(root / "missing.esm", 0x1000804); },
            "Missing native dependency was silently dropped by collision-only save");
        require(!std::filesystem::exists(assets::AssetResolver::projectAssetRoot(root / "missing.esm")),
            "Missing dependency export published partial files");

        const auto unreadablePackage = asset.saveNative(world, 0x1000805);
        const auto unreadableGeometry = assets::loadNifModel(read(unreadablePackage.assetRoot / unreadablePackage.modelPath));
        std::string unreadableTexture = unreadableGeometry.surfaces[0].diffuseTexture;
        std::ranges::replace(unreadableTexture, '\\', '/');
        write(unreadablePackage.assetRoot / unreadableTexture, conflict);
        editor::AssetDocument unreadable(world, unreadablePackage.modelPath);
        require(!unreadable.capabilities().materials && !unreadable.capabilities().motion && unreadable.capabilities().collision,
            "Unreadable native texture enabled lossy conversion to defaults");
        require(read(unreadablePackage.assetRoot / unreadableTexture) == std::vector<std::byte>(conflict.begin(), conflict.end()),
            "Loading malformed native texture changed its source bytes");
    }
}

int main(int argc, char** argv)
{
    const auto root = std::filesystem::temp_directory_path() /
        ("openf4-asset-document-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try
    {
        if (argc == 3 && std::string_view(argv[1]) == "--compatibility-fixtures")
        {
            openf4::tests::runCompatibilityTests(std::filesystem::absolute(argv[2]));
            std::cout << "Compatibility fixtures verified (Fallout 4 runtime gate remains separate)\n";
            return 0;
        }
        require(argc == 1, "Usage: openf4-asset-document-tests [--compatibility-fixtures <new-directory>]");
        std::filesystem::create_directories(root);
        testHistoryAndMaterials(); testCollision(); testNativeSave(root); testAuthoredMassProperties(root);
        openf4::tests::runCompatibilityTests(root / "compatibility");
        std::filesystem::remove_all(root);
        std::cout << "Asset document tests passed\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << " (fixtures: " << root.string() << ")\n";
        return 1;
    }
}
