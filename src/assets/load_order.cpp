#include "assets/load_order.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <zlib.h>

namespace openf4::assets
{
namespace
{
using Bytes = std::vector<std::byte>;
constexpr std::uint32_t tag(const char* s)
{
    return std::uint32_t(s[0]) | std::uint32_t(s[1]) << 8 | std::uint32_t(s[2]) << 16 | std::uint32_t(s[3]) << 24;
}
template<class T> T get(std::span<const std::byte> bytes, std::size_t offset)
{
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) throw std::runtime_error("Truncated plugin data");
    T value{}; std::memcpy(&value, bytes.data() + offset, sizeof(T)); return value;
}
std::string text(std::span<const std::byte> bytes)
{
    const auto* p = reinterpret_cast<const char*>(bytes.data());
    return {p, std::find(p, p + bytes.size(), '\0')};
}
Bytes readAt(std::ifstream& file, std::uint64_t offset, std::size_t size)
{
    Bytes bytes(size); file.clear(); file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!file) throw std::runtime_error("Truncated plugin file");
    return bytes;
}
template<class F> void fields(std::span<const std::byte> body, F&& visit)
{
    for (std::size_t p = 0; p + 6 <= body.size();)
    {
        auto type = get<std::uint32_t>(body, p); std::uint32_t size = get<std::uint16_t>(body, p + 4); p += 6;
        if (type == tag("XXXX"))
        {
            if (size != 4) throw std::runtime_error("Invalid extended subrecord");
            size = get<std::uint32_t>(body, p); p += 4;
            type = get<std::uint32_t>(body, p); p += 6;
        }
        if (size > body.size() - p) throw std::runtime_error("Subrecord exceeds record");
        visit(type, body.subspan(p, size)); p += size;
    }
}
Bytes recordBody(std::ifstream& file, std::uint64_t offset, std::uint32_t size, std::uint32_t flags)
{
    auto body = readAt(file, offset, size);
    if ((flags & 0x40000u) == 0) return body;
    const auto expanded = get<std::uint32_t>(body, 0);
    if (expanded > 64u * 1024u * 1024u) throw std::runtime_error("Compressed record exceeds limit");
    Bytes out(expanded); uLongf actual = expanded;
    if (uncompress(reinterpret_cast<Bytef*>(out.data()), &actual, reinterpret_cast<const Bytef*>(body.data() + 4),
        static_cast<uLong>(body.size() - 4)) != Z_OK || actual != expanded)
        throw std::runtime_error("Invalid compressed record");
    return out;
}
std::string extension(const std::filesystem::path& path) { return foldedName(path.extension().string()); }
}

std::string foldedName(std::string_view value)
{
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

bool PluginHeader::isMasterFile() const noexcept
{
    const auto ext = extension(path.empty() ? std::filesystem::path(name) : path);
    return (flags & 0x1u) != 0 || ext == ".esm" || ext == ".esl";
}

PluginHeader readPluginHeader(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open plugin: " + path.string());
    const auto fileSize = static_cast<std::uint64_t>(file.tellg());
    if (fileSize < 24) throw std::runtime_error("Not a Bethesda plugin (too small): " + path.string());
    const auto header = readAt(file, 0, 24);
    if (get<std::uint32_t>(header, 0) != tag("TES4")) throw std::runtime_error("Not a Bethesda plugin (no TES4): " + path.string());
    const auto size = get<std::uint32_t>(header, 4);
    if (size > fileSize - 24) throw std::runtime_error("Truncated TES4 header: " + path.string());
    PluginHeader result; result.path = path; result.name = path.filename().string();
    result.flags = get<std::uint32_t>(header, 8);
    result.localized = (result.flags & 0x80u) != 0;
    result.light = (result.flags & 0x200u) != 0 || extension(path) == ".esl";
    const auto body = recordBody(file, 24, size, result.flags);
    fields(body, [&](std::uint32_t type, std::span<const std::byte> value) {
        if (type == tag("MAST")) result.masters.push_back(text(value));
        if (type == tag("CNAM")) { const auto author = text(value); result.openf4Owned = author == "OpenF4" || author == "EMPIRE"; }
    });
    return result;
}

std::vector<WorldspaceInfo> listWorldspaces(const std::filesystem::path& path)
{
    const auto plugin = readPluginHeader(path);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto fileSize = static_cast<std::uint64_t>(file.tellg());
    std::vector<WorldspaceInfo> result;
    std::uint64_t p = 24 + get<std::uint32_t>(readAt(file, 0, 24), 4);
    while (p + 24 <= fileSize)
    {
        const auto header = readAt(file, p, 24);
        const auto type = get<std::uint32_t>(header, 0), size = get<std::uint32_t>(header, 4);
        if (type != tag("GRUP")) { p += 24ull + size; continue; }
        if (size < 24 || p + size > fileSize) throw std::runtime_error("Invalid top-level group in " + path.string());
        if (get<std::uint32_t>(header, 8) == tag("WRLD") && get<std::uint32_t>(header, 12) == 0)
        {
            for (auto q = p + 24; q + 24 <= p + size;)
            {
                const auto child = readAt(file, q, 24);
                const auto childType = get<std::uint32_t>(child, 0), childSize = get<std::uint32_t>(child, 4);
                if (childType == tag("GRUP")) { if (childSize < 24) throw std::runtime_error("Invalid WRLD child group"); q += childSize; continue; }
                if (childType == tag("WRLD"))
                {
                    WorldspaceInfo info; info.formId = get<std::uint32_t>(child, 12);
                    const auto slot = info.formId >> 24;
                    info.definingFile = slot < plugin.masters.size() ? plugin.masters[slot] : plugin.name;
                    fields(recordBody(file, q + 24, childSize, get<std::uint32_t>(child, 8)),
                        [&](std::uint32_t field, std::span<const std::byte> value) { if (field == tag("EDID")) info.editorId = text(value); });
                    result.push_back(std::move(info));
                }
                q += 24ull + childSize;
            }
        }
        p += size;
    }
    return result;
}

std::filesystem::path createTimestampedBackup(const std::filesystem::path& file)
{
    if (!std::filesystem::is_regular_file(file)) throw std::runtime_error("Cannot back up missing file: " + file.string());
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char stamp[32]{}; std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
    std::filesystem::path backup;
    for (int attempt = 0; attempt < 100; ++attempt)
    {
        backup = file; backup += "." + std::string(stamp) + (attempt ? "-" + std::to_string(attempt) : std::string()) + ".bak";
        if (!std::filesystem::exists(backup)) break;
    }
    if (std::filesystem::exists(backup)) throw std::runtime_error("Could not choose a free backup filename for " + file.string());
    std::filesystem::copy_file(file, backup, std::filesystem::copy_options::none);
    // Verify the copy byte-for-byte before anyone is allowed to write the master.
    std::ifstream a(file, std::ios::binary), b(backup, std::ios::binary);
    std::vector<char> x(1 << 20), y(1 << 20);
    while (a && b)
    {
        a.read(x.data(), static_cast<std::streamsize>(x.size())); b.read(y.data(), static_cast<std::streamsize>(y.size()));
        if (a.gcount() != b.gcount() || std::memcmp(x.data(), y.data(), static_cast<std::size_t>(a.gcount())) != 0)
        {
            a.close(); b.close(); std::error_code ignored; std::filesystem::remove(backup, ignored);
            throw std::runtime_error("Backup verification failed for " + file.string());
        }
    }
    if (!a.eof() || !b.eof()) throw std::runtime_error("Backup verification could not read " + file.string());
    return backup;
}

LoadOrder::LoadOrder(std::optional<std::filesystem::path> dataDirectory) : mData(std::move(dataDirectory)) {}

std::optional<std::size_t> LoadOrder::find(std::string_view name) const
{
    const auto key = foldedName(name);
    for (std::size_t i = 0; i < mEntries.size(); ++i) if (foldedName(mEntries[i].header.name) == key) return i;
    return std::nullopt;
}

const LoadOrder::Entry* LoadOrder::activeEntry() const noexcept
{
    return mActive && *mActive < mEntries.size() ? &mEntries[*mActive] : nullptr;
}

void LoadOrder::refreshMissing()
{
    for (auto& entry : mEntries)
    {
        entry.missingMasters.clear();
        for (const auto& master : entry.header.masters) if (!find(master)) entry.missingMasters.push_back(master);
    }
}

namespace
{
std::optional<std::string> orderProblem(const std::vector<LoadOrder::Entry>& entries)
{
    std::map<std::string, std::size_t> position;
    for (std::size_t i = 0; i < entries.size(); ++i) position[foldedName(entries[i].header.name)] = i;
    bool sawPlugin = false; std::string firstPlugin;
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        const auto& h = entries[i].header;
        if (!h.isMasterFile()) { if (!sawPlugin) firstPlugin = h.name; sawPlugin = true; }
        else if (sawPlugin) return "master file " + h.name + " cannot load after plugin " + firstPlugin
            + " (Fallout 4 always loads master files first)";
        for (const auto& master : h.masters)
            if (auto found = position.find(foldedName(master)); found != position.end() && found->second >= i)
                return h.name + " requires its master " + master + " to load before it";
    }
    return std::nullopt;
}
}

LoadOrder::AddResult LoadOrder::add(const std::filesystem::path& input)
{
    auto header = readPluginHeader(input);
    header.path = std::filesystem::absolute(input).lexically_normal();
    if (find(header.name)) throw std::runtime_error(header.name + " is already in the load order");
    std::size_t masters = 0; while (masters < mEntries.size() && mEntries[masters].header.isMasterFile()) ++masters;
    std::size_t lower = header.isMasterFile() ? 0 : masters, upper = header.isMasterFile() ? masters : mEntries.size();
    for (const auto& master : header.masters) if (auto at = find(master)) lower = std::max(lower, *at + 1);
    for (std::size_t i = 0; i < mEntries.size(); ++i)
        for (const auto& master : mEntries[i].header.masters)
            if (foldedName(master) == foldedName(header.name)) upper = std::min(upper, i);
    if (lower > upper)
        throw std::runtime_error("Cannot place " + header.name + ": its masters and dependents leave no valid position "
            "(a master file cannot depend on an ordinary plugin)");
    const auto active = activeEntry() ? std::optional(activeEntry()->header.path) : std::nullopt;
    mEntries.insert(mEntries.begin() + static_cast<std::ptrdiff_t>(upper), Entry{header, {}});
    if (active) for (std::size_t i = 0; i < mEntries.size(); ++i) if (mEntries[i].header.path == *active) mActive = i;
    refreshMissing();
    auto result = resolveMissingMasters(upper);
    result.index = upper;
    return result;
}

std::optional<std::filesystem::path> LoadOrder::locate(std::string_view name, const std::filesystem::path& besides) const
{
    std::vector<std::filesystem::path> folders{besides};
    if (mData) folders.push_back(*mData);
    for (const auto& entry : mEntries) folders.push_back(entry.header.path.parent_path());
    for (const auto& folder : folders)
    {
        if (folder.empty()) continue;
        const auto candidate = folder / std::filesystem::u8path(std::string(name));
        if (std::filesystem::is_regular_file(candidate)) return std::filesystem::absolute(candidate).lexically_normal();
    }
    return std::nullopt;
}

LoadOrder::AddResult LoadOrder::resolveMissingMasters(std::size_t index) const
{
    if (index >= mEntries.size()) throw std::out_of_range("Load order index");
    AddResult result; result.index = index; result.missingMasters = mEntries[index].missingMasters;
    std::set<std::string> visiting, done;
    const auto besides = mEntries[index].header.path.parent_path();
    std::function<void(const std::string&)> visit = [&](const std::string& name) {
        const auto key = foldedName(name);
        if (find(name) || done.contains(key) || visiting.contains(key)) return;
        visiting.insert(key);
        if (const auto path = locate(name, besides))
        {
            try
            {
                const auto header = readPluginHeader(*path);
                for (const auto& master : header.masters) visit(master);
                result.resolvable.push_back(*path);
            }
            catch (const std::exception&) { result.unresolvable.push_back(name + " (unreadable)"); }
        }
        else result.unresolvable.push_back(name);
        visiting.erase(key); done.insert(key);
    };
    for (const auto& name : mEntries[index].missingMasters) visit(name);
    return result;
}

std::size_t LoadOrder::addMissingMasters(std::size_t index)
{
    const auto plan = resolveMissingMasters(index);
    std::size_t added = 0;
    for (const auto& path : plan.resolvable) { (void)add(path); ++added; }
    return added;
}

bool LoadOrder::canMove(std::size_t from, std::size_t to, std::string* reason) const
{
    if (from >= mEntries.size() || to >= mEntries.size()) { if (reason) *reason = "invalid position"; return false; }
    if (from == to) return true;
    auto order = mEntries;
    auto moved = order[from]; order.erase(order.begin() + static_cast<std::ptrdiff_t>(from));
    order.insert(order.begin() + static_cast<std::ptrdiff_t>(to), std::move(moved));
    if (const auto problem = orderProblem(order)) { if (reason) *reason = *problem; return false; }
    return true;
}

void LoadOrder::move(std::size_t from, std::size_t to)
{
    std::string reason;
    if (!canMove(from, to, &reason)) throw std::runtime_error("Reorder refused: " + reason);
    if (from == to) return;
    const auto active = activeEntry() ? std::optional(activeEntry()->header.path) : std::nullopt;
    auto moved = mEntries[from]; mEntries.erase(mEntries.begin() + static_cast<std::ptrdiff_t>(from));
    mEntries.insert(mEntries.begin() + static_cast<std::ptrdiff_t>(to), std::move(moved));
    if (active) for (std::size_t i = 0; i < mEntries.size(); ++i) if (mEntries[i].header.path == *active) mActive = i;
}

std::vector<std::string> LoadOrder::dependents(std::size_t index) const
{
    std::vector<std::string> result;
    if (index >= mEntries.size()) return result;
    const auto key = foldedName(mEntries[index].header.name);
    for (const auto& entry : mEntries)
        for (const auto& master : entry.header.masters)
            if (foldedName(master) == key) { result.push_back(entry.header.name); break; }
    return result;
}

void LoadOrder::remove(std::size_t index)
{
    if (index >= mEntries.size()) throw std::out_of_range("Load order index");
    if (const auto users = dependents(index); !users.empty())
    {
        std::string list; for (const auto& u : users) list += (list.empty() ? "" : ", ") + u;
        throw std::runtime_error("Cannot remove " + mEntries[index].header.name + ": required by " + list);
    }
    if (mActive && *mActive == index) throw std::runtime_error("Cannot remove the active file; make another file active first");
    const auto active = activeEntry() ? std::optional(activeEntry()->header.path) : std::nullopt;
    mEntries.erase(mEntries.begin() + static_cast<std::ptrdiff_t>(index));
    mActive.reset();
    if (active) for (std::size_t i = 0; i < mEntries.size(); ++i) if (mEntries[i].header.path == *active) mActive = i;
    mIndexes.clear();
    refreshMissing();
}

void LoadOrder::clear()
{
    mEntries.clear(); mActive.reset(); mMasterEditing = false; mBackup.clear(); mIndexes.clear();
}

std::vector<std::string> LoadOrder::validate() const
{
    std::vector<std::string> problems;
    for (const auto& entry : mEntries)
        for (const auto& missing : entry.missingMasters) problems.push_back(entry.header.name + " is missing master " + missing);
    if (const auto problem = orderProblem(mEntries)) problems.push_back(*problem);
    return problems;
}

bool LoadOrder::isProtected(std::size_t index) const
{
    if (index >= mEntries.size()) return true;
    const auto& h = mEntries[index].header;
    return h.light || (h.isMasterFile() && !h.openf4Owned) || !dependents(index).empty();
}

void LoadOrder::makeActive(std::size_t index)
{
    if (index >= mEntries.size()) throw std::out_of_range("Load order index");
    const auto& target = mEntries[index];
    if (!target.missingMasters.empty())
        throw std::runtime_error("Cannot activate " + target.header.name + ": missing master " + target.missingMasters.front()
            + ". Add it from the Data folder first.");
    if (const auto problem = orderProblem(mEntries)) throw std::runtime_error("Cannot activate: load order is invalid: " + *problem);
    // Every master's own masters must be listed by the target too, or FormIDs cannot be mapped.
    std::set<std::string> listed; for (const auto& m : target.header.masters) listed.insert(foldedName(m));
    for (const auto& m : target.header.masters)
    {
        const auto at = find(m);
        if (!at) continue;
        if (!mEntries[*at].missingMasters.empty())
            throw std::runtime_error("Cannot activate " + target.header.name + ": its master " + m + " is missing "
                + mEntries[*at].missingMasters.front());
        for (const auto& transitive : mEntries[*at].header.masters)
            if (!listed.contains(foldedName(transitive)))
                throw std::runtime_error("Cannot activate " + target.header.name + ": it omits " + transitive
                    + ", which its master " + m + " requires");
    }
    mActive = index; mMasterEditing = false; mBackup.clear();
}

void LoadOrder::adoptActive(const std::filesystem::path& path)
{
    const auto normal = std::filesystem::absolute(path).lexically_normal();
    for (std::size_t i = 0; i < mEntries.size(); ++i)
        if (mEntries[i].header.path == normal) { mActive = i; mMasterEditing = false; mBackup.clear(); return; }
    throw std::runtime_error("Adopted file is not in the load order: " + path.string());
}

std::filesystem::path LoadOrder::enableMasterEditing(std::string_view typed)
{
    const auto* entry = activeEntry();
    if (!entry) throw std::runtime_error("Make a file active before enabling master editing");
    if (!isProtected(*mActive)) throw std::runtime_error(entry->header.name + " is an ordinary plugin; it is already editable");
    if (entry->header.light) throw std::runtime_error("Light (ESL) files cannot be edited by OpenF4 yet");
    if (typed != entry->header.name)
        throw std::runtime_error("Confirmation did not match. Type the exact filename: " + entry->header.name);
    mBackup = createTimestampedBackup(entry->header.path);
    mMasterEditing = true;
    return mBackup;
}

void LoadOrder::disableMasterEditing() noexcept { mMasterEditing = false; mBackup.clear(); }

bool LoadOrder::canWrite(const std::filesystem::path& documentPath, std::string* reason) const
{
    const auto fail = [&](std::string why) { if (reason) *reason = std::move(why); return false; };
    const auto* entry = activeEntry();
    if (!entry) return fail("No active file. Select a file in Load Order and press Make Active File.");
    std::error_code error;
    if (!std::filesystem::equivalent(documentPath, entry->header.path, error) || error)
        return fail("The open document is not the active file (" + entry->header.name + ").");
    if (entry->header.light) return fail(entry->header.name + " is a light (ESL) file; OpenF4 cannot edit light FormIDs yet.");
    if (isProtected(*mActive) && !mMasterEditing)
    {
        const auto users = dependents(*mActive);
        return fail("READ-ONLY: " + entry->header.name + (users.empty() ? " is a master file." : " is a master of " + users.front() + ".")
            + " Make an ordinary plugin active to save overrides, or use Enable Master Editing.");
    }
    return true;
}

std::vector<std::filesystem::path> LoadOrder::activeStack() const
{
    const auto* entry = activeEntry();
    if (!entry) return {};
    std::vector<std::filesystem::path> result;
    for (const auto& master : entry->header.masters)
    {
        if (auto at = find(master)) result.push_back(mEntries[*at].header.path);
        else if (auto path = locate(master, entry->header.path.parent_path())) result.push_back(*path);
        else result.push_back(entry->header.path.parent_path() / std::filesystem::u8path(master));
    }
    result.push_back(entry->header.path);
    return result;
}

std::vector<WorldspaceInfo> LoadOrder::activeWorldspaces() const
{
    const auto* entry = activeEntry();
    if (!entry) return {};
    std::vector<std::string> canonical = entry->header.masters; canonical.push_back(entry->header.name);
    std::map<std::uint32_t, WorldspaceInfo> worlds;
    for (const auto& path : activeStack())
    {
        if (!std::filesystem::is_regular_file(path)) continue;
        const auto header = readPluginHeader(path);
        for (auto info : listWorldspaces(path))
        {
            const auto slot = info.formId >> 24;
            const auto owner = slot < header.masters.size() ? header.masters[slot] : header.name;
            const auto at = std::find_if(canonical.begin(), canonical.end(), [&](const auto& n) { return foldedName(n) == foldedName(owner); });
            if (at == canonical.end()) continue;
            info.formId = (static_cast<std::uint32_t>(at - canonical.begin()) << 24) | (info.formId & 0xFFFFFFu);
            info.definingFile = *at;
            auto& slotInfo = worlds[info.formId];
            if (slotInfo.editorId.empty() || !info.editorId.empty()) { const auto keep = slotInfo.editorId; slotInfo = info; if (info.editorId.empty()) slotInfo.editorId = keep; }
        }
    }
    std::vector<WorldspaceInfo> result;
    for (auto& [id, info] : worlds) result.push_back(std::move(info));
    return result;
}

std::vector<std::string> LoadOrder::otherOverrides(std::string_view ownerName, std::uint32_t objectId) const
{
    std::vector<std::string> result;
    const auto* active = activeEntry();
    if (!active) return result;
    std::set<std::string> stack; for (const auto& m : active->header.masters) stack.insert(foldedName(m));
    stack.insert(foldedName(active->header.name));
    const auto owner = foldedName(ownerName);
    for (std::size_t i = 0; i < mEntries.size(); ++i)
    {
        const auto& h = mEntries[i].header;
        if (stack.contains(foldedName(h.name))) continue;
        auto& index = mIndexes[foldedName(h.name)];
        const std::string where = i < *mActive ? " (loads before active)" : " (loads AFTER active: wins in game)";
        if (!index.indexed)
        {
            index.indexed = true;
            index.owners = h.masters; index.owners.push_back(h.name);
            for (auto& o : index.owners) o = foldedName(o);
            std::error_code error; const auto size = std::filesystem::file_size(h.path, error);
            if (error || size > 64ull * 1024 * 1024) { index.owners.clear(); }
            else
            {
                std::ifstream file(h.path, std::ios::binary);
                Bytes bytes(static_cast<std::size_t>(size));
                file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                std::function<void(std::size_t, std::size_t, unsigned)> scan = [&](std::size_t begin, std::size_t end, unsigned depth) {
                    if (depth > 32) return;
                    for (std::size_t p = begin; p + 24 <= end;)
                    {
                        const std::span<const std::byte> view(bytes);
                        const auto type = get<std::uint32_t>(view, p), length = get<std::uint32_t>(view, p + 4);
                        if (type == tag("GRUP")) { if (length < 24 || p + length > end) return; scan(p + 24, p + length, depth + 1); p += length; continue; }
                        const auto form = get<std::uint32_t>(view, p + 12);
                        if (type != tag("TES4") && (form >> 24) < index.owners.size())
                            index.records.insert((std::uint64_t(form >> 24) << 32) | (form & 0xFFFFFFu));
                        p += 24ull + length;
                    }
                };
                try { scan(0, bytes.size(), 0); } catch (const std::exception&) { index.owners.clear(); index.records.clear(); }
            }
        }
        if (index.owners.empty()) { result.push_back(h.name + " (not indexed)" ); continue; }
        const auto slot = std::find(index.owners.begin(), index.owners.end(), owner);
        if (slot == index.owners.end()) continue;
        if (index.records.contains((std::uint64_t(slot - index.owners.begin()) << 32) | (objectId & 0xFFFFFFu)))
            result.push_back(h.name + where);
    }
    return result;
}
}
