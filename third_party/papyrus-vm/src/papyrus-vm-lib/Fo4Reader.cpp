#include "papyrus-vm/Fo4Reader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <set>
#include <stdexcept>

namespace {
std::string lower(std::string value)
{
  for (auto& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return value;
}

class CheckedReader
{
  std::span<const uint8_t> bytes;
  size_t cursor = 0;
  Fo4Script result;
  std::set<std::string> deps;
  std::set<std::string> unsupported;
  size_t instructionTotal = 0;

  [[noreturn]] void fail(const char* reason) const
  {
    throw std::runtime_error("FO4 PEX at byte " + std::to_string(cursor) + ": " + reason);
  }
  uint64_t number(size_t count)
  {
    if (count > bytes.size() - cursor) fail("truncated data");
    uint64_t value = 0;
    for (size_t i = 0; i < count; ++i) value |= uint64_t(bytes[cursor++]) << (8 * i);
    return value;
  }
  uint8_t u8() { return static_cast<uint8_t>(number(1)); }
  uint16_t u16() { return static_cast<uint16_t>(number(2)); }
  uint32_t u32() { return static_cast<uint32_t>(number(4)); }
  std::string string()
  {
    const auto count = u16();
    if (count > bytes.size() - cursor) fail("truncated string");
    std::string s(reinterpret_cast<const char*>(bytes.data() + cursor), count);
    cursor += count;
    if (s.find('\0') != std::string::npos) fail("embedded NUL in string");
    return s;
  }
  const std::string& symbol()
  {
    const auto index = u16();
    const auto& strings = result.script->stringTable.GetStorage();
    if (index >= strings.size()) fail("invalid string table index");
    return strings[index];
  }
  void type(const std::string& raw)
  {
    auto name = lower(raw);
    if (name == "identifier") fail("invalid declared type");
    if (name.ends_with("[]")) {
      unsupported.insert("array values");
      name.resize(name.size() - 2);
    }
    if (name == "var") unsupported.insert("Var values");
    if (name.find('#') != std::string::npos) {
      unsupported.insert("struct values");
      name.resize(name.find('#'));
    }
    if (!name.empty() && name != "int" && name != "bool" && name != "float" &&
        name != "string" && name != "none" && name != "var") deps.insert(name);
  }
  VarValue value()
  {
    switch (u8()) {
      case 0: return VarValue::None();
      case 1: return VarValue(uint8_t(VarValue::kType_Identifier), symbol().c_str());
      case 2: return VarValue(symbol());
      case 3: return VarValue(std::bit_cast<int32_t>(u32()));
      case 4: {
        const auto f = std::bit_cast<float>(u32());
        if (!std::isfinite(f)) fail("non-finite float");
        return VarValue(double(f));
      }
      case 5: {
        const auto b = u8();
        if (b > 1) fail("invalid boolean");
        return VarValue(b != 0);
      }
      default: fail("unknown value tag");
    }
  }
  void debug()
  {
    auto& info = result.script->debugInfo;
    info.m_flags = u8();
    if (!info.m_flags) return;
    if (info.m_flags != 1) fail("invalid debug flag");
    info.m_sourceModificationTime = number(8);
    const auto count = u16();
    for (size_t i = 0; i < count; ++i) {
      DebugInfo::DebugFunction f;
      f.objName = symbol(); f.stateName = symbol(); f.fnName = symbol(); f.type = u8();
      const auto lines = u16();
      for (size_t j = 0; j < lines; ++j) f.lineNumbers.push_back(u16());
      info.m_data.push_back(std::move(f));
    }
    const auto groups = u16();
    for (size_t i = 0; i < groups; ++i) {
      symbol(); symbol(); symbol(); u32();
      const auto properties = u16();
      for (size_t j = 0; j < properties; ++j) symbol();
    }
    const auto orders = u16();
    for (size_t i = 0; i < orders; ++i) {
      symbol(); symbol();
      const auto members = u16();
      for (size_t j = 0; j < members; ++j) symbol();
    }
  }
  FunctionInfo function()
  {
    FunctionInfo fn;
    fn.valid = true;
    fn.returnType = symbol(); type(fn.returnType);
    fn.docstring = symbol(); fn.userFlags = u32(); fn.flags = u8();
    if (fn.flags & ~3u) fail("unknown function flags");
    std::set<std::string> locals;
    const auto readLocals = [&](auto& into) {
      const auto count = u16();
      for (size_t i = 0; i < count; ++i) {
        FunctionInfo::ParamInfo local{symbol(), symbol()};
        type(local.type);
        if (!locals.insert(lower(local.name)).second) fail("duplicate local/parameter");
        into.push_back(std::move(local));
      }
    };
    readLocals(fn.params); readLocals(fn.locals);
    const auto count = u16();
    instructionTotal += count;
    if (instructionTotal > 100000) fail("instruction limit exceeded");
    constexpr std::array<unsigned, 47> counts{
      0,3,3,3,3,3,3,3,3,3,2,2,2,2,2,3,3,3,3,3,1,2,2,4,3,4,1,3,3,3,
      2,2,3,3,4,4,3,1,3,3,5,5,3,3,1,3,1};
    for (size_t i = 0; i < count; ++i) {
      FunctionCode::Instruction inst;
      inst.op = u8();
      if (inst.op >= counts.size()) fail("unknown FO4 opcode");
      if (inst.op >= 30 && inst.op != 36) unsupported.insert("array/struct instructions");
      for (size_t j = 0; j < counts[inst.op]; ++j) inst.args.push_back(value());
      if (inst.op == 23 || inst.op == 24 || inst.op == 25) {
        const auto& amount = inst.args.back();
        if (amount.GetType() != VarValue::kType_Integer) fail("invalid call argument count");
        const auto n = static_cast<int>(amount);
        if (n < 0 || n > 128) fail("call argument limit exceeded");
        for (int j = 0; j < n; ++j) inst.args.push_back(value());
      }
      const auto nameArg = [&](size_t index) {
        const auto t = inst.args[index].GetType();
        if (t != VarValue::kType_Identifier && t != VarValue::kType_String) fail("invalid name operand");
      };
      if (inst.op == 23 || inst.op == 24 || inst.op == 25 || inst.op == 28 || inst.op == 29) nameArg(0);
      if (inst.op == 25) { nameArg(1); type(static_cast<const char*>(inst.args[0])); }
      if (inst.op == 36) { nameArg(2); type(static_cast<const char*>(inst.args[2])); }
      int dest = -1;
      if ((inst.op >= 1 && inst.op <= 19) || inst.op == 27 || inst.op == 36) dest = 0;
      if (inst.op == 23 || inst.op == 25 || inst.op == 28) dest = 2;
      if (inst.op == 24) dest = 1;
      if (dest >= 0 && inst.args[dest].GetType() != VarValue::kType_Identifier) fail("destination is not a variable");
      if (inst.op >= 20 && inst.op <= 22) {
        const auto& jump = inst.args.back();
        if (jump.GetType() != VarValue::kType_Integer) fail("invalid branch offset");
        const auto target = int64_t(i) + static_cast<int>(jump);
        if (target < 0 || target > count) fail("branch outside function");
      }
      fn.code.instructions.push_back(std::move(inst));
    }
    if (fn.IsNative() && count) fail("native function has bytecode");
    return fn;
  }
  Object object()
  {
    Object obj;
    obj.NameIndex = symbol();
    const auto start = cursor;
    const auto size = u32();
    if (size < 4 || size > bytes.size() - start) fail("invalid object size");
    const auto end = start + size;
    obj.parentClassName = symbol(); type(obj.parentClassName);
    obj.docstring = symbol();
    if (u8() > 1) fail("invalid object const flag");
    obj.userFlags = u32(); obj.autoStateName = symbol();
    const auto structs = u16();
    if (structs) unsupported.insert("struct definitions");
    for (size_t i = 0; i < structs; ++i) {
      symbol(); const auto members = u16();
      for (size_t j = 0; j < members; ++j) {
        symbol(); type(symbol()); u32(); value();
        if (u8() > 1) fail("invalid struct const flag");
        symbol();
      }
    }
    std::set<std::string> names;
    const auto variables = u16();
    for (size_t i = 0; i < variables; ++i) {
      Object::VarInfo v;
      v.name = symbol(); v.typeName = symbol(); type(v.typeName);
      if (!names.insert(lower(v.name)).second) fail("duplicate variable");
      v.userFlags = u32(); v.value = value(); v.value.objectType = v.typeName;
      if (u8() > 1) fail("invalid variable const flag");
      obj.variables.push_back(std::move(v));
    }
    names.clear();
    const auto properties = u16();
    for (size_t i = 0; i < properties; ++i) {
      Object::PropInfo p;
      p.name = symbol(); p.type = symbol(); type(p.type);
      if (!names.insert(lower(p.name)).second) fail("duplicate property");
      result.properties.emplace(p.name, p.type);
      p.docstring = symbol(); p.userFlags = u32(); p.flags = u8();
      if (p.flags & ~7u) fail("unknown property flags");
      if (p.flags & 4) p.autoVarName = symbol();
      else {
        if (p.flags & 1) p.readHandler = function();
        if (p.flags & 2) p.writeHandler = function();
      }
      obj.properties.push_back(std::move(p));
    }
    names.clear();
    const auto states = u16();
    for (size_t i = 0; i < states; ++i) {
      Object::StateInfo state;
      state.name = symbol();
      if (!names.insert(lower(state.name)).second) fail("duplicate state");
      const auto functions = u16();
      std::set<std::string> functionNames;
      for (size_t j = 0; j < functions; ++j) {
        Object::StateInfo::StateFunction f;
        f.name = symbol();
        if (!functionNames.insert(lower(f.name)).second) fail("duplicate function");
        f.function = function();
        state.functions.push_back(std::move(f));
      }
      obj.states.push_back(std::move(state));
    }
    if (cursor != end) fail("object size mismatch");
    return obj;
  }
public:
  explicit CheckedReader(std::span<const uint8_t> input) : bytes(input) {}
  Fo4Script read()
  {
    if (bytes.size() > 8 * 1024 * 1024) fail("file exceeds 8 MiB limit");
    result.script = std::make_shared<PexScript>();
    auto& h = result.script->header;
    h.Signature = u32(); h.VerMajor = u8(); h.VerMinor = u8(); h.GameID = u16();
    if (h.Signature != 0xFA57C0DE || h.VerMajor != 3 || h.VerMinor != 9 || h.GameID != 2)
      fail("expected Fallout 4 PEX 3.9 / game 2");
    h.BuildTime = number(8);
    string(); result.script->user = string(); result.script->machine = string();
    const auto count = u16();
    std::vector<std::string> strings;
    strings.reserve(count);
    for (size_t i = 0; i < count; ++i) strings.push_back(string());
    result.script->stringTable.SetStorage(std::move(strings));
    debug();
    const auto flags = u16();
    for (size_t i = 0; i < flags; ++i) {
      UserFlag f; f.name = symbol(); f.idx = u8();
      if (f.idx >= 32) fail("invalid user flag bit");
      result.script->userFlagTable.push_back(std::move(f));
    }
    if (u16() != 1) fail("expected one script object");
    result.script->objectTable.push_back(object());
    if (cursor != bytes.size()) fail("unexpected trailing data");
    result.script->source = result.script->objectTable.front().NameIndex;
    deps.erase(lower(result.script->source));
    result.dependencies.assign(deps.begin(), deps.end());
    result.limitations.assign(unsupported.begin(), unsupported.end());
    return std::move(result);
  }
};
}

Fo4Script ReadFo4Pex(std::span<const uint8_t> bytes) { return CheckedReader(bytes).read(); }
