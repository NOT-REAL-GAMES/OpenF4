# Papyrus VM fixtures

The sibling PSC files are OpenF4-authored test scripts. `compiled.hpp` embeds
their Fallout 4 PEX output so ordinary builds/tests do not require a Creation Kit
installation. No Bethesda source or compiled scripts are bundled.

They were compiled with Fallout 4 Papyrus Compiler 2.8.0.4, using each PSC's
absolute path, `-i=<this directory>;<Fallout4>/Data/Scripts/Source/Base`,
`-f=<Fallout4>/Data/Scripts/Source/Base/Institute_Papyrus_Flags.flg`, and a temporary
`-o` directory. The resulting PEX files are embedded as byte arrays; changing a
PSC requires recompiling and updating its corresponding array.

Coverage includes scalar/reference properties, inheritance and multi-argument
parent calls, functions and states, native dispatch, events, latent property
setters, repeated waits, latent recursion limits, unsupported arrays, numeric
edge cases and bounded string growth. Tests also truncate/corrupt these bytes.
