# SkyMP Papyrus VM fork

Source: https://github.com/skyrim-multiplayer/skymp
Pinned commit: `2849e676fe008925384b26f34476e382a30bd877`.

Only the MIT `papyrus-vm` library and required MIT `viet` helpers are included.
`support/overlay_ports/makeid/MakeID.h` carries its upstream public-domain notice.
The original Skyrim reader is retained for provenance but is not built. OpenF4 adds a checked FO4 reader, bounded execution and explicit error propagation to the existing interpreter. No Bethesda scripts or SkyMP server/client code are included.

Supported execution is deliberately limited to project-object scripts. It is not a complete Fallout 4 VM or savegame implementation. Unknown natives and unsupported instructions fail explicitly.

The host dispatches at most 32 events/ready continuations per tick, with a fresh
instruction cap for each dispatch (default 10,000; maximum 100,000). An over-budget
stack faults its owning object, rather than yielding an arbitrary partial opcode.
Logical call depth is capped at 128 across latent waits; strings at 64 KiB; queued
events at 4,096; suspended waits at 1,024. `Utility.Wait` uses the host's unpaused
delta time, and repeated waits retain one completion per logical function.

The new reader structurally inspects FO4 arrays, structs, and Var values for export
dependency analysis, but runtime attachment rejects those unsupported features.
Native ScriptObject/Form/ObjectReference/Utility/Debug ancestry is supplied as
minimal host metadata, not loaded from Bethesda's full base-script implementation.
Only host-registered natives execute. Session variables are not a Fallout savegame.
