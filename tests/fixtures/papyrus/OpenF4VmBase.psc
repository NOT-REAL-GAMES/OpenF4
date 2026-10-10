Scriptname OpenF4VmBase extends ObjectReference

Int Property BaseValue = 3 Auto

Int Function Twice(Int value)
    return value * 2
EndFunction

Int Function Sum(Int a, Int b)
    return a + b
EndFunction

Event OnUnload()
    BaseValue += 1
EndEvent
