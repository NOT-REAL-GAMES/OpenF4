Scriptname OpenF4VmFixture extends OpenF4VmBase

Int Property Counter = 0 Auto
Float Property Impulse = 1.0 Auto
ObjectReference Property Target Auto

Int Property Computed
    Int Function Get()
        return Counter * 2
    EndFunction
    Function Set(Int value)
        Counter = value / 2
    EndFunction
EndProperty

Int Property LatentComputed
    Int Function Get()
        return Counter
    EndFunction
    Function Set(Int value)
        Utility.Wait(0.0)
        Counter = value
    EndFunction
EndProperty

Event OnInit()
    Counter += 1
EndEvent

Event OnLoad()
    SetMotionType(Motion_Dynamic, true)
    ApplyHavokImpulse(0.0, 0.0, 1.0, Impulse)
    Counter = Parent.Twice(Counter + 2)
    Utility.Wait(0.1)
    Counter += 10
    GotoState("Ready")
EndEvent

State Ready
    Event OnLoad()
        Counter += 100
    EndEvent
EndState

Function SetComputed(Int value)
    Computed = value
EndFunction

Function CheckReference()
    If Target != None && Target == Self && Target is ObjectReference
        Counter += 5
    EndIf
EndFunction

Function Spin()
    While true
        Counter += 1
    EndWhile
EndFunction

Function MissingNative()
    GetPositionX()
    Counter = -999
EndFunction

Function CheckParent()
    Counter = Parent.Sum(17, 25)
EndFunction

Function CheckLatentSetter()
    Int input = 21
    LatentComputed = input
    Counter += input
EndFunction

Function WaitLoop()
    Counter = 0
    While Counter < 4096
        Utility.Wait(0.0)
        Counter += 1
    EndWhile
EndFunction

Function LatentRecursion()
    Utility.Wait(0.0)
    LatentRecursion()
EndFunction

Function GrowString()
    String text = "a"
    While true
        text += text
    EndWhile
EndFunction

Function DivideByZero()
    Int zero = 0
    Counter = 1 / zero
EndFunction

Function ArithmeticEdges()
    Int maximum = 2147483647
    Int minimum = maximum + 1
    Counter = minimum / -1
    Counter = Counter % -1
    Counter += 0 / 2
    Float fraction = 0.25 * 2.0
    If fraction == 0.5
        Counter += 1
    EndIf
EndFunction
