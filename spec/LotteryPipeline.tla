------------------------- MODULE LotteryPipeline -------------------------
EXTENDS Integers, FiniteSets, TLC

CONSTANTS 
    MaxDays,
    TestCutoffDay,
    TicketCost,
    ThresholdPositiveEV

VARIABLES 
    phase,
    loadedDays,
    markovState,
    zscoreState,
    picksCount,
    bankroll,
    futureLeak,
    jackpotValue,
    arbitrageMode

vars == <<phase, loadedDays, markovState, zscoreState, picksCount, bankroll, futureLeak, jackpotValue, arbitrageMode>>

Init ==
    /\ phase = "IDLE"
    /\ loadedDays = {}
    /\ markovState = "UNINITIALIZED"
    /\ zscoreState = "PENDING"
    /\ picksCount = 0
    /\ bankroll \in {0, 10}
    /\ futureLeak = FALSE
    /\ arbitrageMode \in {FALSE, TRUE}
    /\ jackpotValue \in {100, 250}

IngestData ==
    /\ phase = "IDLE"
    /\ loadedDays' = 1..MaxDays
    /\ phase' = "INGESTED"
    /\ UNCHANGED <<markovState, zscoreState, picksCount, bankroll, futureLeak, jackpotValue, arbitrageMode>>

ApplyCutoff ==
    /\ phase = "INGESTED"
    /\ loadedDays' = {d \in loadedDays : d < TestCutoffDay}
    /\ phase' = "CUTOFF_VERIFIED"
    /\ futureLeak' = (\E d \in {d \in loadedDays : d < TestCutoffDay} : d >= TestCutoffDay)
    /\ UNCHANGED <<markovState, zscoreState, picksCount, bankroll, jackpotValue, arbitrageMode>>

ExtractFeatures ==
    /\ phase = "CUTOFF_VERIFIED"
    /\ markovState' = "CALCULATED"
    /\ zscoreState' = "COMPUTED"
    /\ phase' = "FEATURES_READY"
    /\ UNCHANGED <<loadedDays, picksCount, bankroll, futureLeak, jackpotValue, arbitrageMode>>

EvaluateArbitrage ==
    /\ phase = "FEATURES_READY"
    /\ arbitrageMode = TRUE
    /\ jackpotValue < ThresholdPositiveEV
    /\ phase' = "DORMANT_SKIP"
    /\ UNCHANGED <<loadedDays, markovState, zscoreState, picksCount, bankroll, futureLeak, jackpotValue, arbitrageMode>>

GeneratePrediction ==
    /\ phase = "FEATURES_READY"
    /\ (arbitrageMode = FALSE \/ jackpotValue >= ThresholdPositiveEV)
    /\ bankroll >= TicketCost
    /\ picksCount' = 6
    /\ bankroll' = bankroll - TicketCost
    /\ phase' = "PREDICTED"
    /\ UNCHANGED <<loadedDays, markovState, zscoreState, futureLeak, jackpotValue, arbitrageMode>>

HaltInsolvent ==
    /\ phase = "FEATURES_READY"
    /\ (arbitrageMode = FALSE \/ jackpotValue >= ThresholdPositiveEV)
    /\ bankroll < TicketCost
    /\ phase' = "HALTED_INSOLVENT"
    /\ UNCHANGED <<loadedDays, markovState, zscoreState, picksCount, bankroll, futureLeak, jackpotValue, arbitrageMode>>

EvaluateDraw ==
    /\ phase = "PREDICTED"
    /\ \/ /\ bankroll' = bankroll + 5
       \/ /\ bankroll' = bankroll
    /\ phase' = "EVALUATED"
    /\ UNCHANGED <<loadedDays, markovState, zscoreState, picksCount, futureLeak, jackpotValue, arbitrageMode>>

Done ==
    /\ phase \in {"EVALUATED", "HALTED_INSOLVENT", "DORMANT_SKIP"}
    /\ UNCHANGED vars

Next ==
    \/ IngestData
    \/ ApplyCutoff
    \/ ExtractFeatures
    \/ EvaluateArbitrage
    \/ GeneratePrediction
    \/ HaltInsolvent
    \/ EvaluateDraw
    \/ Done

ZeroFutureLeakage ==
    (phase \in {"CUTOFF_VERIFIED", "FEATURES_READY", "PREDICTED", "EVALUATED", "HALTED_INSOLVENT", "DORMANT_SKIP"}) =>
        (\A d \in loadedDays : d < TestCutoffDay)

BankrollSolvency ==
    bankroll >= 0

ValidPickSize ==
    /\ (phase \in {"PREDICTED", "EVALUATED"}) => (picksCount = 6)
    /\ (phase = "DORMANT_SKIP") => (picksCount = 0)

FeaturesSynchronized ==
    (phase \in {"FEATURES_READY", "PREDICTED", "EVALUATED", "HALTED_INSOLVENT", "DORMANT_SKIP"}) =>
        (markovState = "CALCULATED" /\ zscoreState = "COMPUTED")

ArbitragePositiveEV ==
    /\ ((phase \in {"PREDICTED", "EVALUATED"}) /\ arbitrageMode = TRUE) =>
           (jackpotValue >= ThresholdPositiveEV)
    /\ (phase = "DORMANT_SKIP") =>
           (arbitrageMode = TRUE /\ jackpotValue < ThresholdPositiveEV)

KellySolvencyBound ==
    /\ bankroll >= 0
    /\ (phase \in {"PREDICTED", "EVALUATED"} => (bankroll >= 0 /\ picksCount = 6))
    /\ (phase \in {"HALTED_INSOLVENT", "DORMANT_SKIP"} => (picksCount = 0))

=============================================================================
