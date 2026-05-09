# Purpose: Army composition definitions for SpawnController / ProductionController
# Key Decisions: Early game roach/ling/ravager, mid-game adds infestor,
#   hydra only when air threats detected (response_manager signals this)
# Limitations: No dynamic composition switching beyond air detection yet

from sc2.ids.unit_typeid import UnitTypeId as UnitID

# Early game (< 6 min): roach/ling/ravager focused
# Queens excluded — managed separately via inject/creep/defense
EARLY_COMP: dict[UnitID, dict] = {
    UnitID.ROACH: {"proportion": 0.58, "priority": 1},
    UnitID.ZERGLING: {"proportion": 0.30, "priority": 2},
    UnitID.RAVAGER: {"proportion": 0.12, "priority": 3},
}

# Mid game (6+ min): roach/ravager/ling/infestor — no hydra unless air detected
MID_COMP: dict[UnitID, dict] = {
    UnitID.ROACH: {"proportion": 0.45, "priority": 1},
    UnitID.RAVAGER: {"proportion": 0.20, "priority": 2},
    UnitID.ZERGLING: {"proportion": 0.25, "priority": 3},
    UnitID.INFESTOR: {"proportion": 0.10, "priority": 4},
}

# Anti-air variant: heavy hydra for air-heavy opponents
# Hydralisk Den is built reactively by response_manager when air_signs detected
ANTI_AIR_COMP: dict[UnitID, dict] = {
    UnitID.ROACH: {"proportion": 0.30, "priority": 1},
    UnitID.HYDRALISK: {"proportion": 0.35, "priority": 2},
    UnitID.RAVAGER: {"proportion": 0.10, "priority": 3},
    UnitID.ZERGLING: {"proportion": 0.20, "priority": 4},
    UnitID.INFESTOR: {"proportion": 0.05, "priority": 5},
}

# Time threshold for switching from early to mid comp (seconds)
MID_GAME_TIME: float = 360.0


def get_army_comp(time: float, air_threat: bool = False) -> dict[UnitID, dict]:
    """Return the appropriate army composition based on game time and air threat.

    Args:
        time: Current game time in seconds.
        air_threat: True if response_manager detected air signs (stargate,
            starport techlab, fusion core, fleet beacon).
    """
    if time < MID_GAME_TIME:
        return EARLY_COMP

    if air_threat:
        return ANTI_AIR_COMP

    return MID_COMP
