# Purpose: Army composition definitions for SpawnController / ProductionController
# Key Decisions: Early game roach/ling/bane/ravager, mid-game adds infestor,
#   hydra only when air threats detected. Composition switches on economy
#   (drone count) with time as fallback, not time alone.
# Limitations: No dynamic composition switching beyond air/economy detection yet

from sc2.ids.unit_typeid import UnitTypeId as UnitID

# Early game (< 6 min or < 36 drones): roach/ling/bane/ravager
# Queens excluded — managed separately via inject/creep/defense
# Banelings provide AOE for early mass-light encounters
EARLY_COMP: dict[UnitID, dict] = {
    UnitID.ROACH: {"proportion": 0.48, "priority": 1},
    UnitID.ZERGLING: {"proportion": 0.25, "priority": 2},
    UnitID.BANELING: {"proportion": 0.15, "priority": 3},
    UnitID.RAVAGER: {"proportion": 0.12, "priority": 4},
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

# Economy thresholds for composition switching
# 36 drones = gas phase 2, enough economy to support infestors
MID_GAME_DRONES: int = 36
# Time fallback: switch at 6 min even if drone count is low (e.g. heavy pressure)
MID_GAME_TIME: float = 360.0


def get_army_comp(
    time: float, air_threat: bool = False, drone_count: int = 0
) -> dict[UnitID, dict]:
    """Return the appropriate army composition based on game state.

    Switches to mid-game comp when either the economy is ready (36+ drones)
    or enough time has passed (6 min). Air threat overrides to anti-air comp.

    Args:
        time: Current game time in seconds.
        air_threat: True if air signs detected (stargate, starport techlab,
            fusion core, fleet beacon, or visible air units).
        drone_count: Current number of workers (supply_workers).
    """
    if air_threat:
        return ANTI_AIR_COMP

    if time < MID_GAME_TIME and drone_count < MID_GAME_DRONES:
        return EARLY_COMP

    return MID_COMP
