# Purpose: Research and upgrade management for Zerg B2GM build
# Key Decisions: Priority order: Ling Speed → +1/+1 → Hydra Grooved Spines → +2/+2 → +3/+3
# Limitations: No dynamic upgrade switching based on opponent composition yet

from sc2.ids.upgrade_id import UpgradeId as UpgradeID
from sc2.ids.unit_typeid import UnitTypeId as UnitID

# Upgrade priority order (lower = higher priority)
UPGRADE_PRIORITY: list[tuple[UpgradeID, callable]] = [
    # Ling Speed is handled by build runner opening, but include as fallback
    (UpgradeID.ZERGLINGMOVEMENTSPEED, lambda ai: ai.can_afford(UpgradeID.ZERGLINGMOVEMENTSPEED)),
    # +1 Melee and +1 Missile (double evo)
    (UpgradeID.ZERGMELEEWEAPONSLEVEL1, lambda ai: ai.can_afford(UpgradeID.ZERGMELEEWEAPONSLEVEL1)),
    (UpgradeID.ZERGMISSILEWEAPONSLEVEL1, lambda ai: ai.can_afford(UpgradeID.ZERGMISSILEWEAPONSLEVEL1)),
    # Hydra Speed after hydra den
    (UpgradeID.EVOLVEGROOVEDSPINES, lambda ai: ai.structures(UnitID.HYDRALISKDEN).ready.exists and ai.can_afford(UpgradeID.EVOLVEGROOVEDSPINES)),
    # +2 upgrades
    (UpgradeID.ZERGMELEEWEAPONSLEVEL2, lambda ai: ai.can_afford(UpgradeID.ZERGMELEEWEAPONSLEVEL2)),
    (UpgradeID.ZERGMISSILEWEAPONSLEVEL2, lambda ai: ai.can_afford(UpgradeID.ZERGMISSILEWEAPONSLEVEL2)),
    # +3 upgrades
    (UpgradeID.ZERGMELEEWEAPONSLEVEL3, lambda ai: ai.can_afford(UpgradeID.ZERGMELEEWEAPONSLEVEL3)),
    (UpgradeID.ZERGMISSILEWEAPONSLEVEL3, lambda ai: ai.can_afford(UpgradeID.ZERGMISSILEWEAPONSLEVEL3)),
    # Bane Speed (conditional, lower priority)
    (UpgradeID.CENTRIFICALHOOKS, lambda ai: ai.can_afford(UpgradeID.CENTRIFICALHOOKS)),
]


def research_upgrades(ai) -> None:
    """Research the next available upgrade in priority order.

    Only researches one upgrade per frame to avoid starving production.
    Uses `already_pending_upgrade` to avoid duplicate research.
    """
    for upgrade_id, condition_fn in UPGRADE_PRIORITY:
        if ai.already_pending_upgrade(upgrade_id) > 0:
            continue
        if not condition_fn(ai):
            continue

        # Find a structure that can research this upgrade
        # python-sc2's research method handles finding the right structure
        if ai.can_afford(upgrade_id):
            ai.research(upgrade_id)
            break  # Only one upgrade per frame
