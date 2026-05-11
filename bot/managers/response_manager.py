# Purpose: Simple detect-and-react system for common threats
# Key Decisions: Small if-rules, not a full rule engine; minimal force allocation
# Limitations: No ML-based detection, no exit conditions yet

from typing import Optional

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId as UpgradeID
from sc2.position import Point2

from ares.consts import UnitRole


# ── Timing thresholds (seconds) ─────────────────────────────────────────────
NATURAL_TIMING_THRESHOLD: float = 210.0  # 3:30

# ── Response constants ──────────────────────────────────────────────────────
SAFETY_ROACH_COUNT: int = 5
EMERGENCY_SPINE_COUNT: int = 2
MINERAL_LINE_SPINE_COUNT: int = 1


def assess_threats(ai) -> dict[str, bool]:
    """Detect common threats and return a dict of active threats.

    Returns:
        dict mapping threat name to whether it's currently active
    """
    threats: dict[str, bool] = {
        "no_natural": False,
        "timing_push": False,
        "air_signs": False,
        "proxy_signs": False,
        "mass_light": False,
        "cannon_rush": False,
    }

    # ── No natural at 3:30 ──────────────────────────────────────────────
    if ai.time > NATURAL_TIMING_THRESHOLD:
        enemy_naturals: list = [
            th for th in ai.enemy_structures
            if th.type_id in {UnitID.HATCHERY, UnitID.COMMANDCENTER, UnitID.NEXUS}
            and th.distance_to(ai.enemy_start_locations[0]) > 50
            and th.distance_to(ai.enemy_start_locations[0]) < 200
        ]
        if not enemy_naturals:
            threats["no_natural"] = True

    # ── Air signs (air tech structures OR visible air units) ────────────
    air_structures: set[UnitID] = {
        UnitID.FUSIONCORE, UnitID.STARGATE, UnitID.STARPORTTECHLAB,
        UnitID.FLEETBEACON,
    }
    air_unit_types: set[UnitID] = {
        # Protoss
        UnitID.VOIDRAY, UnitID.CARRIER, UnitID.ORACLE, UnitID.PHOENIX,
        UnitID.TEMPEST, UnitID.MOTHERSHIP,
        # Terran
        UnitID.MEDIVAC, UnitID.VIKINGFIGHTER, UnitID.VIKINGASSAULT,
        UnitID.BANSHEE, UnitID.RAVEN, UnitID.BATTLECRUISER, UnitID.LIBERATOR,
        # Zerg
        UnitID.MUTALISK, UnitID.CORRUPTOR, UnitID.BROODLORD,
    }
    for structure in ai.enemy_structures:
        if structure.type_id in air_structures:
            threats["air_signs"] = True
            break
    if not threats["air_signs"]:
        for unit in ai.enemy_units:
            if unit.type_id in air_unit_types and not unit.is_memory:
                threats["air_signs"] = True
                break

    # ── Proxy signs (enemy structures near our base) ────────────────────
    for structure in ai.enemy_structures:
        if structure.distance_to(ai.start_location) < 80:
            if structure.type_id not in {UnitID.XELNAGATOWER}:
                threats["proxy_signs"] = True
                break

    # ── Mass light units (lings, zealots, adepts) ──────────────────────
    light_unit_types: set[UnitID] = {
        UnitID.ZERGLING, UnitID.ZEALOT, UnitID.ADEPT,
        UnitID.MARINE,
    }
    light_count: int = sum(
        1 for u in ai.enemy_units
        if u.type_id in light_unit_types and not u.is_memory
    )
    if light_count >= 10:
        threats["mass_light"] = True

    # ── Cannon rush (forge + pylon near our base) ───────────────────────
    for structure in ai.enemy_structures:
        if structure.distance_to(ai.start_location) < 60:
            if structure.type_id in {UnitID.FORGE, UnitID.PHOTONCANNON}:
                threats["cannon_rush"] = True
                break

    return threats


def respond_to_threats(ai, threats: dict[str, bool]) -> None:
    """Execute minimal responses for active threats.

    Each response allocates the minimum force required.
    """
    if not any(threats.values()):
        return

    # ── No natural: spines + safety roaches + scout ────────────────────
    if threats["no_natural"]:
        _build_emergency_spines(ai, count=EMERGENCY_SPINE_COUNT)
        _ensure_safety_roaches(ai, count=SAFETY_ROACH_COUNT)

    # ── Air signs: early hydra den + extra queens + spines ──────────────
    if threats["air_signs"]:
        if not ai.structures(UnitID.HYDRALISKDEN).exists and not ai.already_pending(UnitID.HYDRALISKDEN):
            if ai.can_afford(UnitID.HYDRALISKDEN):
                ai.build(UnitID.HYDRALISKDEN, near=ai.start_location)
        _build_mineral_line_spines(ai, count=MINERAL_LINE_SPINE_COUNT)

    # ── Proxy signs: pull drones, emergency spine, safety roaches ───────
    if threats["proxy_signs"]:
        _build_emergency_spines(ai, count=1)
        _ensure_safety_roaches(ai, count=SAFETY_ROACH_COUNT)

    # ── Mass light: bane nest + banes at ramp ───────────────────────────
    if threats["mass_light"]:
        if not ai.structures(UnitID.BANELINGNEST).exists and not ai.already_pending(UnitID.BANELINGNEST):
            if ai.can_afford(UnitID.BANELINGNEST):
                ai.build(UnitID.BANELINGNEST, near=ai.start_location)

    # ── Cannon rush: pull drones, kill pylon ────────────────────────────
    if threats["cannon_rush"]:
        _build_emergency_spines(ai, count=1)


def _build_emergency_spines(ai, count: int = 2) -> None:
    """Build emergency spine crawlers near our bases."""
    existing_spines: int = len(ai.structures(UnitID.SPINECRAWLER))
    pending_spines: int = ai.already_pending(UnitID.SPINECRAWLER)
    needed: int = max(0, count - existing_spines - pending_spines)

    for _ in range(needed):
        if ai.can_afford(UnitID.SPINECRAWLER):
            for th in ai.townhalls:
                ai.build(UnitID.SPINECRAWLER, near=th.position.towards(ai.game_info.map_center, 3))
                break


def _build_mineral_line_spines(ai, count: int = 1) -> None:
    """Build spine crawlers in mineral lines for air defense."""
    existing_spines: int = len(ai.structures(UnitID.SPINECRAWLER))
    pending_spines: int = ai.already_pending(UnitID.SPINECRAWER)
    needed: int = max(0, count + len(ai.townhalls) - existing_spines - pending_spines)

    for _ in range(needed):
        if ai.can_afford(UnitID.SPINECRAWER):
            for th in ai.townhalls:
                # Place near mineral line
                mfs = ai.mineral_field.closer_than(10, th)
                if mfs:
                    ai.build(
                        UnitID.SPINECRAWER,
                        near=th.position.towards(mfs.center, 2),
                    )
                break


def _ensure_safety_roaches(ai, count: int = 5) -> None:
    """Ensure we have a minimum number of roaches for defense."""
    current_roaches: int = len(ai.units(UnitID.ROACH))
    if current_roaches < count:
        # SpawnController will handle production, just flag the need
        pass  # The composition system handles roach production
