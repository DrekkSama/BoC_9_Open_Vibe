# Purpose: Centralized macro management — workers, supply, gas, spawning,
#   tech, upgrades, expansions, and reactive building.
# Key Decisions: All MacroPlan behaviors live here. Research and response
#   logic folded in from their old standalone modules. The attack_target
#   property is also owned here since it's a macro-level decision.
# Limitations: No nydus network support yet, no dynamic composition
#   switching beyond air detection.

from cython_extensions import cy_closest_to
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId as UpgradeID
from sc2.position import Point2

from ares import AresBot
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    Mining,
    SpawnController,
    TechUp,
    UpgradeController,
)

from bot.compositions import get_army_comp

# ── Constants ────────────────────────────────────────────────────────────────
BEGIN_ATTACK_SUPPLY: float = 6.0
MID_GAME_TIME: float = 360.0
NATURAL_TIMING_THRESHOLD: float = 210.0  # 3:30

# Upgrade priority order (lower = higher priority)
UPGRADE_PRIORITY: list[UpgradeID] = [
    UpgradeID.ZERGLINGMOVEMENTSPEED,
    UpgradeID.ZERGMELEEWEAPONSLEVEL1,
    UpgradeID.ZERGMISSILEWEAPONSLEVEL1,
    UpgradeID.EVOLVEGROOVEDSPINES,
    UpgradeID.ZERGMELEEWEAPONSLEVEL2,
    UpgradeID.ZERGMISSILEWEAPONSLEVEL2,
    UpgradeID.ZERGMELEEWEAPONSLEVEL3,
    UpgradeID.ZERGMISSILEWEAPONSLEVEL3,
    UpgradeID.CENTRIFICALHOOKS,
    UpgradeID.OVERLORDSPEED,
]

# Air-detection structure types
AIR_STRUCTURES: set[UnitID] = {
    UnitID.FUSIONCORE, UnitID.STARGATE, UnitID.STARPORTTECHLAB,
    UnitID.FLEETBEACON,
}

# Light unit types for mass detection
LIGHT_UNIT_TYPES: set[UnitID] = {
    UnitID.ZERGLING, UnitID.ZEALOT, UnitID.ADEPT, UnitID.MARINE,
}

# Response constants
SAFETY_ROACH_COUNT: int = 5
EMERGENCY_SPINE_COUNT: int = 2
MINERAL_LINE_SPINE_COUNT: int = 1


class MacroManager:
    """Owns all macro decisions: economy, production, tech, upgrades, responses."""

    def __init__(self, ai: AresBot) -> None:
        self._ai: AresBot = ai
        self._commenced_attack: bool = False
        self._threats: dict[str, bool] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def attack_target(self) -> Point2:
        """Determine where the army should attack."""
        if self._ai.enemy_structures:
            return cy_closest_to(
                self._ai.start_location, self._ai.enemy_structures
            ).position
        if self._ai.time < 240.0:
            return self._ai.enemy_start_locations[0]
        # Late game: search expansions
        for expand_pos in self._ai.expansion_locations_list:
            if not self._ai.is_visible(expand_pos):
                return expand_pos
        return self._ai.enemy_start_locations[0]

    @property
    def commenced_attack(self) -> bool:
        return self._commenced_attack

    def start_attack(self) -> None:
        """One-way gate: mark that the army should begin attacking."""
        self._commenced_attack = True

    @property
    def threats(self) -> dict[str, bool]:
        return self._threats

    def update(self) -> None:
        """Run every frame: economy, production, tech, upgrades, responses."""
        # Always mine
        self._ai.register_behavior(Mining())

        # Only run dynamic macro after the opening build completes
        if not self._ai.build_order_runner.build_completed:
            return

        self._assess_threats()
        self._do_macro_plan()
        self._research_upgrades()
        self._respond_to_threats()

    # ── Macro Plan ──────────────────────────────────────────────────────────

    def _do_macro_plan(self) -> None:
        """Build the main MacroPlan: supply, workers, gas, spawning, tech, expand."""
        macro_plan: MacroPlan = MacroPlan()
        structure_dict: dict = self._ai.mediator.get_own_structures_dict

        # Supply
        macro_plan.add(AutoSupply(base_location=self._ai.start_location))

        # Workers — scale with base count, cap at 70
        max_workers: int = min(70, len(self._ai.townhalls) * 22)
        idle_ths: list = [
            th for th in self._ai.townhalls if th.is_ready and th.is_idle
        ]
        if not idle_ths or (
            self._ai.supply_workers < 30
            and not self._threats.get("rush_detected", False)
        ):
            macro_plan.add(BuildWorkers(to_count=max_workers))

        # Gas — scale with worker count
        if self._ai.supply_workers > 20:
            gas_count: int = (
                3 if self._ai.supply_workers > 70
                else (2 if self._ai.supply_workers > 40 else 1)
            )
            macro_plan.add(GasBuildingController(to_count=gas_count))

        # Spawning — use composition from compositions.py
        army_comp: dict[UnitID, dict] = get_army_comp(
            self._ai.time,
            air_threat=self._threats.get("air_signs", False),
        )
        macro_plan.add(SpawnController(army_comp))

        # Tech — Lair when we have enough queens and gas
        lair_tech: bool = (
            len(structure_dict[UnitID.LAIR]) > 0
            or len(structure_dict[UnitID.HIVE]) > 0
        )
        if (
            self._ai.vespene >= 100
            and not lair_tech
            and len(self._ai.mediator.get_own_army_dict[UnitID.QUEEN]) >= 4
        ):
            macro_plan.add(
                TechUp(desired_tech=UnitID.LAIR, base_location=self._ai.start_location)
            )

        # Hive at high supply
        if (
            self._ai.supply_used > 170.0
            and len(structure_dict.get(UnitID.HIVE, [])) == 0
        ):
            macro_plan.add(
                TechUp(desired_tech=UnitID.HIVE, base_location=self._ai.start_location)
            )

        # Upgrades — only when we have gas to spare
        if self._upgrades_enabled:
            macro_plan.add(
                UpgradeController(
                    upgrade_list=self._required_upgrades,
                    base_location=self._ai.start_location,
                )
            )

        # Expansions
        if self._threats.get("rush_detected", False) and self._ai.supply_army < 16:
            max_pending: int = 0
        else:
            max_pending = 3 if self._ai.minerals < 1250 else 4
        macro_plan.add(ExpansionController(to_count=99, max_pending=max_pending))

        self._ai.register_behavior(macro_plan)

    # ── Upgrades ────────────────────────────────────────────────────────────

    @property
    def _required_upgrades(self) -> list[UpgradeID]:
        """Upgrade list for UpgradeController."""
        return UPGRADE_PRIORITY

    @property
    def _upgrades_enabled(self) -> bool:
        """Only research upgrades when we have gas to spare."""
        if self._ai.supply_workers < 36:
            return False
        return (self._ai.vespene > 95) or (
            self._ai.minerals > 500 and self._ai.vespene > 350
        )

    def _research_upgrades(self) -> None:
        """Research the next available upgrade in priority order.

        Only one upgrade per frame to avoid starving production.
        Uses `already_pending_upgrade` to avoid duplicate research.
        """
        for upgrade_id in UPGRADE_PRIORITY:
            if self._ai.already_pending_upgrade(upgrade_id) > 0:
                continue
            if not self._ai.can_afford(upgrade_id):
                continue
            self._ai.research(upgrade_id)
            break  # Only one upgrade per frame

    # ── Threat Assessment ──────────────────────────────────────────────────

    def _assess_threats(self) -> None:
        """Detect common threats and update internal threat dict."""
        self._threats = {
            "no_natural": False,
            "timing_push": False,
            "air_signs": False,
            "proxy_signs": False,
            "mass_light": False,
            "cannon_rush": False,
            "rush_detected": False,
        }

        ai = self._ai

        # Rush detection (from ARES mediator)
        if ai.mediator.get_did_enemy_rush:
            self._threats["rush_detected"] = True

        # No natural at 3:30
        if ai.time > NATURAL_TIMING_THRESHOLD:
            enemy_naturals: list = [
                th for th in ai.enemy_structures
                if th.type_id in {UnitID.HATCHERY, UnitID.COMMANDCENTER, UnitID.NEXUS}
                and 50 < th.distance_to(ai.enemy_start_locations[0]) < 200
            ]
            if not enemy_naturals:
                self._threats["no_natural"] = True

        # Air signs
        for structure in ai.enemy_structures:
            if structure.type_id in AIR_STRUCTURES:
                self._threats["air_signs"] = True
                break

        # Proxy signs
        for structure in ai.enemy_structures:
            if (
                structure.distance_to(ai.start_location) < 80
                and structure.type_id != UnitID.XELNAGATOWER
            ):
                self._threats["proxy_signs"] = True
                break

        # Mass light units
        light_count: int = sum(
            1 for u in ai.enemy_units
            if u.type_id in LIGHT_UNIT_TYPES and not u.is_memory
        )
        if light_count >= 10:
            self._threats["mass_light"] = True

        # Cannon rush
        for structure in ai.enemy_structures:
            if structure.distance_to(ai.start_location) < 60:
                if structure.type_id in {UnitID.FORGE, UnitID.PHOTONCANNON}:
                    self._threats["cannon_rush"] = True
                    break

    # ── Threat Responses ────────────────────────────────────────────────────

    def _respond_to_threats(self) -> None:
        """Execute minimal responses for active threats."""
        if not any(self._threats.values()):
            return

        ai = self._ai

        # No natural: spines + safety roaches
        if self._threats["no_natural"]:
            self._build_emergency_spines(count=EMERGENCY_SPINE_COUNT)

        # Air signs: hydra den + mineral line spines
        if self._threats["air_signs"]:
            if (
                not ai.structures(UnitID.HYDRALISKDEN).exists
                and not ai.already_pending(UnitID.HYDRALISKDEN)
                and ai.can_afford(UnitID.HYDRALISKDEN)
            ):
                ai.build(UnitID.HYDRALISKDEN, near=ai.start_location)
            self._build_mineral_line_spines(count=MINERAL_LINE_SPINE_COUNT)

        # Proxy signs: spine + safety roaches
        if self._threats["proxy_signs"]:
            self._build_emergency_spines(count=1)

        # Mass light: bane nest
        if self._threats["mass_light"]:
            if (
                not ai.structures(UnitID.BANELINGNEST).exists
                and not ai.already_pending(UnitID.BANELINGNEST)
                and ai.can_afford(UnitID.BANELINGNEST)
            ):
                ai.build(UnitID.BANELINGNEST, near=ai.start_location)

        # Cannon rush: spine
        if self._threats["cannon_rush"]:
            self._build_emergency_spines(count=1)

    # ── Building Helpers ────────────────────────────────────────────────────

    def _build_emergency_spines(self, count: int = 2) -> None:
        """Build emergency spine crawlers near our bases."""
        ai = self._ai
        existing: int = len(ai.structures(UnitID.SPINECRAWLER))
        pending: int = ai.already_pending(UnitID.SPINECRAWLER)
        needed: int = max(0, count - existing - pending)

        for _ in range(needed):
            if ai.can_afford(UnitID.SPINECRAWLER):
                for th in ai.townhalls:
                    ai.build(
                        UnitID.SPINECRAWLER,
                        near=th.position.towards(ai.game_info.map_center, 3),
                    )
                    break

    def _build_mineral_line_spines(self, count: int = 1) -> None:
        """Build spine crawlers in mineral lines for air defense."""
        ai = self._ai
        existing: int = len(ai.structures(UnitID.SPINECRAWLER))
        pending: int = ai.already_pending(UnitID.SPINECRAWLER)
        needed: int = max(0, count + len(ai.townhalls) - existing - pending)

        for _ in range(needed):
            if ai.can_afford(UnitID.SPINECRAWLER):
                for th in ai.townhalls:
                    mfs = ai.mineral_field.closer_than(10, th)
                    if mfs:
                        ai.build(
                            UnitID.SPINECRAWLER,
                            near=th.position.towards(mfs.center, 2),
                        )
                    break
