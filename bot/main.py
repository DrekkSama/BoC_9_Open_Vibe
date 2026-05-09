# Purpose: Main AresBot subclass wiring all modules together
# Key Decisions: Use ARES role system, MacroPlan, CombatManager for micro dispatch
# Limitations: No ML-based engagement decisions, no neural parasite yet

from typing import Optional

from cython_extensions import cy_closest_to, cy_distance_to
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import AresBot
from ares.behaviors.combat.individual import QueenSpreadCreep, TumorSpreadCreep
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    GasBuildingController,
    MacroPlan,
    Mining,
    SpawnController,
)
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole

from bot.combat import CombatManager
from bot.compositions import get_army_comp
from bot.managers.research_manager import research_upgrades
from bot.managers.response_manager import assess_threats, respond_to_threats

# ── Constants ────────────────────────────────────────────────────────────────
QUEEN_INJECT_ENERGY_COST: float = 25.0
CREEP_TUMOR_ENERGY: float = 25.0
BEGIN_ATTACK_SUPPLY: float = 6.0

# Unit types that shouldn't be assigned combat roles
IGNORE_ROLE_TYPES: set[UnitID] = {
    UnitID.EGG,
    UnitID.LARVA,
    UnitID.CREEPTUMORBURROWED,
    UnitID.CREEPTUMORQUEEN,
    UnitID.CREEPTUMOR,
    UnitID.MULE,
    UnitID.OVERLORD,
    UnitID.OVERSEER,
    UnitID.DRONE,
}


class GLM_Bot(AresBot):
    """Zerg B2GM Roach Ravager bot using ARES framework."""

    def __init__(self, game_step_override: Optional[int] = None):
        super().__init__(game_step_override)
        self._commenced_attack: bool = False
        self._combat_mgr: Optional[CombatManager] = None

    @property
    def attack_target(self) -> Point2:
        """Determine where the army should attack."""
        if self.enemy_structures:
            return cy_closest_to(self.start_location, self.enemy_structures).position
        if self.time < 240.0:
            return self.enemy_start_locations[0]
        # Late game: search expansions
        for expand_pos in self.expansion_locations_list:
            if not self.is_visible(expand_pos):
                return expand_pos
        return self.enemy_start_locations[0]

    async def on_start(self) -> None:
        """Called once at the start of the game."""
        await super().on_start()
        self._combat_mgr = CombatManager(self)

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)
        if not self.all_own_units:
            return

        # ── Macro ───────────────────────────────────────────────────────────
        self._macro()

        # ── Combat ──────────────────────────────────────────────────────────
        forces: Units = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)

        if not self._commenced_attack:
            if self.get_total_supply(forces) >= BEGIN_ATTACK_SUPPLY:
                self._commenced_attack = True

        if self._commenced_attack and forces and self._combat_mgr is not None:
            self._combat_mgr.step(forces)

        # ── Queen inject + creep spread (always run) ────────────────────────
        self._queen_inject()
        self._creep_tumor_spread()

        # ── Research upgrades (after build completes) ───────────────────────
        if self.build_order_runner.build_completed:
            research_upgrades(self)

        # ── Response book (after build completes) ──────────────────────────
        if self.build_order_runner.build_completed:
            self._threats: dict[str, bool] = assess_threats(self)
            respond_to_threats(self, self._threats)

    async def on_unit_created(self, unit: Unit) -> None:
        """Assign combat units to ATTACKING role on creation."""
        await super().on_unit_created(unit)

        if unit.type_id in IGNORE_ROLE_TYPES or unit.type_id in ALL_STRUCTURES:
            return
        if unit.type_id in WORKER_TYPES:
            return

        self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    # ── Macro ───────────────────────────────────────────────────────────────
    def _macro(self) -> None:
        """Run macro behaviors: mining, supply, workers, gas, spawning."""
        self.register_behavior(Mining())

        army_comp: dict[UnitID, dict] = get_army_comp(
            self.time,
            air_threat=getattr(self, "_threats", {}).get("air_signs", False),
        )

        macro_plan: MacroPlan = MacroPlan()

        if self.build_order_runner.build_completed:
            # Dynamic macro after opening completes
            macro_plan.add(AutoSupply(base_location=self.start_location))
            macro_plan.add(BuildWorkers(to_count=54))
            macro_plan.add(GasBuildingController(to_count=6))
            macro_plan.add(SpawnController(army_comp))

        self.register_behavior(macro_plan)

    # ── Queen Inject (prioritize inject over creep tumor) ────────────────────
    def _queen_inject(self) -> None:
        """Inject larva on townhalls with idle queens. Prioritize inject."""
        all_queens: list[Unit] = list(self.units(UnitID.QUEEN))

        for queen in all_queens:
            if queen.energy < QUEEN_INJECT_ENERGY_COST:
                continue

            # Priority 1: Inject closest uninjected townhall
            for th in self.townhalls:
                if cy_distance_to(queen.position, th.position) < 10.0:
                    if not th.has_buff(BuffId.QUEENSPAWNLARVATIMER):
                        queen(AbilityId.EFFECT_INJECTLARVA, th)
                        break

    # ── Creep Tumor Spread ──────────────────────────────────────────────────
    def _creep_tumor_spread(self) -> None:
        """Spread creep tumors and queens with excess energy."""
        target: Point2 = self.attack_target

        # Tumors: spread toward enemy
        tumors: Units = self.structures({UnitID.CREEPTUMORBURROWED, UnitID.CREEPTUMORQUEEN})
        for tumor in tumors:
            self.register_behavior(TumorSpreadCreep(unit=tumor, target=target))

        # Queens with excess energy (after inject) spread creep
        for queen in self.units(UnitID.QUEEN):
            if queen.energy >= CREEP_TUMOR_ENERGY + QUEEN_INJECT_ENERGY_COST:
                # Only spread creep if all nearby townhalls are injected
                nearby_ths = [
                    th for th in self.townhalls
                    if cy_distance_to(queen.position, th.position) < 10.0
                ]
                all_injected: bool = all(
                    th.has_buff(BuffId.QUEENSPAWNLARVATIMER)
                    for th in nearby_ths
                ) if nearby_ths else True
                if all_injected:
                    self.register_behavior(
                        QueenSpreadCreep(unit=queen, cancel_if_close_enemy=True)
                    )

    # ── Micro (delegated to CombatManager) ────────────────────────────────────
    # All micro logic lives in bot/combat/ — CombatManager.step() dispatches
    # per-unit micro via UnitMicro, formation helpers via Formation,
    # focus-fire via target_scoring.assign_focus_fire


# Alias for run.py compatibility
MicroBot = GLM_Bot
