# Purpose: Per-unit micro for Zerg combat unit types (excludes Queens)
# Key Decisions: CombatManeuver per unit, Clicadinha-style priority chain,
#   manual cooldown tracking for ravager bile and infestor fungal
# Limitations: No neural parasite, no burrow micro yet; Queens managed separately

from typing import Optional

import numpy as np
from cython_extensions import (
    cy_closest_to,
    cy_distance_to,
    cy_find_aoe_position,
    cy_in_attack_range,
    cy_pick_enemy_target,
)
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    KeepUnitSafe,
    PathUnitToTarget,
    ShootTargetInRange,
    StutterUnitBack,
    StutterUnitForward,
)
from ares.consts import ALL_STRUCTURES, WORKER_TYPES

# ── Constants ────────────────────────────────────────────────────────────────
BANELING_SPLASH_RADIUS: float = 2.2
BANELING_MIN_TARGETS: int = 2
ROACH_RANGE: float = 6.0
BILE_RANGE: int = 9
BILE_COOLDOWN_FRAMES: int = int(22.4 * 7.0) + 6  # ~7s
FUNGAL_COOLDOWN_FRAMES: int = int(22.4 * 3.0) + 4  # ~3s
FUNGAL_MIN_TARGETS: int = 3
FUNGAL_RADIUS: float = 2.0
FUNGAL_ENERGY_COST: float = 75.0

# Bile priority targets (lower = higher priority)
BILE_AIR_TYPES: set[UnitID] = {UnitID.LIBERATORAG, UnitID.MEDIVAC}


class UnitMicro:
    """Per-unit micro methods for each Zerg combat unit type."""

    def __init__(self, ai: AresBot) -> None:
        self._ai: AresBot = ai
        # Manual cooldown trackers
        self._bile_cd: dict[int, int] = {}
        self._fungal_global_cd: int = 0

    def execute(
        self,
        squad_units: list[Unit],
        all_close_enemy: Units,
        near_enemy: dict[int, Units],
        can_engage: bool,
        aggressive: bool,
        target: Point2,
        attack_target: Point2,
    ) -> None:
        """Dispatch each unit to its type-specific micro."""
        ai = self._ai
        grid: np.ndarray = ai.mediator.get_ground_grid
        avoid_grid: np.ndarray = ai.mediator.get_ground_avoidance_grid

        for unit in squad_units:
            # Get close enemies for this specific unit
            all_close: Units = near_enemy.get(unit.tag, Units([], ai)).filter(
                lambda u: not u.is_memory and u.type_id not in {
                    UnitID.EGG, UnitID.LARVA,
                    UnitID.CREEPTUMOR, UnitID.CREEPTUMORQUEEN, UnitID.CREEPTUMORBURROWED,
                }
            )
            only_enemy_units: Units = all_close.filter(
                lambda u: u.type_id not in ALL_STRUCTURES
            )

            maneuver: CombatManeuver = CombatManeuver()

            # ── Dispatch by unit type ─────────────────────────────────────
            tid = unit.type_id

            if tid == UnitID.RAVAGER:
                self._control_ravager(unit, maneuver, all_close, only_enemy_units, grid, target)
            elif tid == UnitID.ROACH:
                self._control_roach(unit, maneuver, all_close, only_enemy_units, grid, target, can_engage)
            elif tid == UnitID.ZERGLING:
                self._control_zergling(unit, maneuver, all_close, only_enemy_units, grid, avoid_grid, target, can_engage, aggressive)
            elif tid == UnitID.HYDRALISK:
                self._control_hydra(unit, maneuver, all_close, only_enemy_units, grid, target, can_engage)
            elif tid == UnitID.INFESTOR:
                self._control_infestor(unit, maneuver, all_close, only_enemy_units, grid, target)
            elif tid == UnitID.BANELING:
                self._control_baneling(unit, maneuver, all_close, only_enemy_units, avoid_grid, target)
            else:
                # Fallback: generic attack-move
                if all_close:
                    maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
                    maneuver.add(AMove(unit=unit, target=target))
                else:
                    maneuver.add(PathUnitToTarget(unit=unit, grid=grid, target=target))
                    maneuver.add(AMove(unit=unit, target=target))

            ai.register_behavior(maneuver)

    # ── Roach: Shoot → Stutter back ──────────────────────────────────────────
    def _control_roach(
        self,
        unit: Unit,
        maneuver: CombatManeuver,
        all_close: Units,
        only_enemy_units: Units,
        grid: np.ndarray,
        target: Point2,
        can_engage: bool,
    ) -> None:
        if all_close:
            # Shoot enemy units in range first
            if in_range := cy_in_attack_range(unit, only_enemy_units):
                maneuver.add(ShootTargetInRange(unit=unit, targets=in_range))
            # Then structures
            elif in_range := cy_in_attack_range(unit, all_close):
                maneuver.add(ShootTargetInRange(unit=unit, targets=in_range))

            if can_engage:
                enemy_target: Unit = cy_pick_enemy_target(all_close)
                maneuver.add(StutterUnitBack(unit=unit, target=enemy_target, grid=grid))
            else:
                maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                maneuver.add(PathUnitToTarget(unit=unit, grid=grid, target=target, success_at_distance=12.0))
        else:
            maneuver.add(PathUnitToTarget(unit=unit, grid=grid, target=target))
            maneuver.add(AMove(unit=unit, target=target))

    # ── Ravager: Manual bile → Shoot → Stutter ──────────────────────────────
    def _control_ravager(
        self,
        unit: Unit,
        maneuver: CombatManeuver,
        all_close: Units,
        only_enemy_units: Units,
        grid: np.ndarray,
        target: Point2,
    ) -> None:
        ai = self._ai

        # Manual bile with cooldown tracking
        if self._bile_cd.get(unit.tag, 0) <= ai.state.game_loop:
            air_bile: list[Unit] = [
                u for u in ai.enemy_units
                if u.type_id in BILE_AIR_TYPES
                and not u.is_memory
                and cy_distance_to(unit.position, u.position) <= BILE_RANGE
            ]
            ground_bile: list[Unit] = [
                u for u in all_close
                if u.type_id != UnitID.BANSHEE
            ]

            def _bile_tier(u: Unit) -> int:
                t = u.type_id
                if t == UnitID.SIEGETANKSIEGED: return 0
                if t == UnitID.LIBERATORAG:     return 1
                if t == UnitID.MEDIVAC:         return 2
                if t in WORKER_TYPES:           return 4
                if t in ALL_STRUCTURES:         return 5
                return 3

            all_bile_candidates: list[Unit] = air_bile + ground_bile
            if all_bile_candidates:
                best_bile = min(
                    all_bile_candidates,
                    key=lambda u: (_bile_tier(u), cy_distance_to(unit.position, u.position)),
                )
                unit(AbilityId.EFFECT_CORROSIVEBILE, best_bile.position)
                self._bile_cd[unit.tag] = ai.state.game_loop + BILE_COOLDOWN_FRAMES
                # Bile fired: don't register other behaviors this frame
                return

        # Bile on cooldown: normal attack behavior
        if all_close:
            if in_range := cy_in_attack_range(unit, only_enemy_units):
                maneuver.add(ShootTargetInRange(unit=unit, targets=in_range))
            elif in_range := cy_in_attack_range(unit, all_close):
                maneuver.add(ShootTargetInRange(unit=unit, targets=in_range))

            enemy_target: Unit = cy_pick_enemy_target(all_close)
            maneuver.add(StutterUnitBack(unit=unit, target=enemy_target, grid=grid))
        else:
            maneuver.add(PathUnitToTarget(unit=unit, grid=grid, target=target))
            maneuver.add(AMove(unit=unit, target=target))

    # ── Zergling: AMove or stay safe ─────────────────────────────────────────
    def _control_zergling(
        self,
        unit: Unit,
        maneuver: CombatManeuver,
        all_close: Units,
        only_enemy_units: Units,
        grid: np.ndarray,
        avoid_grid: np.ndarray,
        target: Point2,
        can_engage: bool,
        aggressive: bool,
    ) -> None:
        ai = self._ai

        if all_close:
            maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
            if can_engage:
                # If roaches exist and are attacking, lings can a-move
                if ai.units(UnitID.ROACH).amount > 0:
                    maneuver.add(AMove(unit=unit, target=target))
                else:
                    closest: Unit = cy_closest_to(unit.position, all_close)
                    maneuver.add(AMove(unit=unit, target=closest.position))
            else:
                maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
        else:
            maneuver.add(PathUnitToTarget(unit=unit, grid=grid, target=target))
            maneuver.add(AMove(unit=unit, target=target))

    # ── Hydra: Shoot → Stutter ───────────────────────────────────────────────
    def _control_hydra(
        self,
        unit: Unit,
        maneuver: CombatManeuver,
        all_close: Units,
        only_enemy_units: Units,
        grid: np.ndarray,
        target: Point2,
        can_engage: bool,
    ) -> None:
        avoid_grid: np.ndarray = self._ai.mediator.get_ground_avoidance_grid
        maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))

        if all_close:
            # Shoot enemy units in range
            if in_range := cy_in_attack_range(unit, only_enemy_units):
                maneuver.add(ShootTargetInRange(unit=unit, targets=in_range))
            elif in_range := cy_in_attack_range(unit, all_close):
                maneuver.add(ShootTargetInRange(unit=unit, targets=in_range))

            if can_engage:
                enemy_target: Unit = cy_pick_enemy_target(all_close)
                maneuver.add(StutterUnitBack(unit=unit, target=enemy_target, grid=grid))
            else:
                maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                maneuver.add(PathUnitToTarget(unit=unit, grid=grid, target=target, success_at_distance=12.0))
        else:
            maneuver.add(PathUnitToTarget(unit=unit, grid=grid, target=target))
            maneuver.add(AMove(unit=unit, target=target))

    # ── Infestor: Fungal on clumps → Stay safe ──────────────────────────────
    def _control_infestor(
        self,
        unit: Unit,
        maneuver: CombatManeuver,
        all_close: Units,
        only_enemy_units: Units,
        grid: np.ndarray,
        target: Point2,
    ) -> None:
        ai = self._ai
        avoid_grid: np.ndarray = ai.mediator.get_ground_avoidance_grid

        # Global fungal cooldown: no infestor can cast while active
        fungal_global_ready: bool = self._fungal_global_cd <= ai.state.game_loop
        if not fungal_global_ready:
            maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
            return

        # Find best fungal target
        fungal_targets: list[Unit] = [
            u for u in only_enemy_units
            if u.type_id != UnitID.RAVEN
        ]
        if fungal_targets and unit.energy >= FUNGAL_ENERGY_COST:
            best_pos: Optional[Point2] = None
            best_count: int = 0
            for candidate in fungal_targets:
                count = sum(
                    1 for u in fungal_targets
                    if cy_distance_to(candidate.position, u.position) <= FUNGAL_RADIUS
                )
                if count > best_count:
                    best_count = count
                    best_pos = candidate.position
            if best_pos and best_count >= FUNGAL_MIN_TARGETS:
                unit(AbilityId.FUNGALGROWTH_FUNGALGROWTH, best_pos)
                self._fungal_global_cd = ai.state.game_loop + FUNGAL_COOLDOWN_FRAMES
                maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
                return

        # No fungal: stay safe, path behind army
        maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
        if all_close:
            maneuver.add(PathUnitToTarget(unit=unit, grid=grid, target=target, success_at_distance=8.0))

    # ── Baneling: AOE detonation evaluation ──────────────────────────────────
    def _control_baneling(
        self,
        unit: Unit,
        maneuver: CombatManeuver,
        all_close: Units,
        only_enemy_units: Units,
        avoid_grid: np.ndarray,
        target: Point2,
    ) -> None:
        enemies: list[Unit] = list(only_enemy_units)
        aoe_pos: Optional[np.ndarray] = cy_find_aoe_position(
            BANELING_SPLASH_RADIUS, enemies, BANELING_MIN_TARGETS, set()
        )

        if aoe_pos is not None:
            target_point: Point2 = Point2(aoe_pos)
            # Check friendly fire: don't detonate near own zerglings
            friendly_near: bool = any(
                cy_distance_to(u.position, target_point) < BANELING_SPLASH_RADIUS
                for u in self._ai.all_own_units
                if u.type_id == UnitID.ZERGLING and u.tag != unit.tag
            )
            if not friendly_near:
                maneuver.add(PathUnitToTarget(
                    unit=unit, grid=avoid_grid, target=target_point,
                    success_at_distance=0.0,
                ))
            else:
                # Hold behind lines
                maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
        else:
            maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))

        maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))