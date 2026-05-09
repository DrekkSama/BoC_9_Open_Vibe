# Purpose: Per-unit micro for all Zerg combat unit types
# Key Decisions: CombatManeuver per unit, ARES behaviors (KeepUnitSafe, StutterUnitBack, etc.)
# Limitations: No neural parasite yet, infestor only uses AutoUseAOEAbility for fungal

from typing import Optional

import numpy as np
from cython_extensions import (
    cy_closest_to,
    cy_distance_to,
    cy_find_aoe_position,
)
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
    UseTransfuse,
)
from ares.behaviors.combat.individual.auto_use_aoe_ability import AutoUseAOEAbility

from bot.combat.formation import Formation
from bot.combat.target_scoring import assign_focus_fire

# ── Constants ────────────────────────────────────────────────────────────────
BANELING_SPLASH_RADIUS: float = 2.2
BANELING_MIN_TARGETS: int = 2
QUEEN_TRANSFUSE_ENERGY_COST: float = 50.0
ROACH_RANGE: float = 6.0


class UnitMicro:
    """Per-unit micro methods for each Zerg combat unit type."""

    def __init__(self, ai: AresBot) -> None:
        self._ai: AresBot = ai
        self._formation: Formation = Formation(ai)

    # ── Queens: Transfuse + ranged stutter-step ─────────────────────────────
    def control_queens(
        self,
        queens: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
    ) -> None:
        for queen in queens:
            maneuver = CombatManeuver()

            # Priority 1: Transfuse injured friendlies
            if queen.energy >= QUEEN_TRANSFUSE_ENERGY_COST:
                maneuver.add(UseTransfuse(unit=queen, targets=self._ai.all_own_units))

            # Priority 2: Stutter-step attack nearest enemy
            if enemies:
                target: Unit = cy_closest_to(queen.position, enemies)
                maneuver.add(
                    StutterUnitBack(
                        unit=queen, target=target, kite_via_pathing=True, grid=avoid_grid
                    )
                )

            # Priority 3: Stay safe from AOE
            maneuver.add(KeepUnitSafe(unit=queen, grid=avoid_grid))

            self._ai.register_behavior(maneuver)

    # ── Banelings: AOE detonation evaluation ────────────────────────────────
    def control_banelings(
        self,
        banelings: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
    ) -> None:
        if not banelings:
            return

        aoe_pos: Optional[np.ndarray] = cy_find_aoe_position(
            BANELING_SPLASH_RADIUS, enemies, BANELING_MIN_TARGETS, set()
        )

        for bane in banelings:
            maneuver = CombatManeuver()

            if aoe_pos is not None:
                target_point: Point2 = Point2(aoe_pos)
                # Check friendly fire: don't detonate near own zerglings
                friendly_near: bool = any(
                    cy_distance_to(u.position, target_point) < BANELING_SPLASH_RADIUS
                    for u in self._ai.all_own_units
                    if u.type_id == UnitID.ZERGLING and u.tag != bane.tag
                )
                if not friendly_near:
                    maneuver.add(
                        PathUnitToTarget(
                            unit=bane,
                            grid=avoid_grid,
                            target=target_point,
                            success_at_distance=0.0,
                        )
                    )
                else:
                    self._formation.hold_behind_lines(bane, maneuver, avoid_grid)
            else:
                self._formation.hold_behind_lines(bane, maneuver, avoid_grid)

            maneuver.add(KeepUnitSafe(unit=bane, grid=avoid_grid))
            self._ai.register_behavior(maneuver)

    # ── Ravagers: Auto bile + stutter-step ──────────────────────────────────
    def control_ravagers(
        self,
        ravagers: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
        near_enemy: dict[int, Units],
    ) -> None:
        for ravager in ravagers:
            maneuver = CombatManeuver()

            # Priority 1: Stay safe from AOE
            maneuver.add(KeepUnitSafe(unit=ravager, grid=avoid_grid))

            # Priority 2: Auto-use corrosive bile on clumped enemies
            close_enemies: Units = near_enemy.get(ravager.tag, Units([], self._ai))
            if close_enemies:
                maneuver.add(
                    AutoUseAOEAbility(
                        unit=ravager,
                        targets=list(close_enemies),
                    )
                )

            # Priority 3: Stutter-step if enemies nearby
            if close_enemies:
                target: Unit = cy_closest_to(ravager.position, close_enemies)
                maneuver.add(
                    StutterUnitBack(
                        unit=ravager, target=target, kite_via_pathing=True, grid=avoid_grid
                    )
                )
            elif enemies:
                # Move toward fight
                closest: Unit = cy_closest_to(ravager.position, enemies)
                maneuver.add(
                    PathUnitToTarget(
                        unit=ravager,
                        grid=avoid_grid,
                        target=closest.position,
                        success_at_distance=ROACH_RANGE,
                    )
                )

            self._ai.register_behavior(maneuver)

    # ── Roaches: Focus-fire + stutter-step ──────────────────────────────────
    def control_roaches(
        self,
        roaches: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
        near_enemy: dict[int, Units],
    ) -> None:
        if not roaches or not enemies:
            return

        # Overkill-aware focus-fire assignment
        assignments: dict[int, Unit] = assign_focus_fire(roaches, enemies)

        for roach in roaches:
            maneuver = CombatManeuver()

            if roach.tag in assignments:
                assigned_target: Unit = assignments[roach.tag]
                maneuver.add(
                    StutterUnitBack(
                        unit=roach,
                        target=assigned_target,
                        kite_via_pathing=True,
                        grid=avoid_grid,
                    )
                )
            else:
                # Unassigned: shoot whatever is in range, move toward fight
                close_enemies: Units = near_enemy.get(roach.tag, Units([], self._ai))
                if close_enemies:
                    maneuver.add(ShootTargetInRange(unit=roach, targets=close_enemies))
                    closest: Unit = cy_closest_to(roach.position, close_enemies)
                    maneuver.add(
                        PathUnitToTarget(
                            unit=roach,
                            grid=avoid_grid,
                            target=closest.position,
                            success_at_distance=ROACH_RANGE,
                        )
                    )
                elif enemies:
                    closest: Unit = cy_closest_to(roach.position, enemies)
                    maneuver.add(
                        PathUnitToTarget(
                            unit=roach,
                            grid=avoid_grid,
                            target=closest.position,
                            success_at_distance=ROACH_RANGE,
                        )
                    )

            maneuver.add(KeepUnitSafe(unit=roach, grid=avoid_grid))
            self._ai.register_behavior(maneuver)

    # ── Hydras: Ranged stutter-step, maintain concave ──────────────────────
    def control_hydras(
        self,
        hydras: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
        near_enemy: dict[int, Units],
    ) -> None:
        for hydra in hydras:
            maneuver = CombatManeuver()

            # Priority 1: Stay safe from AOE
            maneuver.add(KeepUnitSafe(unit=hydra, grid=avoid_grid))

            # Priority 2: Stutter-step if enemies nearby
            close_enemies: Units = near_enemy.get(hydra.tag, Units([], self._ai))
            if close_enemies:
                target: Unit = cy_closest_to(hydra.position, close_enemies)
                maneuver.add(
                    StutterUnitBack(
                        unit=hydra, target=target, kite_via_pathing=True, grid=avoid_grid
                    )
                )
            elif enemies:
                closest: Unit = cy_closest_to(hydra.position, enemies)
                maneuver.add(
                    PathUnitToTarget(
                        unit=hydra,
                        grid=avoid_grid,
                        target=closest.position,
                        success_at_distance=6.0,
                    )
                )

            self._ai.register_behavior(maneuver)

    # ── Infestors: Fungal growth on clumps, stay behind roach line ──────────
    def control_infestors(
        self,
        infestors: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
        near_enemy: dict[int, Units],
    ) -> None:
        for infestor in infestors:
            maneuver = CombatManeuver()

            # Priority 1: Stay safe (infestors are fragile)
            maneuver.add(KeepUnitSafe(unit=infestor, grid=avoid_grid))

            # Priority 2: Auto-use fungal growth on clumped enemies
            close_enemies: Units = near_enemy.get(infestor.tag, Units([], self._ai))
            if close_enemies:
                maneuver.add(
                    AutoUseAOEAbility(
                        unit=infestor,
                        targets=list(close_enemies),
                    )
                )

            # Priority 3: Stay behind the roach line
            if enemies:
                army_center: Optional[Point2] = self._formation.army_center_mass()
                if army_center is not None:
                    retreat_dir: Point2 = self._formation.retreat_direction(army_center)
                    safe_pos: Point2 = army_center + retreat_dir * 3.0
                    maneuver.add(
                        PathUnitToTarget(
                            unit=infestor,
                            grid=avoid_grid,
                            target=safe_pos,
                            success_at_distance=2.0,
                        )
                    )

            self._ai.register_behavior(maneuver)

    # ── Zerglings: Melee chase + AOE dodge ──────────────────────────────────
    def control_zerglings(
        self,
        zerglings: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
        near_enemy: dict[int, Units],
        target: Point2,
    ) -> None:
        for ling in zerglings:
            maneuver = CombatManeuver()

            # Priority 1: Dodge AOE (storms, biles, disruptors)
            maneuver.add(KeepUnitSafe(unit=ling, grid=avoid_grid))

            # Priority 2: Attack-move closest enemy or target
            if enemies:
                closest: Unit = cy_closest_to(ling.position, enemies)
                maneuver.add(AMove(unit=ling, target=closest, success_at_distance=0.0))
            else:
                maneuver.add(AMove(unit=ling, target=target))

            self._ai.register_behavior(maneuver)