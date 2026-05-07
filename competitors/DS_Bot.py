"""Zerg micro-only bot for python-sc2 + Ares framework.

Purpose: Ranged stutter-step kiting with overkill-aware focus fire, melee chase
    with AOE dodging, Baneling suicide-AOE via cy_find_aoe_position, and Queen
    Transfuse support. No macro, no production, no economy.

Key Decisions:
    - Individual CombatManeuver per unit (not group behaviors) for per-unit micro.
    - Greedy focus-fire: sort enemies by lowest HP, assign just enough ranged units
      to kill each, excess units fall through to ShootTargetInRange.
    - StutterUnitBack uses ground_grid for kiting; KeepUnitSafe with avoid_grid
      is a higher-priority layer for spell dodging.
    - Banelings call cy_find_aoe_position directly (not via UseAOEAbility since
      Banelings aren't spellcasters).

Limitations:
    - Ground units only (Zergling, Roach, Baneling, Queen).
    - Assumes pre-spawned units on micro test maps; no production/economy.
    - No advanced threat maps or role-switching.
"""

from typing import Optional

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
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
from cython_extensions import (
    cy_closest_to,
    cy_distance_to,
    cy_find_aoe_position,
)

# ── Unit type sets ─────────────────────────────────────────────────────
RANGED_TYPES: set[UnitTypeId] = {UnitTypeId.ROACH, UnitTypeId.QUEEN}
MELEE_TYPES: set[UnitTypeId] = {UnitTypeId.ZERGLING}
BANELING_TYPE: set[UnitTypeId] = {UnitTypeId.BANELING}
QUEEN_TYPE: set[UnitTypeId] = {UnitTypeId.QUEEN}
ALL_COMBAT_TYPES: set[UnitTypeId] = RANGED_TYPES | MELEE_TYPES | BANELING_TYPE

# ── Baneling constants ─────────────────────────────────────────────────
BANELING_SPLASH_RADIUS: float = 2.2
BANELING_MIN_TARGETS: int = 2


class DSBot(AresBot):
    """Zerg micro-only bot.

    Controls: Zergling (melee chase), Roach (stutter-step + focus-fire),
    Baneling (suicide AOE), Queen (Transfuse + ranged attack).
    """

    def __init__(self, game_step_override: Optional[int] = None):
        super().__init__(game_step_override)

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        # Collect friendly combat units (exclude structures, workers, eggs)
        my_units: Units = self.units.filter(
            lambda u: u.type_id in ALL_COMBAT_TYPES and not u.is_structure
        )
        if not my_units:
            return

        # Collect visible enemy units (exclude structures)
        enemy_units: Units = self.enemy_units.filter(
            lambda u: not u.is_structure
        )
        if not enemy_units:
            return

        # Separate by type
        queens: Units = my_units.filter(lambda u: u.type_id in QUEEN_TYPE)
        banelings: Units = my_units.filter(lambda u: u.type_id in BANELING_TYPE)
        roaches: Units = my_units.filter(lambda u: u.type_id == UnitTypeId.ROACH)
        zerglings: Units = my_units.filter(lambda u: u.type_id in MELEE_TYPES)

        # Grids
        ground_grid: np.ndarray = self.mediator.get_ground_grid
        avoid_grid: np.ndarray = self.mediator.get_ground_avoidance_grid

        # Frame loop order (per plan):
        # 1. Queens first — Transfuse then combat
        self._control_queens(queens, enemy_units, my_units, ground_grid, avoid_grid)

        # 2. Banelings — evaluate detonation targets
        self._control_banelings(banelings, enemy_units, avoid_grid)

        # 3. Roaches — focus-fire + stutter-step
        self._control_roaches(roaches, enemy_units, ground_grid, avoid_grid)

        # 4. Zerglings — AMove closest enemy, dodge AOE
        self._control_zerglings(zerglings, enemy_units, avoid_grid)

    # ── Focus-fire assignment ──────────────────────────────────────────

    @staticmethod
    def _assign_focus_fire(
        ranged_units: list[Unit], enemies: list[Unit]
    ) -> dict[int, int]:
        """Greedy overkill-aware focus-fire assignment.

        Sorts enemies by lowest HP first, then assigns just enough ranged
        units to kill each enemy. Excess units get no assignment and fall
        through to ShootTargetInRange.

        Returns:
            Dict mapping unit.tag -> enemy.tag for assigned units.
        """
        assignments: dict[int, int] = {}
        assigned_tags: set[int] = set()

        # Sort enemies by lowest HP first to secure kills
        sorted_enemies: list[Unit] = sorted(
            [e for e in enemies if e.health + e.shield > 0],
            key=lambda e: e.health + e.shield,
        )

        for enemy in sorted_enemies:
            hp_remaining: float = enemy.health + enemy.shield

            for unit in ranged_units:
                if unit.tag in assigned_tags:
                    continue

                dmg: float = unit.calculate_damage_vs_target(enemy)[0]
                if dmg <= 0:
                    continue

                assignments[unit.tag] = enemy.tag
                assigned_tags.add(unit.tag)
                hp_remaining -= dmg

                if hp_remaining <= 0:
                    break  # Enough units assigned to kill this enemy

        return assignments

    # ── Queen control ──────────────────────────────────────────────────

    def _control_queens(
        self,
        queens: Units,
        enemies: Units,
        all_friendlies: Units,
        ground_grid: np.ndarray,
        avoid_grid: np.ndarray,
    ) -> None:
        """Queens: Transfuse low-HP friendlies first, then stutter-step attack."""
        enemy_list: list[Unit] = list(enemies)
        queen_list: list[Unit] = list(queens)

        assignments: dict[int, int] = self._assign_focus_fire(queen_list, enemy_list)

        for queen in queens:
            maneuver: CombatManeuver = CombatManeuver()

            # Priority 1: Transfuse friendly units below 40% HP
            maneuver.add(UseTransfuse(unit=queen, targets=all_friendlies))

            # Priority 2: Dodge AOE effects (storms, biles, disruptors)
            maneuver.add(KeepUnitSafe(unit=queen, grid=avoid_grid))

            # Priority 3: Stutter-step with assigned focus-fire target
            assigned_enemy_tag: int | None = assignments.get(queen.tag)
            if assigned_enemy_tag is not None:
                target: Unit | None = enemies.find_by_tag(assigned_enemy_tag)
                if target is not None:
                    maneuver.add(
                        StutterUnitBack(
                            unit=queen,
                            target=target,
                            kite_via_pathing=True,
                            grid=ground_grid,
                        )
                    )

            # Priority 4: Shoot any target in range (fallback for excess units)
            maneuver.add(ShootTargetInRange(unit=queen, targets=enemy_list))

            # Priority 5: AMove toward closest enemy (last resort)
            closest: Unit | None = cy_closest_to(queen.position, enemy_list)
            if closest is not None:
                maneuver.add(AMove(unit=queen, target=closest))

            self.register_behavior(maneuver)

    # ── Roach control ──────────────────────────────────────────────────

    def _control_roaches(
        self,
        roaches: Units,
        enemies: Units,
        ground_grid: np.ndarray,
        avoid_grid: np.ndarray,
    ) -> None:
        """Roaches: overkill-aware focus-fire with stutter-step kiting."""
        enemy_list: list[Unit] = list(enemies)
        roach_list: list[Unit] = list(roaches)

        assignments: dict[int, int] = self._assign_focus_fire(roach_list, enemy_list)

        for roach in roaches:
            maneuver: CombatManeuver = CombatManeuver()

            # Priority 1: Dodge AOE effects
            maneuver.add(KeepUnitSafe(unit=roach, grid=avoid_grid))

            # Priority 2: Stutter-step with assigned focus-fire target
            assigned_enemy_tag: int | None = assignments.get(roach.tag)
            if assigned_enemy_tag is not None:
                target: Unit | None = enemies.find_by_tag(assigned_enemy_tag)
                if target is not None:
                    maneuver.add(
                        StutterUnitBack(
                            unit=roach,
                            target=target,
                            kite_via_pathing=True,
                            grid=ground_grid,
                        )
                    )

            # Priority 3: Shoot any target in range (fallback for excess units)
            maneuver.add(ShootTargetInRange(unit=roach, targets=enemy_list))

            # Priority 4: AMove toward closest enemy (last resort)
            closest: Unit | None = cy_closest_to(roach.position, enemy_list)
            if closest is not None:
                maneuver.add(AMove(unit=roach, target=closest))

            self.register_behavior(maneuver)

    # ── Zergling control ───────────────────────────────────────────────

    def _control_zerglings(
        self, zerglings: Units, enemies: Units, avoid_grid: np.ndarray
    ) -> None:
        """Zerglings: AMove closest enemy, dodge AOE, prefer low-HP targets."""
        enemy_list: list[Unit] = list(enemies)

        for ling in zerglings:
            maneuver: CombatManeuver = CombatManeuver()

            # Priority 1: Dodge AOE effects (banelings, tanks, storms, etc.)
            maneuver.add(KeepUnitSafe(unit=ling, grid=avoid_grid))

            # Priority 2: Attack-move toward best target
            target: Unit | None = self._pick_ling_target(ling, enemy_list)
            if target is not None:
                maneuver.add(AMove(unit=ling, target=target))

            self.register_behavior(maneuver)

    @staticmethod
    def _pick_ling_target(ling: Unit, enemies: list[Unit]) -> Unit | None:
        """Pick best target for a zergling: closest, with low-HP preference.

        Uses a composite score: distance + (health * 0.001) so that among
        equally-distant targets, the lower-HP one is preferred.
        """
        if not enemies:
            return None

        scored: list[tuple[float, Unit]] = [
            (
                cy_distance_to(ling.position, e.position)
                + (e.health + e.shield) * 0.001,
                e,
            )
            for e in enemies
        ]
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    # ── Baneling control ───────────────────────────────────────────────

    def _control_banelings(
        self, banelings: Units, enemies: Units, avoid_grid: np.ndarray
    ) -> None:
        """Banelings: find optimal AOE detonation point, move there.

        Uses cy_find_aoe_position to find the point that hits the most
        enemies within splash radius. If no good target exists, retreat
        to a safe rally point instead of wasting the Baneling.
        """
        enemy_list: list[Unit] = list(enemies)

        for baneling in banelings:
            maneuver: CombatManeuver = CombatManeuver()

            # Priority 1: Dodge AOE effects (banelings are high-value)
            maneuver.add(KeepUnitSafe(unit=baneling, grid=avoid_grid))

            # Priority 2: Find optimal detonation point
            aoe_pos: np.ndarray | None = cy_find_aoe_position(
                BANELING_SPLASH_RADIUS,
                enemy_list,
                BANELING_MIN_TARGETS,
            )

            if aoe_pos is not None:
                target_point: Point2 = Point2(aoe_pos)
                # Path toward detonation point (auto-detonates on contact)
                maneuver.add(
                    PathUnitToTarget(
                        unit=baneling,
                        grid=avoid_grid,
                        target=target_point,
                        success_at_distance=0.0,
                    )
                )
                # AMove as fallback to trigger detonation on contact
                maneuver.add(AMove(unit=baneling, target=target_point))
            else:
                # No good target — retreat to safe rally point
                safe_pos: Point2 = self._get_safe_rally_point(baneling)
                maneuver.add(
                    PathUnitToTarget(
                        unit=baneling,
                        grid=avoid_grid,
                        target=safe_pos,
                        success_at_distance=3.0,
                    )
                )

            self.register_behavior(maneuver)

    def _get_safe_rally_point(self, unit: Unit) -> Point2:
        """Get a safe position behind friendly lines for idle banelings."""
        if self.start_location:
            return self.start_location
        return self.game_info.map_center
