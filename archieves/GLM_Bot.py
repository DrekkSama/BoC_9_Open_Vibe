"""Zerg Tier-1 Micro Bot | Ranged stutter-step, melee chase, baneling AOE, queen heal
Key Decisions: Focus-fire overkill-aware assignment for ranged; cy_find_aoe_position for banelings;
               Ares CombatManeuver per-unit for consistent behavior chains.
Limitations: No macro/production/economy. Ground units only (Zergling, Roach, Baneling, Queen).
"""
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

# ── Constants ────────────────────────────────────────────────────────────────
BANELING_SPLASH_RADIUS: float = 2.2
BANELING_MIN_TARGETS: int = 2
QUEEN_TRANSFUSE_HP_THRESHOLD: float = 0.4
QUEEN_TRANSFUSE_ENERGY_COST: float = 50.0
QUEEN_RANGE: float = 7.0
ROACH_RANGE: float = 6.0


class MicroBot(AresBot):
    """Zerg micro-only bot: stutter-step, focus-fire, melee chase, baneling AOE, queen heal."""

    def __init__(self, game_step_override: Optional[int] = None):
        super().__init__(game_step_override)

    async def on_step(self, iteration: int) -> None:
        await super(MicroBot, self).on_step(iteration)
        if not self.all_own_units:
            return

        # ── Collect units by type ───────────────────────────────────────────
        zerglings: list[Unit] = [
            u for u in self.all_own_units if u.type_id == UnitID.ZERGLING
        ]
        roaches: list[Unit] = [
            u for u in self.all_own_units if u.type_id == UnitID.ROACH
        ]
        banelings: list[Unit] = [
            u for u in self.all_own_units if u.type_id == UnitID.BANELING
        ]
        queens: list[Unit] = [
            u for u in self.all_own_units if u.type_id == UnitID.QUEEN
        ]

        enemies: list[Unit] = [
            u for u in self.enemy_units
            if not u.is_memory and (not u.is_cloaked or u.is_revealed)
        ]

        if not enemies:
            # No enemies visible: regroup toward map center
            center: Point2 = self.game_info.map_center
            for unit_list in (zerglings, roaches, banelings, queens):
                for u in unit_list:
                    self.register_behavior(AMove(u, center))
            return

        avoid_grid: np.ndarray = self.mediator.get_ground_avoidance_grid

        # ── Execution order: Queens → Banelings → Roaches → Zerglings ───────
        self._control_queens(queens, enemies, avoid_grid)
        self._control_banelings(banelings, enemies, avoid_grid)
        self._control_roaches(roaches, enemies, avoid_grid)
        self._control_zerglings(zerglings, enemies, avoid_grid)

    # ── Queens: Transfuse + ranged stutter-step ──────────────────────────────
    def _control_queens(
        self,
        queens: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
    ) -> None:
        for queen in queens:
            maneuver = CombatManeuver()

            # Priority 1: Transfuse injured friendlies
            if queen.energy >= QUEEN_TRANSFUSE_ENERGY_COST:
                maneuver.add(
                    UseTransfuse(unit=queen, targets=self.all_own_units)
                )

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

            self.register_behavior(maneuver)

    # ── Banelings: AOE detonation evaluation ─────────────────────────────────
    def _control_banelings(
        self,
        banelings: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
    ) -> None:
        if not banelings:
            return

        # Find the best AOE position once for all banelings (shared target pool)
        aoe_pos: Optional[np.ndarray] = cy_find_aoe_position(
            BANELING_SPLASH_RADIUS, enemies, BANELING_MIN_TARGETS, set()
        )

        for bane in banelings:
            maneuver = CombatManeuver()

            if aoe_pos is not None:
                target_point: Point2 = Point2(aoe_pos)
                # Check friendly fire: don't detonate near own zerglings if avoidable
                friendly_near: bool = any(
                    cy_distance_to(u.position, target_point) < BANELING_SPLASH_RADIUS
                    for u in self.all_own_units
                    if u.type_id == UnitID.ZERGLING and u.tag != bane.tag
                )
                if not friendly_near:
                    # Move to detonation point; baneling auto-detonates on contact
                    maneuver.add(
                        PathUnitToTarget(
                            unit=bane,
                            grid=avoid_grid,
                            target=target_point,
                            success_at_distance=0.0,
                        )
                    )
                else:
                    # Friendly fire risk: hold position behind army
                    self._hold_behind_lines(bane, maneuver, avoid_grid)
            else:
                # Not enough targets: hold behind friendly lines
                self._hold_behind_lines(bane, maneuver, avoid_grid)

            # Always keep safe from AOE
            maneuver.add(KeepUnitSafe(unit=bane, grid=avoid_grid))
            self.register_behavior(maneuver)

    def _hold_behind_lines(
        self,
        bane: Unit,
        maneuver: CombatManeuver,
        avoid_grid: np.ndarray,
    ) -> None:
        """Move baneling to a rally point behind the main army."""
        army_center: Optional[Point2] = self._army_center_mass()
        if army_center is not None:
            # Move slightly behind army center (away from enemy start)
            retreat_dir: Point2 = self._retreat_direction(army_center)
            rally: Point2 = army_center + retreat_dir * 3.0
            maneuver.add(
                PathUnitToTarget(
                    unit=bane, grid=avoid_grid, target=rally, success_at_distance=2.0
                )
            )

    # ── Roaches: Focus-fire + stutter-step ───────────────────────────────────
    def _control_roaches(
        self,
        roaches: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
    ) -> None:
        if not roaches or not enemies:
            return

        # Overkill-aware focus-fire assignment
        assignments: dict[int, Unit] = self._assign_focus_fire(roaches, enemies)

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
                maneuver.add(ShootTargetInRange(unit=roach, targets=enemies))
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
            self.register_behavior(maneuver)

    def _assign_focus_fire(
        self,
        roaches: list[Unit],
        enemies: list[Unit],
    ) -> dict[int, Unit]:
        """Assign just enough roaches per enemy to kill it; excess retarget to next enemy.

        Returns a mapping of roach tag → assigned enemy target.
        """
        assignments: dict[int, Unit] = {}
        assigned_roaches: set[int] = set()

        # Sort enemies by distance to roach center mass (closest first)
        roach_center: Point2 = self._unit_list_center(roaches)
        sorted_enemies: list[Unit] = sorted(
            enemies, key=lambda e: cy_distance_to(roach_center, e.position)
        )

        for enemy in sorted_enemies:
            if len(assigned_roaches) >= len(roaches):
                break

            enemy_hp: float = enemy.health + enemy.shield
            if enemy_hp <= 0:
                continue

            # Estimate damage per roach shot
            # Use first unassigned roach as representative for damage calc
            sample_roach: Optional[Unit] = None
            for r in roaches:
                if r.tag not in assigned_roaches:
                    sample_roach = r
                    break

            if sample_roach is None:
                break

            damage_per_shot: float = max(
                sample_roach.calculate_damage_vs_target(enemy)[0], 1.0
            )
            shots_needed: int = max(1, int(enemy_hp / damage_per_shot) + 1)

            # Assign that many roaches (or as many as available)
            assigned_count: int = 0
            for r in roaches:
                if r.tag in assigned_roaches:
                    continue
                if assigned_count >= shots_needed:
                    break
                assignments[r.tag] = enemy
                assigned_roaches.add(r.tag)
                assigned_count += 1

        return assignments

    # ── Zerglings: Melee chase + AOE dodge ───────────────────────────────────
    def _control_zerglings(
        self,
        zerglings: list[Unit],
        enemies: list[Unit],
        avoid_grid: np.ndarray,
    ) -> None:
        for ling in zerglings:
            maneuver = CombatManeuver()

            # Priority 1: Dodge AOE (storms, biles, disruptors)
            maneuver.add(KeepUnitSafe(unit=ling, grid=avoid_grid))

            # Priority 2: Attack-move closest enemy
            if enemies:
                target: Unit = cy_closest_to(ling.position, enemies)
                maneuver.add(AMove(unit=ling, target=target, success_at_distance=0.0))

            self.register_behavior(maneuver)

    # ── Utility helpers ──────────────────────────────────────────────────────
    def _army_center_mass(self) -> Optional[Point2]:
        """Center-of-mass of all own combat units."""
        combat_units: list[Unit] = [
            u for u in self.all_own_units
            if u.type_id in {UnitID.ZERGLING, UnitID.ROACH, UnitID.BANELING, UnitID.QUEEN}
        ]
        if not combat_units:
            return None
        return self._unit_list_center(combat_units)

    @staticmethod
    def _unit_list_center(units: list[Unit]) -> Point2:
        """Simple center-of-mass for a list of units."""
        if not units:
            return Point2((0.0, 0.0))
        x: float = sum(u.position.x for u in units) / len(units)
        y: float = sum(u.position.y for u in units) / len(units)
        return Point2((x, y))

    def _retreat_direction(self, from_pos: Point2) -> Point2:
        """Unit vector pointing away from enemy start location."""
        enemy_start: Point2 = self.enemy_start_locations[0]
        dx: float = from_pos.x - enemy_start.x
        dy: float = from_pos.y - enemy_start.y
        dist: float = (dx * dx + dy * dy) ** 0.5
        if dist < 0.1:
            return Point2((1.0, 0.0))
        return Point2((dx / dist, dy / dist))
