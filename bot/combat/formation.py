# Purpose: Formation geometry helpers — center of mass, retreat direction, hold-behind-lines
# Key Decisions: Simple vector math, no pathing grid dependency for direction calc
# Limitations: Retreat direction uses enemy start location as heuristic

from typing import Optional

from cython_extensions import cy_distance_to
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit

from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import PathUnitToTarget

# ── Combat unit types for center-of-mass calculation ─────────────────────────
COMBAT_UNIT_TYPES: set[UnitID] = {
    UnitID.ZERGLING, UnitID.ROACH, UnitID.RAVAGER,
    UnitID.BANELING, UnitID.QUEEN, UnitID.HYDRALISK,
}


class Formation:
    """Formation geometry helpers for army positioning."""

    def __init__(self, ai: AresBot) -> None:
        self._ai: AresBot = ai

    def army_center_mass(self) -> Optional[Point2]:
        """Center-of-mass of all own combat units."""
        combat_units: list[Unit] = [
            u for u in self._ai.all_own_units
            if u.type_id in COMBAT_UNIT_TYPES
        ]
        if not combat_units:
            return None
        return unit_list_center(combat_units)

    def retreat_direction(self, from_pos: Point2) -> Point2:
        """Unit vector pointing away from enemy start location."""
        enemy_start: Point2 = self._ai.enemy_start_locations[0]
        dx: float = from_pos.x - enemy_start.x
        dy: float = from_pos.y - enemy_start.y
        dist: float = (dx * dx + dy * dy) ** 0.5
        if dist < 0.1:
            return Point2((1.0, 0.0))
        return Point2((dx / dist, dy / dist))

    def hold_behind_lines(
        self,
        unit: Unit,
        maneuver: CombatManeuver,
        avoid_grid,
    ) -> None:
        """Move a unit to a rally point behind the main army."""
        army_center: Optional[Point2] = self.army_center_mass()
        if army_center is not None:
            retreat_dir: Point2 = self.retreat_direction(army_center)
            rally: Point2 = army_center + retreat_dir * 3.0
            maneuver.add(
                PathUnitToTarget(
                    unit=unit, grid=avoid_grid, target=rally, success_at_distance=2.0
                )
            )


def unit_list_center(units: list[Unit]) -> Point2:
    """Simple center-of-mass for a list of units."""
    if not units:
        return Point2((0.0, 0.0))
    x: float = sum(u.position.x for u in units) / len(units)
    y: float = sum(u.position.y for u in units) / len(units)
    return Point2((x, y))