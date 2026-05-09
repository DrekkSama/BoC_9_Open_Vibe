# Purpose: Core army control logic — squad dispatch, engagement tracking, micro entry point
# Key Decisions: Use ARES squad system for grouping, hysteresis for engagement stability
# Limitations: No ML-based engagement decisions, no neural parasite yet

from typing import Optional

import numpy as np
from cython_extensions import cy_closest_to
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.consts import UnitRole, UnitTreeQueryType
from ares.managers.squad_manager import UnitSquad

from bot.combat.unit_micro import UnitMicro
from bot.combat.formation import Formation

# ── Constants ────────────────────────────────────────────────────────────────
ATTACKING_SQUAD_RADIUS: float = 9.0
DEFENDER_SQUAD_RADIUS: float = 6.0
ENGAGEMENT_HYSTERESIS_SECONDS: float = 5.0
ENGAGEMENT_PRUNE_SECONDS: float = 30.0


class CombatManager:
    """Orchestrates squad-based micro dispatch and engagement tracking."""

    def __init__(self, ai: AresBot) -> None:
        self._ai: AresBot = ai
        self._micro: UnitMicro = UnitMicro(ai)
        self._formation: Formation = Formation(ai)
        # Per-squad engagement tracking: squad_center_key -> last_engage_time
        self._squad_engaged: dict[tuple[float, float], float] = {}

    def step(self, forces: Units) -> None:
        """Main micro dispatch using ARES squad system."""
        avoid_grid: np.ndarray = self._ai.mediator.get_ground_avoidance_grid
        target: Point2 = self._ai.attack_target

        squads: list[UnitSquad] = self._ai.mediator.get_squads(
            role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS
        )

        for squad in squads:
            squad_units: list[Unit] = squad.squad_units
            if not squad_units:
                continue

            # Separate units by type within the squad
            zerglings: list[Unit] = []
            roaches: list[Unit] = []
            ravagers: list[Unit] = []
            banelings: list[Unit] = []
            queens: list[Unit] = []
            hydras: list[Unit] = []
            infestors: list[Unit] = []

            for unit in squad_units:
                tid = unit.type_id
                if tid == UnitID.ZERGLING:
                    zerglings.append(unit)
                elif tid == UnitID.ROACH:
                    roaches.append(unit)
                elif tid == UnitID.RAVAGER:
                    ravagers.append(unit)
                elif tid == UnitID.BANELING:
                    banelings.append(unit)
                elif tid == UnitID.QUEEN:
                    queens.append(unit)
                elif tid == UnitID.HYDRALISK:
                    hydras.append(unit)
                elif tid == UnitID.INFESTOR:
                    infestors.append(unit)

            # Per-unit distance queries for individual micro
            near_enemy: dict[int, Units] = self._ai.mediator.get_units_in_range(
                start_points=squad_units,
                distances=15,
                query_tree=UnitTreeQueryType.EnemyGround,
                return_as_dict=True,
            )

            enemies: list[Unit] = [
                u for u in self._ai.enemy_units
                if not u.is_memory and (not u.is_cloaked or u.is_revealed)
            ]

            # ── Engagement tracking with hysteresis ─────────────────────────
            squad_pos: Point2 = squad.squad_position
            squad_key: tuple[float, float] = (round(squad_pos.x, 0), round(squad_pos.y, 0))
            near_enemy_squad: Units = self._ai.mediator.get_units_in_range(
                start_points=[squad_pos],
                distances=15,
                query_tree=UnitTreeQueryType.EnemyGround,
            )[0]
            if len(near_enemy_squad) > 0:
                self._squad_engaged[squad_key] = self._ai.time

            # ── Execution order: Queens → Banelings → Ravagers → Roaches → Hydras → Infestors → Lings
            self._micro.control_queens(queens, enemies, avoid_grid)
            self._micro.control_banelings(banelings, enemies, avoid_grid)
            self._micro.control_ravagers(ravagers, enemies, avoid_grid, near_enemy)
            self._micro.control_roaches(roaches, enemies, avoid_grid, near_enemy)
            self._micro.control_hydras(hydras, enemies, avoid_grid, near_enemy)
            self._micro.control_infestors(infestors, enemies, avoid_grid, near_enemy)
            self._micro.control_zerglings(zerglings, enemies, avoid_grid, near_enemy, target)

        # Prune stale engagement entries
        cutoff: float = self._ai.time - ENGAGEMENT_PRUNE_SECONDS
        self._squad_engaged = {
            k: v for k, v in self._squad_engaged.items() if v > cutoff
        }

    def is_squad_engaged(self, squad_key: tuple[float, float]) -> bool:
        """Check if a squad was recently engaged (within hysteresis window)."""
        if squad_key not in self._squad_engaged:
            return False
        return self._ai.time - self._squad_engaged[squad_key] < ENGAGEMENT_HYSTERESIS_SECONDS