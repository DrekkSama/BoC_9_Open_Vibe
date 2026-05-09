# Purpose: Queen role assignment and control using ARES Queen-specific roles
# Key Decisions: Simplified from Clicadinha example — no custom mediator,
#   direct method calls. Queens get QUEEN_DEFENCE by default, then are
#   dynamically reassigned to QUEEN_INJECT / QUEEN_CREEP as needed.
#   Each queen gets exactly ONE behavior registration per frame (no conflicts).
# Limitations: No nydus queen support, no offensive queen mode yet

import numpy as np
from cython_extensions import cy_distance_to, cy_in_attack_range, cy_pick_enemy_target
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe,
    PathUnitToTarget,
    QueenSpreadCreep,
    ShootTargetInRange,
    StutterUnitBack,
    TumorSpreadCreep,
    UseTransfuse,
)
from ares.consts import UnitRole

# ── Constants ────────────────────────────────────────────────────────────────
QUEEN_INJECT_ENERGY: float = 25.0
CREEP_TUMOR_ENERGY: float = 25.0
TRANSFUSE_ENERGY: float = 50.0
# How close a queen must be to a townhall for inject
INJECT_RANGE: float = 10.0
# Max creep spreaders when not rushed/aggressive
MAX_CREEP_SPREADERS: int = 5
# Min queens before we start assigning inject/creep roles
MIN_QUEENS_FOR_SPECIALIZATION: int = 4


class QueenManager:
    """Manages Queen roles and behavior using ARES role system.

    Role hierarchy (from queen_role_controller example):
    - New queens default to QUEEN_DEFENCE
    - Steal from QUEEN_DEFENCE for inject/creep as needed
    - If no longer needed in a role, return to QUEEN_DEFENCE
    """

    def __init__(self, ai: AresBot) -> None:
        self._ai: AresBot = ai
        # Track which queen tag is assigned to which townhall tag for inject
        self._inject_queen_to_th: dict[int, int] = {}

    def assign_new_queen(self, queen: Unit) -> None:
        """Assign a newly created queen to QUEEN_DEFENCE role.

        Called from main.py on_unit_created for Queen units.
        The role controller will dynamically reassign as needed.
        """
        self._ai.mediator.assign_role(tag=queen.tag, role=UnitRole.QUEEN_DEFENCE)

    def update(self) -> None:
        """Run every frame: adjust roles, then execute queen behaviors."""
        # Get queens by current role
        inject_queens: Units = self._ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_INJECT
        )
        creep_queens: Units = self._ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_CREEP
        )
        defence_queens: Units = self._ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_DEFENCE
        )

        # Adjust role assignments
        self._manage_inject_role(defence_queens, inject_queens)
        self._manage_creep_role(defence_queens, creep_queens)

        # Re-fetch after role adjustments (roles may have changed)
        inject_queens = self._ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_INJECT
        )
        creep_queens = self._ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_CREEP
        )
        defence_queens = self._ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_DEFENCE
        )

        # Execute behaviors per role
        self._control_inject_queens(inject_queens)
        self._control_creep_queens(creep_queens)
        self._control_defence_queens(defence_queens)

        # Tumor spread (always run for existing tumors)
        self._spread_tumors()

    # ── Role Management ─────────────────────────────────────────────────────

    def _manage_inject_role(
        self, defence_queens: Units, inject_queens: Units
    ) -> None:
        """Assign/unassign inject queens based on game state.

        Simplified from example: 1 inject queen when not rushed and
        we have enough queens. No inject during rush (all defend).
        """
        num_required: int = self._required_injectors

        # Assign: steal from defence if we need more injectors
        if num_required and len(inject_queens) < num_required and defence_queens:
            # Find a townhall that doesn't have an inject queen yet
            th_tags_taken: set[int] = set(self._inject_queen_to_th.values())
            available_ths: list[Unit] = [
                th for th in self._ai.townhalls
                if th.build_progress > 0.95 and th.tag not in th_tags_taken
            ]
            if available_ths:
                queen: Unit = defence_queens[0]
                # Pick closest townhall to this queen
                closest_th: Unit = min(
                    available_ths,
                    key=lambda th: cy_distance_to(queen.position, th.position),
                )
                self._ai.mediator.assign_role(
                    tag=queen.tag, role=UnitRole.QUEEN_INJECT
                )
                self._inject_queen_to_th[queen.tag] = closest_th.tag

        # Unassign: too many injectors
        if len(inject_queens) > num_required:
            num_to_unassign: int = len(inject_queens) - num_required
            for i in range(num_to_unassign):
                tag: int = inject_queens[i].tag
                self._ai.mediator.assign_role(tag=tag, role=UnitRole.QUEEN_DEFENCE)
                self._inject_queen_to_th.pop(tag, None)

        # Clean up stale mappings (queen died or role changed)
        active_tags: set[int] = {q.tag for q in inject_queens}
        stale: list[int] = [
            t for t in self._inject_queen_to_th if t not in active_tags
        ]
        for tag in stale:
            del self._inject_queen_to_th[tag]

    def _manage_creep_role(
        self, defence_queens: Units, creep_queens: Units
    ) -> None:
        """Assign/unassign creep spreaders based on game state."""
        num_required: int = self._required_creep_spreaders

        # Assign: steal from defence
        if num_required and len(creep_queens) < num_required and defence_queens:
            queen: Unit = defence_queens[0]
            self._ai.mediator.assign_role(
                tag=queen.tag, role=UnitRole.QUEEN_CREEP
            )

        # Unassign: too many creep queens
        if len(creep_queens) > num_required:
            num_to_unassign: int = len(creep_queens) - num_required
            for i in range(num_to_unassign):
                tag: int = creep_queens[i].tag
                self._ai.mediator.assign_role(tag=tag, role=UnitRole.QUEEN_DEFENCE)

    @property
    def _required_injectors(self) -> int:
        """How many inject queens we need.

        From the example: return 1 when we have enough queens and
        aren't rushed/aggressive. 0 otherwise.
        """
        num_queens: int = len(self._ai.units(UnitID.QUEEN))
        if num_queens < MIN_QUEENS_FOR_SPECIALIZATION:
            return 0
        # Don't inject if we have 5+ bases (too many to manage)
        if len(self._ai.townhalls) >= 5:
            return 0
        return 1

    @property
    def _required_creep_spreaders(self) -> int:
        """How many creep queens we need.

        From the example: scale with queen count, but cap.
        Don't spread creep when few queens.
        """
        num_queens: int = len(self._ai.units(UnitID.QUEEN))
        if num_queens < MIN_QUEENS_FOR_SPECIALIZATION:
            return 0
        # Scale with queen count, cap at MAX_CREEP_SPREADERS
        return min(MAX_CREEP_SPREADERS, max(1, num_queens - 3))

    # ── Queen Behaviors ─────────────────────────────────────────────────────

    def _control_inject_queens(self, inject_queens: Units) -> None:
        """Inject queens: move to assigned townhall, inject when ready.

        Uses direct commands for inject (not CombatManeuver) because
        inject is a non-combat ability that should not conflict with
        combat behaviors. We only issue one command per frame per queen.
        """
        for queen in inject_queens:
            th_tag: int = self._inject_queen_to_th.get(queen.tag, -1)
            assigned_th: Unit | None = None
            for th in self._ai.townhalls:
                if th.tag == th_tag:
                    assigned_th = th
                    break

            # If we have energy and an assigned TH, try to inject
            if queen.energy >= QUEEN_INJECT_ENERGY and assigned_th:
                if not assigned_th.has_buff(BuffId.QUEENSPAWNLARVATIMER):
                    queen(AbilityId.EFFECT_INJECTLARVA, assigned_th)
                    continue

            # If assigned TH no longer exists, reassign
            if assigned_th is None:
                th_tags_taken: set[int] = set(self._inject_queen_to_th.values())
                available_ths: list[Unit] = [
                    th for th in self._ai.townhalls
                    if th.build_progress > 0.95 and th.tag not in th_tags_taken
                ]
                if available_ths:
                    # Pick closest TH to this queen
                    new_th: Unit = min(
                        available_ths,
                        key=lambda th: cy_distance_to(queen.position, th.position),
                    )
                    self._inject_queen_to_th[queen.tag] = new_th.tag
                    assigned_th = new_th

            # Move toward assigned TH if not in range
            if assigned_th:
                dist: float = cy_distance_to(queen.position, assigned_th.position)
                if dist > INJECT_RANGE:
                    queen.move(assigned_th.position)

    def _control_creep_queens(self, creep_queens: Units) -> None:
        """Creep queens: spread tumors, transfuse if needed.

        Each queen gets exactly ONE CombatManeuver per frame.
        QueenSpreadCreep handles movement to creep edge internally.
        """
        grid: np.ndarray = self._ai.mediator.get_ground_avoidance_grid

        for queen in creep_queens:
            maneuver: CombatManeuver = CombatManeuver()

            # Priority 1: Stay safe from AoE
            maneuver.add(KeepUnitSafe(unit=queen, grid=grid))

            # Priority 2: Transfuse injured friendlies
            if queen.energy >= TRANSFUSE_ENERGY:
                maneuver.add(UseTransfuse(unit=queen, targets=self._ai.all_own_units))

            # Priority 3: Spread creep (handles movement to edge internally)
            if queen.energy >= CREEP_TUMOR_ENERGY:
                maneuver.add(
                    QueenSpreadCreep(unit=queen, cancel_if_close_enemy=True)
                )

            self._ai.register_behavior(maneuver)

    def _control_defence_queens(self, defence_queens: Units) -> None:
        """Defence queens: stay near bases, transfuse, attack enemies.

        Each queen gets exactly ONE CombatManeuver per frame.
        No separate QueenSpreadCreep registration (avoids double-command).
        """
        if not defence_queens:
            return

        grid: np.ndarray = self._ai.mediator.get_ground_avoidance_grid

        for queen in defence_queens:
            maneuver: CombatManeuver = CombatManeuver()

            # Priority 1: Stay safe from AoE
            maneuver.add(KeepUnitSafe(unit=queen, grid=grid))

            # Priority 2: Transfuse injured friendlies
            if queen.energy >= TRANSFUSE_ENERGY:
                maneuver.add(UseTransfuse(unit=queen, targets=self._ai.all_own_units))

            # Priority 3: Attack enemies near base
            enemies_near_base: Units = self._enemies_near_bases()
            if enemies_near_base:
                if in_range := cy_in_attack_range(queen, enemies_near_base):
                    maneuver.add(ShootTargetInRange(unit=queen, targets=in_range))
                enemy_target: Unit = cy_pick_enemy_target(enemies_near_base)
                maneuver.add(
                    StutterUnitBack(unit=queen, target=enemy_target, grid=grid)
                )
            else:
                # No threats: stay near closest townhall
                if self._ai.townhalls:
                    closest_th: Unit = min(
                        self._ai.townhalls,
                        key=lambda th: cy_distance_to(queen.position, th.position),
                    )
                    dist_to_th: float = cy_distance_to(
                        queen.position, closest_th.position
                    )
                    if dist_to_th > INJECT_RANGE:
                        maneuver.add(
                            PathUnitToTarget(
                                unit=queen, grid=grid, target=closest_th.position
                            )
                        )
                    elif queen.energy >= CREEP_TUMOR_ENERGY + QUEEN_INJECT_ENERGY:
                        # Idle near base with excess energy: spread creep
                        maneuver.add(
                            QueenSpreadCreep(
                                unit=queen, cancel_if_close_enemy=True
                            )
                        )

            self._ai.register_behavior(maneuver)

    def _spread_tumors(self) -> None:
        """Spread existing creep tumors toward enemy."""
        target: Point2 = self._ai.attack_target
        tumors: Units = self._ai.structures(
            {UnitID.CREEPTUMORBURROWED, UnitID.CREEPTUMORQUEEN}
        )
        for tumor in tumors:
            self._ai.register_behavior(TumorSpreadCreep(unit=tumor, target=target))

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _enemies_near_bases(self) -> Units:
        """Get enemy units near any of our townhalls.

        Uses a single query per townhall. Perf note: O(bases * enemies)
        but typically 1-3 bases and <50 visible enemies, so fine.
        """
        result: list[Unit] = []
        seen_tags: set[int] = set()
        for th in self._ai.townhalls:
            near_th: Units = self._ai.enemy_units.closer_than(
                15.0, th.position
            )
            for unit in near_th:
                if unit.tag not in seen_tags:
                    result.append(unit)
                    seen_tags.add(unit.tag)
        return Units(result, self._ai)
