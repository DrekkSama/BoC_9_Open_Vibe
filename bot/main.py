# Purpose: Main AresBot subclass wiring all modules together
# Key Decisions: MacroManager owns economy/tech/upgrades/responses,
#   CombatManager owns army micro, QueenManager owns queen roles.
#   main.py is a thin orchestrator — no game logic lives here.
# Limitations: No ML-based engagement decisions, no neural parasite yet

from typing import Optional

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import AresBot
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole

from bot.combat import CombatManager
from bot.managers.macro_manager import MacroManager
from bot.managers.queen_manager import QueenManager

# ── Constants ────────────────────────────────────────────────────────────────
BEGIN_ATTACK_SUPPLY: float = 6.0

# Unit types that shouldn't be assigned ATTACKING role
# Queens get their own role system via QueenManager
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
    UnitID.QUEEN,  # Queens managed by QueenManager with QUEEN_* roles
}


class GLM_Bot(AresBot):
    """Zerg B2GM Roach Ravager bot using ARES framework."""

    def __init__(self, game_step_override: Optional[int] = None):
        super().__init__(game_step_override)
        self._combat_mgr: Optional[CombatManager] = None
        self._queen_mgr: Optional[QueenManager] = None
        self._macro_mgr: Optional[MacroManager] = None

    @property
    def attack_target(self) -> Point2:
        """Delegate to MacroManager for attack target calculation."""
        if self._macro_mgr is not None:
            return self._macro_mgr.attack_target
        return self.enemy_start_locations[0]

    async def on_start(self) -> None:
        """Called once at the start of the game."""
        await super().on_start()
        self._combat_mgr = CombatManager(self)
        self._queen_mgr = QueenManager(self)
        self._macro_mgr = MacroManager(self)

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)
        if not self.all_own_units:
            return

        # ── Macro (economy, production, tech, upgrades, responses) ──────────
        if self._macro_mgr is not None:
            self._macro_mgr.update()

        # ── Combat (army micro) ─────────────────────────────────────────────
        forces: Units = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)

        if self._macro_mgr is not None and not self._macro_mgr.commenced_attack:
            if self.get_total_supply(forces) >= BEGIN_ATTACK_SUPPLY:
                self._macro_mgr.start_attack()

        if (
            self._macro_mgr is not None
            and self._macro_mgr.commenced_attack
            and forces
            and self._combat_mgr is not None
        ):
            self._combat_mgr.step(forces)

        # ── Queen management (always run) ───────────────────────────────────
        if self._queen_mgr is not None:
            self._queen_mgr.update()

    async def on_unit_created(self, unit: Unit) -> None:
        """Assign combat units to ATTACKING role, Queens to QueenManager."""
        await super().on_unit_created(unit)

        if unit.type_id in ALL_STRUCTURES:
            return
        if unit.type_id in WORKER_TYPES:
            return

        # Queens get their own role system via QueenManager
        if unit.type_id == UnitID.QUEEN:
            if self._queen_mgr is not None:
                self._queen_mgr.assign_new_queen(unit)
            return

        if unit.type_id in IGNORE_ROLE_TYPES:
            return

        self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    # ── Micro (delegated to managers) ────────────────────────────────────────
    # CombatManager: bot/combat/combat.py
    # QueenManager:  bot/managers/queen_manager.py
    # MacroManager:  bot/managers/macro_manager.py


# Alias for run.py compatibility
MicroBot = GLM_Bot
