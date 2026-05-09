# Purpose: Unit role assignment and retrieval helpers
# Key Decisions: Default ATTACKING role, DEFENDING for base defense, HARASSING for runby
# Limitations: No dynamic role switching based on game state yet

from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.unit import Unit
from sc2.units import Units

from ares.consts import UnitRole


# Combat unit types that should get combat roles
# Queens excluded — managed by QueenManager with QUEEN_* roles
COMBAT_UNIT_TYPES: set[UnitID] = {
    UnitID.ZERGLING,
    UnitID.BANELING,
    UnitID.ROACH,
    UnitID.RAVAGER,
    UnitID.HYDRALISK,
    UnitID.INFESTOR,
    UnitID.OVERSEER,
    UnitID.ULTRALISK,
    UnitID.BROODLORD,
    UnitID.LURKERMP,
    UnitID.VIPER,
    UnitID.CORRUPTOR,
    UnitID.MUTALISK,
}


def get_attackers(mediator) -> Units:
    """Get all units with ATTACKING role."""
    return mediator.get_units_from_role(role=UnitRole.ATTACKING)


def get_defenders(mediator) -> Units:
    """Get all units with DEFENDING role."""
    return mediator.get_units_from_role(role=UnitRole.DEFENDING)


def get_role_units(mediator, role: UnitRole, unit_type: UnitID = None) -> Units:
    """Get units by role, optionally filtered by unit type."""
    if unit_type:
        return mediator.get_units_from_role(role=role, unit_type=unit_type)
    return mediator.get_units_from_role(role=role)


def set_defense_squad(mediator, n_required: int, base_position) -> None:
    """Slice off nearest units from ATTACKING and assign them DEFENDING role.

    Args:
        mediator: ARES ManagerMediator
        n_required: Number of units needed for defense
        base_position: Position to find nearest units to
    """
    attackers: Units = mediator.get_units_from_role(role=UnitRole.ATTACKING)
    if not attackers:
        return

    # Sort by distance to base, take nearest
    defenders_needed: int = min(n_required, len(attackers))
    sorted_attackers: list[Unit] = sorted(
        attackers,
        key=lambda u: u.distance_to(base_position),
    )

    for unit in sorted_attackers[:defenders_needed]:
        mediator.assign_role(tag=unit.tag, role=UnitRole.DEFENDING)


def release_defenders(mediator) -> None:
    """Return all DEFENDING units back to ATTACKING role."""
    defenders: Units = mediator.get_units_from_role(role=UnitRole.DEFENDING)
    for unit in defenders:
        mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)
