# Purpose: Priority target selection and focus-fire assignment
# Key Decisions: Overkill-aware assignment — just enough roaches per target, excess retarget
# Limitations: Only used for roach focus-fire currently

from typing import Optional

from cython_extensions import cy_distance_to
from sc2.position import Point2
from sc2.unit import Unit


def assign_focus_fire(
    roaches: list[Unit],
    enemies: list[Unit],
) -> dict[int, Unit]:
    """Assign just enough roaches per enemy to kill it; excess retarget.

    Returns a dict mapping roach tag -> assigned enemy Unit.
    """
    assignments: dict[int, Unit] = {}
    assigned_roaches: set[int] = set()

    if not roaches or not enemies:
        return assignments

    roach_center: Point2 = _unit_list_center(roaches)
    sorted_enemies: list[Unit] = sorted(
        enemies, key=lambda e: cy_distance_to(roach_center, e.position)
    )

    for enemy in sorted_enemies:
        if len(assigned_roaches) >= len(roaches):
            break

        enemy_hp: float = enemy.health + enemy.shield
        if enemy_hp <= 0:
            continue

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


def _unit_list_center(units: list[Unit]) -> Point2:
    """Simple center-of-mass for a list of units."""
    if not units:
        return Point2((0.0, 0.0))
    x: float = sum(u.position.x for u in units) / len(units)
    y: float = sum(u.position.y for u in units) / len(units)
    return Point2((x, y))