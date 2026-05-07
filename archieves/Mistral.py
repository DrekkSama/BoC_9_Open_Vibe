from typing import Optional, List, Dict
from scipy.spatial import distance
import numpy as np

from ares import AresBot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from cython_extensions import cy_distance_to, cy_closest_to, cy_in_attack_range, cy_find_aoe_position


class MicroBot(AresBot):
    """Zerg micro-only bot focusing on unit control without macro."""

    def __init__(self, game_step_override: Optional[int] = None):
        super().__init__(game_step_override)

    async def on_step(self, iteration: int) -> None:
        await super(MicroBot, self).on_step(iteration)
        
        # Collect friendly units by type
        zerglings: Units = self.get_units_by_type([UnitTypeId.ZERGLING])
        roaches: Units = self.get_units_by_type([UnitTypeId.ROACH])
        banelings: Units = self.get_units_by_type([UnitTypeId.BANELING])
        queens: Units = self.get_units_by_type([UnitTypeId.QUEEN])
        
        # Collect visible enemy units
        enemy_units: Units = self.enemy_units
        
        # Execute micro logic in priority order
        self.control_queens(queens, enemy_units)
        self.control_banelings(banelings, enemy_units)
        self.control_roaches(roaches, enemy_units)
        self.control_zerglings(zerglings, enemy_units)

    def get_units_by_type(self, unit_types: List[UnitTypeId]) -> Units:
        """Helper to get friendly units of specific types."""
        return self.units.filter(lambda unit: unit.type_id in unit_types)

    def control_queens(self, queens: Units, enemy_units: Units) -> None:
        """Control Queens: heal friendly units below 40% HP or attack with stutter-step."""
        for queen in queens:
            # Check if any friendly unit needs healing
            friendly_units = self.units.exclude_type([UnitTypeId.EGG, UnitTypeId.LARVA])
            target_to_heal = self.find_unit_to_heal(friendly_units, queen.position)
            
            if target_to_heal and queen.energy >= 50:
                # Cast Transfuse on the target
                queen(AbilityId.TRANSFUSION_TRANSFUSION, target_to_heal)
            else:
                # Attack with stutter-step
                self.stutter_step_attack(queen, enemy_units)

    def find_unit_to_heal(self, friendly_units: Units, queen_position: Point2) -> Optional[Unit]:
        """Find a friendly unit below 40% HP to heal, prioritizing high-value units."""
        candidates = [
            unit for unit in friendly_units
            if unit.health_percentage < 0.4 and unit.type_id != UnitTypeId.ZERGLING
        ]
        
        if not candidates:
            candidates = [
                unit for unit in friendly_units
                if unit.health_percentage < 0.4
            ]
        
        if candidates:
            # Prioritize closest unit to the queen
            return cy_closest_to((queen_position.x, queen_position.y), candidates)
        return None

    def control_banelings(self, banelings: Units, enemy_units: Units) -> None:
        """Control Banelings: detonate only when 2+ enemies are within splash radius."""
        if not enemy_units:
            return
        
        for baneling in banelings:
            # Find optimal detonation position
            detonation_pos = cy_find_aoe_position(
                2.2,
                enemy_units,
                2,
                set()
            )
            
            if detonation_pos is not None:
                # Move to detonation position
                baneling.move(Point2(detonation_pos))
            else:
                # Hold position if no good target
                pass

    def control_roaches(self, roaches: Units, enemy_units: Units) -> None:
        """Control Roaches: stutter-step kiting and focus-fire assignment."""
        if not enemy_units:
            return
        
        # Assign targets to Roaches
        target_assignment = self.assign_focus_fire_targets(roaches, enemy_units)
        
        for roach in roaches:
            target = target_assignment.get(roach.tag)
            if target and roach.weapon_ready:
                roach.attack(target)
            else:
                # Stutter-step backward if on cooldown
                if roach.weapon_cooldown > 0:
                    retreat_point = self.kite_point(roach.position, target.position if target else enemy_units[0].position)
                    roach.move(retreat_point)

    def assign_focus_fire_targets(self, attackers: Units, targets: Units) -> Dict[int, Unit]:
        """Assign targets to attackers to minimize overkill."""
        assignment = {}
        
        # Sort targets by distance to make assignment deterministic
        sorted_targets = sorted(targets, key=lambda t: t.health + t.shield)
        
        for target in sorted_targets:
            required_attackers = self.calculate_required_attackers(attackers, target)
            
            # Assign closest attackers to this target
            unassigned_attackers = [a for a in attackers if a.tag not in assignment]
            if not unassigned_attackers:
                break
            
            for attacker in sorted(
                unassigned_attackers,
                key=lambda a: cy_distance_to((a.position.x, a.position.y), (target.position.x, target.position.y))
            )[:required_attackers]:
                assignment[attacker.tag] = target
        
        return assignment

    def calculate_required_attackers(self, attackers: Units, target: Unit) -> int:
        """Calculate how many attackers are needed to kill the target."""
        if not attackers:
            return 0
        
        # Use the first attacker's damage as representative
        sample_attacker = attackers[0]
        damage_per_shot = sample_attacker.calculate_damage_vs_target(target)
        
        # Extract the first damage value from the tuple
        if isinstance(damage_per_shot, tuple):
            damage_per_shot = damage_per_shot[0] if damage_per_shot else 0
        
        if damage_per_shot == 0:
            return 0
        
        # Calculate shots needed
        health_total = target.health + target.shield
        shots_needed = max(1, int(health_total / damage_per_shot))
        
        return min(shots_needed, len(attackers))

    def control_zerglings(self, zerglings: Units, enemy_units: Units) -> None:
        """Control Zerglings: attack-move closest enemy, dodge AOE."""
        if not enemy_units:
            return
        
        # Get avoidance grid for AOE dodging
        avoidance_grid = self.mediator.get_ground_avoidance_grid
        
        for zergling in zerglings:
            # Find closest enemy
            closest_enemy = cy_closest_to((zergling.position.x, zergling.position.y), enemy_units)
            
            if closest_enemy:
                # Check if AOE is nearby
                if self.is_aoe_nearby(zergling.position, avoidance_grid):
                    # Retreat from AOE
                    retreat_point = self.kite_point(zergling.position, closest_enemy.position)
                    zergling.move(retreat_point)
                else:
                    # Attack-move toward closest enemy
                    zergling.attack(closest_enemy)

    def is_aoe_nearby(self, position: Point2, avoidance_grid: np.ndarray, radius: float = 5.0) -> bool:
        """Check if there's dangerous AOE nearby using the avoidance grid."""
        # Check a small area around the unit
        x, y = int(position.x), int(position.y)
        for dx in range(-int(radius), int(radius) + 1):
            for dy in range(-int(radius), int(radius) + 1):
                if 0 <= x + dx < avoidance_grid.shape[0] and 0 <= y + dy < avoidance_grid.shape[1]:
                    if avoidance_grid[x + dx][y + dy] > 1.0:
                        return True
        return False

    def kite_point(self, attacker_pos: Point2, target_pos: Point2, distance: float = 3.0) -> Point2:
        """Calculate a retreat point for kiting away from the target."""
        direction = (attacker_pos - target_pos).normalized
        return attacker_pos + direction * distance

    def stutter_step_attack(self, unit: Unit, enemy_units: Units) -> None:
        """Stutter-step attack logic for ranged units."""
        if not enemy_units:
            return
        
        closest_enemy = cy_closest_to((unit.position.x, unit.position.y), enemy_units)
        
        if unit.weapon_ready and closest_enemy:
            unit.attack(closest_enemy)
        elif unit.weapon_cooldown > 0:
            # Move backward while on cooldown
            retreat_point = self.kite_point(unit.position, closest_enemy.position if closest_enemy else enemy_units[0].position)
            unit.move(retreat_point)

    """
    Can use `python-sc2` hooks as usual, but make a call the inherited method in the superclass
    Examples:
    """
    # async def on_start(self) -> None:
    #     await super(MicroBot, self).on_start()
    #
    #     # on_start logic here ...
    #
    # async def on_end(self, game_result: Result) -> None:
    #     await super(MicroBot, self).on_end(game_result)
    #
    #     # custom on_end logic here ...
    #
    # async def on_building_construction_complete(self, unit: Unit) -> None:
    #     await super(MicroBot, self).on_building_construction_complete(unit)
    #
    #     # custom on_building_construction_complete logic here ...
    #
    # async def on_unit_created(self, unit: Unit) -> None:
    #     await super(MicroBot, self).on_unit_created(unit)
    #
    #     # custom on_unit_created logic here ...
    #
    # async def on_unit_destroyed(self, unit_tag: int) -> None:
    #     await super(MicroBot, self).on_unit_destroyed(unit_tag)
    #
    #     # custom on_unit_destroyed logic here ...
    #
    # async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
    #     await super(MicroBot, self).on_unit_took_damage(unit, amount_damage_taken)
    #
    #     # custom on_unit_took_damage logic here ...
