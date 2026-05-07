from typing import Optional

from ares import AresBot
from ares.behaviors.combat.individual import (
    AMove,
    KeepUnitSafe,
    ShootTargetInRange,
    StutterUnitBack,
    UseTransfuse,
)
from ares.behaviors.combat.combat_maneuver import CombatManeuver
from ares.consts import UnitRole
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units
from cython_extensions import (
    cy_distance_to, cy_in_attack_range, cy_attack_ready, cy_pick_enemy_target,
    cy_find_aoe_position, cy_closest_to, cy_point_below_value
)
import numpy as np

# Import our constants
from competitors.Qwen_constants import (
    ZERGLING, ROACH, BANELING, QUEEN,
    ATTACKING, BASE_DEFENDER,
    BANELING_SPLASH_RADIUS, QUEEN_HEAL_THRESHOLD, QUEEN_HEAL_RANGE
)

class QwenBot(AresBot):
    def __init__(self, game_step_override: Optional[int] = None):
        """Initiate custom bot

        Parameters
        ----------
        game_step_override :
            If provided, set the game_step to this value regardless of how it was
            specified elsewhere
        """
        super().__init__(game_step_override)

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)
        
        # Get our units and enemy units
        our_units: Units = self.units
        enemy_units: Units = self.enemy_units
        
        # Filter for the units we want to control (Zerglings, Roaches, Banelings, Queens)
        zerglings: Units = our_units(ZERGLING)
        roaches: Units = our_units(ROACH)
        banelings: Units = our_units(BANELING)
        queens: Units = our_units(QUEEN)
        
        # Control Zerglings - melee surround and chase behavior
        self._control_zerglings(zerglings, enemy_units)
        
        # Control Roaches - ranged stutter-step kiting behavior
        self._control_roaches(roaches, enemy_units)
        
        # Control Banelings - suicide AOE behavior
        self._control_banelings(banelings, enemy_units)
        
        # Control Queens - ranged + heal support behavior
        self._control_queens(queens, enemy_units, our_units)
        
    def _control_zerglings(self, zerglings: Units, enemy_units: Units) -> None:
        """Control Zerglings with melee surround and chase behavior"""
        if not zerglings or not enemy_units:
            return
            
        for zergling in zerglings:
            # Get the closest enemy unit
            if enemy_units:
                closest_enemy = min(enemy_units, key=lambda e: cy_distance_to(zergling.position, e.position))
            else:
                continue
            if not closest_enemy:
                continue
                
            # Check if we should retreat due to enemy AOE
            ground_avoidance_grid = self.mediator.get_ground_avoidance_grid
            is_safe = cy_point_below_value(grid=ground_avoidance_grid, position=zergling.position.rounded, weight_safety_limit=1.0)
            
            # If enemy has AOE nearby, retreat first, then re-engage
            if not is_safe:
                # Move away from danger first, then re-engage
                maneuver = CombatManeuver()
                maneuver.add(KeepUnitSafe(zergling, ground_avoidance_grid, 1.0))
                self.register_behavior(maneuver)
            else:
                # Attack-move toward closest enemy
                maneuver = CombatManeuver()
                maneuver.add(AMove(zergling, closest_enemy))
                self.register_behavior(maneuver)
    
    def _control_roaches(self, roaches: Units, enemy_units: Units) -> None:
        """Control Roaches with ranged stutter-step kiting behavior"""
        if not roaches or not enemy_units:
            return
            
        for roach in roaches:
            # Get a target for attack
            target = self._get_best_target_for_roach(roach, enemy_units)
            
            # Simple approach: always try to attack if we have a target
            if target is not None:
                # Find targets that are in range and prioritize low HP targets
                in_range_enemies = []
                for e in enemy_units:
                    # Use a simple distance check instead of cy_in_attack_range to avoid issues
                    if cy_distance_to(roach.position, e.position) <= roach.radius + e.radius + 5.0:  # Approximate attack range
                        in_range_enemies.append(e)
                        
                if in_range_enemies:
                    # Sort by health to prioritize low HP targets for kill confirms
                    in_range_enemies.sort(key=lambda e: e.health + e.shield)
                    target = in_range_enemies[0]
                    maneuver = CombatManeuver()
                    maneuver.add(ShootTargetInRange(roach, [target]))
                    self.register_behavior(maneuver)
                else:
                    # No targets in range, move to closest enemy
                    if enemy_units:
                        closest_enemy = min(enemy_units, key=lambda e: cy_distance_to(roach.position, e.position))
                        if closest_enemy:
                            maneuver = CombatManeuver()
                            maneuver.add(AMove(roach, closest_enemy))
                            self.register_behavior(maneuver)
            else:
                # No target available, move to closest enemy
                if enemy_units:
                    closest_enemy = min(enemy_units, key=lambda e: cy_distance_to(roach.position, e.position))
                    if closest_enemy:
                        maneuver = CombatManeuver()
                        maneuver.add(AMove(roach, closest_enemy))
                        self.register_behavior(maneuver)
    
    def _get_best_target_for_roach(self, roach: Unit, enemy_units: Units) -> Unit:
        """Get the best target for a roach based on damage calculation and distance"""
        if not enemy_units:
            return None
        # Find closest enemy that's in range to prioritize
        # Use a simple distance check instead of cy_in_attack_range to avoid issues
        in_range_enemies = []
        for e in enemy_units:
            # Approximate attack range check using distance
            if cy_distance_to(roach.position, e.position) <= roach.radius + e.radius + 5.0:
                in_range_enemies.append(e)
        if in_range_enemies:
            # Sort by distance to prioritize closer enemies
            in_range_enemies.sort(key=lambda e: cy_distance_to(roach.position, e.position))
            return in_range_enemies[0]
        else:
            # No enemies in range, return closest enemy
            # Use a safer approach to get the closest enemy
            if enemy_units:
                closest_enemy = min(enemy_units, key=lambda e: cy_distance_to(roach.position, e.position))
                return closest_enemy
            return None
    
    def _control_banelings(self, banelings: Units, enemy_units: Units) -> None:
        """Control Banelings with suicide AOE behavior"""
        if not banelings or not enemy_units:
            return
            
        for baneling in banelings:
            # Check if we have a good target to detonate on
            # Find enemies within splash radius (approximately 2.2 tiles)
            splash_radius = BANELING_SPLASH_RADIUS
            nearby_enemies = [e for e in enemy_units if cy_distance_to(baneling.position, e.position) <= splash_radius]
            
            # If we have enough enemies to make it worthwhile, find the best detonation point
            if len(nearby_enemies) >= 2:
                # Find the best position to detonate that hits the most enemies
                aoe_position = cy_find_aoe_position(splash_radius, nearby_enemies, 2)
                if aoe_position is not None:
                    # Move to the detonation point and attack (auto-detonate on contact)
                    maneuver = CombatManeuver()
                    maneuver.add(AMove(baneling, aoe_position))
                    self.register_behavior(maneuver)
                else:
                    # No good position found, hold position
                    maneuver = CombatManeuver()
                    maneuver.add(AMove(baneling, baneling.position))
                    self.register_behavior(maneuver)
            else:
                # Not enough enemies, hold position or move to a rally point behind friendly lines
                maneuver = CombatManeuver()
                maneuver.add(AMove(baneling, baneling.position))
                self.register_behavior(maneuver)
    
    def _control_queens(self, queens: Units, enemy_units: Units, our_units: Units) -> None:
        """Control Queens with ranged + heal support behavior"""
        if not queens or not enemy_units:
            return
            
        for queen in queens:
            # Check if we should heal a friendly unit first
            # Find friendly units below 40% HP
            low_health_friends = [u for u in our_units if u.health_percentage < QUEEN_HEAL_THRESHOLD and u != queen]
            if low_health_friends and queen.energy >= 50:
                # Sort by health to prioritize the most injured units
                low_health_friends.sort(key=lambda u: u.health_percentage)
                target_friend = low_health_friends[0]
                # Check if we're in range to heal
                if cy_distance_to(queen.position, target_friend.position) <= QUEEN_HEAL_RANGE + queen.radius + target_friend.radius:
                    # Use Transfuse on the low health unit
                    maneuver = CombatManeuver()
                    maneuver.add(UseTransfuse(queen, [target_friend]))
                    self.register_behavior(maneuver)
                else:
                    # Not in range, move to heal them
                    maneuver = CombatManeuver()
                    maneuver.add(AMove(queen, target_friend))
                    self.register_behavior(maneuver)
            else:
                # No healing needed, attack-move with stutter-step behavior like Roaches
                # Get a target for attack
                target = self._get_best_target_for_roach(queen, enemy_units)
                
                # Simple approach: always try to attack if we have a target
                if target is not None:
                    # Find targets that are in range and prioritize low HP targets
                    in_range_enemies = []
                    for e in enemy_units:
                        # Use a simple distance check instead of cy_in_attack_range to avoid issues
                        if cy_distance_to(queen.position, e.position) <= queen.radius + e.radius + 7.0:  # Queen attack range
                            in_range_enemies.append(e)
                            
                    if in_range_enemies:
                        # Sort by health to prioritize low HP targets for kill confirms
                        in_range_enemies.sort(key=lambda e: e.health + e.shield)
                        target = in_range_enemies[0]
                        maneuver = CombatManeuver()
                        maneuver.add(ShootTargetInRange(queen, [target]))
                        self.register_behavior(maneuver)
                    else:
                        # No targets in range, move to closest enemy
                        if enemy_units:
                            closest_enemy = min(enemy_units, key=lambda e: cy_distance_to(queen.position, e.position))
                            if closest_enemy:
                                maneuver = CombatManeuver()
                                maneuver.add(AMove(queen, closest_enemy))
                                self.register_behavior(maneuver)
                else:
                    # No target available, move to closest enemy
                    closest_enemy = cy_closest_to(queen.position, enemy_units)
                    if closest_enemy:
                        maneuver = CombatManeuver()
                        maneuver.add(AMove(queen, closest_enemy))
                        self.register_behavior(maneuver)
