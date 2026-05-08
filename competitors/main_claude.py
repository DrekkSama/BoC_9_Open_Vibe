"""
Terran Tier-1 Micro Bot
Purpose: Micro-only bot for SCV, Marine, Reaper, Marauder, Hellion/Hellbat
Key Decisions: Overkill-aware focus fire, stutter-step kiting, melee chase
Limitations: No macro, production, or economy
"""
from typing import Optional

from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import AMove, KeepUnitSafe, StutterUnitBack
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units


class MainClaudeBot(AresBot):
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
        
        # 1. Collect friendly units (tier-1 ground only)
        friendly_units = self._get_friendly_combat_units()
        if not friendly_units:
            return
        
        # 2. Collect visible enemy units
        enemy_units = self.enemy_units.filter(lambda u: not u.is_flying)
        if not enemy_units:
            return
        
        # 3. Split units: ranged vs melee
        ranged_units, melee_units = self._split_ranged_melee(friendly_units)
        
        # 4. Micro ranged units (stutter-step + focus fire)
        if ranged_units:
            self._micro_ranged_units(ranged_units, enemy_units)
        
        # 5. Micro melee units (attack-move to closest enemy)
        if melee_units:
            self._micro_melee_units(melee_units, enemy_units)

    def _get_friendly_combat_units(self) -> Units:
        """Get all tier-1 Terran ground units we control."""
        combat_types = {
            UnitTypeId.SCV,
            UnitTypeId.MARINE,
            UnitTypeId.REAPER,
            UnitTypeId.MARAUDER,
            UnitTypeId.HELLION,
            UnitTypeId.HELLIONTANK,  # Hellbat
        }
        return self.units.filter(lambda u: u.type_id in combat_types)

    def _split_ranged_melee(self, units: Units) -> tuple[Units, Units]:
        """Split units into ranged and melee groups."""
        melee_types = {UnitTypeId.SCV, UnitTypeId.HELLIONTANK}
        melee = units.filter(lambda u: u.type_id in melee_types)
        ranged = units.filter(lambda u: u.type_id not in melee_types)
        return ranged, melee

    def _micro_ranged_units(self, ranged_units: Units, enemies: Units) -> None:
        """
        Ranged micro: overkill-aware focus fire + stutter-step.
        Assign just-enough units to each enemy, excess retarget next closest.
        """
        # Calculate target assignments: {unit_tag: target_unit}
        assignments = self._assign_focus_fire(ranged_units, enemies)
        
        # Issue behaviors per unit
        for unit in ranged_units:
            target = assignments.get(unit.tag)
            if not target:
                continue
            
            # StutterUnitBack handles both: shoot when ready, kite when cooling
            maneuver = CombatManeuver()
            maneuver.add(
                StutterUnitBack(
                    unit=unit,
                    target=target,
                    kite_via_pathing=True,
                    grid=self.mediator.get_ground_grid,
                )
            )
            self.register_behavior(maneuver)

    def _micro_melee_units(self, melee_units: Units, enemies: Units) -> None:
        """Melee micro: attack-move to closest enemy."""
        for unit in melee_units:
            closest_enemy = enemies.closest_to(unit)
            if closest_enemy:
                maneuver = CombatManeuver()
                maneuver.add(AMove(unit=unit, target=closest_enemy))
                self.register_behavior(maneuver)

    def _assign_focus_fire(self, ranged_units: Units, enemies: Units) -> dict[int, Unit]:
        """
        Overkill-aware focus fire assignment.
        Returns: {unit_tag: target_unit}
        
        Algorithm:
        1. Sort enemies by distance (closest first)
        2. For each enemy, calculate shots needed to kill
        3. Assign just-enough units to that enemy
        4. Remaining units target next enemy
        """
        assignments: dict[int, Unit] = {}
        available_units = list(ranged_units)
        
        # Sort enemies by proximity to our ranged army center
        army_center = ranged_units.center
        sorted_enemies = sorted(enemies, key=lambda e: e.distance_to(army_center))
        
        for enemy in sorted_enemies:
            if not available_units:
                break
            
            # Calculate how many shots needed to kill this enemy
            shots_needed = self._calculate_shots_needed(available_units[0], enemy)
            
            # Assign up to `shots_needed` units to this enemy
            units_to_assign = available_units[:shots_needed]
            for unit in units_to_assign:
                assignments[unit.tag] = enemy
            
            # Remove assigned units from available pool
            available_units = available_units[shots_needed:]
        
        # Any remaining units target the closest enemy
        for unit in available_units:
            closest = enemies.closest_to(unit)
            if closest:
                assignments[unit.tag] = closest
        
        return assignments

    def _calculate_shots_needed(self, attacker: Unit, target: Unit) -> int:
        """
        Calculate how many shots from this unit type are needed to kill target.
        Returns at least 1.
        """
        damage_per_shot = attacker.calculate_damage_vs_target(target)[0]
        if damage_per_shot <= 0:
            return 1  # Fallback
        
        target_hp = target.health + target.shield
        shots = int(target_hp / damage_per_shot) + (1 if target_hp % damage_per_shot > 0 else 0)
        return max(1, shots)

    """
    Can use `python-sc2` hooks as usual, but make a call the inherited method in the superclass
    Examples:
    """
    # async def on_start(self) -> None:
    #     await super(MyBot, self).on_start()
    #
    #     # on_start logic here ...
    #
    # async def on_end(self, game_result: Result) -> None:
    #     await super(MyBot, self).on_end(game_result)
    #
    #     # custom on_end logic here ...
    #
    # async def on_building_construction_complete(self, unit: Unit) -> None:
    #     await super(MyBot, self).on_building_construction_complete(unit)
    #
    #     # custom on_building_construction_complete logic here ...
    #
    # async def on_unit_created(self, unit: Unit) -> None:
    #     await super(MyBot, self).on_unit_created(unit)
    #
    #     # custom on_unit_created logic here ...
    #
    # async def on_unit_destroyed(self, unit_tag: int) -> None:
    #     await super(MyBot, self).on_unit_destroyed(unit_tag)
    #
    #     # custom on_unit_destroyed logic here ...
    #
    # async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
    #     await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)
    #
    #     # custom on_unit_took_damage logic here ...
