"""Zerg Test Bot — Modular Zerg opponent for testing threat detection.
Purpose: Controlled Zerg opponent with selectable behavior profiles.
         Each profile tests a specific threat type that PiGBot needs to handle.
Key Decisions: Minimal economy, profiles are toggled via class attributes.
Limitations: No defense, no upgrades, no injects. The notes assume combat losses
             free up supply, so Overlords are morphed whenever supply-blocked.

Profiles:
  - "12_pool_zerg_rush" (default): 12 Pool rush, attacks at 16 Zerglings
  - "fungal_test" (enable_rush=False, enable_fungal=True): Infestor Fungal module
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

# ── Profile: "12_pool_zerg_rush" ────────────────────────────────────────────
# 12  Spawning Pool
# 14  Overlord / drones to 14
# 14  Zergling x3 (waves stream as larvae allow)
# 16  Hatchery (natural)
# 18  Queen
# 20  Zergling x2  → attack once the wave is complete
RUSH_DRONE_CAP: int = 14         # drone target once the pool has started
RUSH_HATCHERY_SUPPLY: int = 16   # expand to the natural
RUSH_QUEEN_SUPPLY: int = 18      # train the first Queen
RUSH_ATTACK_ZERGLINGS: int = 16  # 8 pairs; attack-move when reached


class ZergTestBot(BotAI):
    """Zerg test bot with modular threat behaviors.

    Profiles are selected via class attributes:
      - enable_rush (default True): "12_pool_zerg_rush" profile
      - enable_fungal (default False): Infestor Fungal Growth module
    """

    enable_rush: bool = True
    enable_fungal: bool = False

    async def on_step(self, iteration: int):
        if iteration == 0:
            for worker in self.workers:
                worker.gather(self.mineral_field.closest_to(worker))

        if not self.townhalls:
            return
        cc = self.townhalls.first

        # --- Shared economy ---
        await self._run_economy(cc)

        # --- Behavior profiles ---
        if self.enable_rush:
            await self._run_rush(cc)

        if self.enable_fungal:
            await self._run_fungal(cc)

    async def _run_economy(self, cc) -> None:
        """Profile-aware economy.

        12_pool_zerg_rush: drones to 14, Overlords only when supply-blocked, no gas.
        fungal_test: original freeform economy (Drones, Overlords, Extractors).
        """
        larvae = self.larva
        if not larvae:
            return

        if self.enable_rush:
            # Drones only until the cap, then larvae are reserved for lings
            if self.workers.amount < RUSH_DRONE_CAP and self.supply_left > 0:
                if self.can_afford(UnitTypeId.DRONE):
                    larvae.random.train(UnitTypeId.DRONE)
            # Rush notes assume Overlords only when supply-blocked
            if self.supply_left <= 0 and not self.already_pending(UnitTypeId.OVERLORD):
                if self.can_afford(UnitTypeId.OVERLORD):
                    larvae.random.train(UnitTypeId.OVERLORD)
        else:
            if self.can_afford(UnitTypeId.DRONE) and self.supply_left > 0:
                larvae.random.train(UnitTypeId.DRONE)

            if self.supply_left < 4 and not self.already_pending(UnitTypeId.OVERLORD):
                if self.can_afford(UnitTypeId.OVERLORD):
                    larvae.random.train(UnitTypeId.OVERLORD)

            if self.structures.of_type(UnitTypeId.EXTRACTOR).amount < 2:
                if self.can_afford(UnitTypeId.EXTRACTOR):
                    vgs = self.vespene_geyser.closer_than(15, cc)
                    if vgs:
                        for vg in vgs:
                            if not self.structures.of_type(UnitTypeId.EXTRACTOR).closer_than(1, vg):
                                await self.build(UnitTypeId.EXTRACTOR, near=vg)
                                break

        for drone in self.workers.idle:
            if not self.enable_rush and self.structures.of_type(UnitTypeId.EXTRACTOR).ready:
                ref = self.structures.of_type(UnitTypeId.EXTRACTOR).ready.first
                if ref.surplus_harvesters < 0:
                    drone.gather(ref)
                    continue
            drone.gather(self.mineral_field.closest_to(drone))

    async def _run_rush(self, cc) -> None:
        """12_pool_zerg_rush profile: Pool at 12, lings from 14, natural at 16,
        Queen at 18, then all-in attack once the zergling wave is complete."""
        # Spawning Pool at 12 supply
        if (
            not self.structures(UnitTypeId.SPAWNINGPOOL)
            and not self.already_pending(UnitTypeId.SPAWNINGPOOL)
            and self.can_afford(UnitTypeId.SPAWNINGPOOL)
            and self.workers.amount >= 11
        ):
            await self.build(
                UnitTypeId.SPAWNINGPOOL,
                near=cc.position.towards(self.game_info.map_center, 8),
            )

        # Hatchery at the natural (16 supply)
        if (
            self.supply_workers >= RUSH_HATCHERY_SUPPLY
            and len(self.townhalls) < 2
            and not self.already_pending(UnitTypeId.HATCHERY)
            and self.can_afford(UnitTypeId.HATCHERY)
        ):
            await self.expand_now()

        # Queen from the natural once it completes (18 supply)
        if (
            len(self.townhalls.ready) >= 2
            and             self.units(UnitTypeId.QUEEN).amount < 1
            and cc.is_idle
            and not self.already_pending(UnitTypeId.QUEEN)
            and self.can_afford(UnitTypeId.QUEEN)
        ):
            cc.train(UnitTypeId.QUEEN)

        # Zerglings: all remaining larvae go to lings while the pool is up
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            if self.can_afford(UnitTypeId.ZERGLING) and self.supply_left >= 2:
                larvae = self.larva
                if larvae:
                    larvae.random.train(UnitTypeId.ZERGLING)

        # Attack trigger: once the wave is complete, send everything
        zerglings = self.units(UnitTypeId.ZERGLING)
        if zerglings.amount >= RUSH_ATTACK_ZERGLINGS:
            target = self._rush_target()
            for ling in zerglings.idle:
                ling.attack(target)

    def _rush_target(self):
        """Prefer known enemy structures, else the enemy start location."""
        if self.enemy_structures:
            return self.enemy_structures.closest_to(self.start_location)
        if self.enemy_start_locations:
            return self.enemy_start_locations[0]
        return self.game_info.map_center

    async def _run_fungal(self, cc) -> None:
        """Fungal Growth module: build Infestors and cast on enemy clumps."""
        # Build Spawning Pool (prerequisite for Lair)
        if not self.structures.of_type(UnitTypeId.SPAWNINGPOOL) and self.can_afford(UnitTypeId.SPAWNINGPOOL):
            await self.build(UnitTypeId.SPAWNINGPOOL, near=cc.position.towards(self.game_info.map_center, 8))

        # Upgrade to Lair (prerequisite for Infestation Pit)
        if (
            self.structures.of_type(UnitTypeId.SPAWNINGPOOL).ready
            and not self.structures.of_type(UnitTypeId.LAIR)
            and not self.already_pending(UnitTypeId.LAIR)
            and self.can_afford(UnitTypeId.LAIR)
        ):
            cc.build(UnitTypeId.LAIR)

        # Build Infestation Pit (prerequisite for Infestor)
        if (
            self.structures.of_type(UnitTypeId.LAIR).ready
            and not self.structures.of_type(UnitTypeId.INFESTATIONPIT)
            and self.can_afford(UnitTypeId.INFESTATIONPIT)
        ):
            await self.build(UnitTypeId.INFESTATIONPIT, near=cc.position.towards(self.game_info.map_center, 10))

        # Produce Infestors from Larva
        if self.structures.of_type(UnitTypeId.INFESTATIONPIT).ready:
            if self.can_afford(UnitTypeId.INFESTOR) and self.supply_left > 0:
                larvae = self.larva
                if larvae:
                    larvae.random.train(UnitTypeId.INFESTOR)

        # Infestor micro: move toward enemy and cast Fungal
        infestors = self.units(UnitTypeId.INFESTOR)
        if not infestors:
            return

        enemy_units = self.enemy_units
        if not enemy_units:
            if self.enemy_start_locations:
                target = self.enemy_start_locations[0]
                for inf in infestors:
                    inf.move(target)
            return

        for inf in infestors:
            if inf.energy >= 75:
                # Find best fungal target: largest clump of enemies
                best_target = None
                best_count = 0
                for enemy in enemy_units:
                    count = len(enemy_units.closer_than(2.0, enemy))
                    if count > best_count:
                        best_count = count
                        best_target = enemy.position

                if best_target is not None and best_count >= 2:
                    inf(AbilityId.FUNGALGROWTH_FUNGALGROWTH, best_target)
                else:
                    closest_enemy = enemy_units.closest_to(inf)
                    inf.move(closest_enemy.position.towards(inf.position, -5))
            else:
                # Not enough energy — move toward enemy
                closest_enemy = enemy_units.closest_to(inf)
                dist = inf.distance_to(closest_enemy)
                if dist > 10:
                    inf.move(closest_enemy.position.towards(inf.position, -5))