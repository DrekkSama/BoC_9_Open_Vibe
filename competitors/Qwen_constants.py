"""Constants for the Zerg micro bot."""

from sc2.ids.unit_typeid import UnitTypeId
from ares.consts import UnitRole

# Unit types we're focusing on for micro
ZERGLING = UnitTypeId.ZERGLING
ROACH = UnitTypeId.ROACH
BANELING = UnitTypeId.BANELING
QUEEN = UnitTypeId.QUEEN

# Unit roles for our micro bot
ATTACKING = UnitRole.ATTACKING
BASE_DEFENDER = UnitRole.BASE_DEFENDER

# Constants for micro behaviors
BANELING_SPLASH_RADIUS = 2.2
QUEEN_HEAL_THRESHOLD = 0.4
QUEEN_HEAL_RANGE = 7.0