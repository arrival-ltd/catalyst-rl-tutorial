from catalyst_rl.rl import registry

from src.env import CoppeliaSimEnvWrapper
from src.actor import UR5Actor
from src.critic import UR5StateActionCritic

registry.Environment(CoppeliaSimEnvWrapper)
registry.Agent(UR5Actor)
registry.Agent(UR5StateActionCritic)
