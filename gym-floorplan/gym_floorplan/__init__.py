import gymnasium as gym

def _register(id, entry_point, force=True):
    # env_specs = gym.envs.registry.env_specs
    if id in gym.envs.registry: #env_specs.keys(): # gym.envs.registry:
        if not force:
            return
        del gym.envs.registry[id] #env_specs[id] # gym.envs.registry[id]
    gym.register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=1000,
    )

# Register modified versions of existing environments
_register(
    id="DOLW-v0",
    entry_point="gym_floorplan.envs.master_env:MasterEnv",
)

