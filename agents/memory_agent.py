from tensorforce.agents import Agent
from tensorforce.core.memories import Memory
from tensorforce.util import relatively_safe_pickle_dump, pickle_load
import os


class MemoryAgent(Agent):
    default_config = dict(
        batch_size=1,
        memory_capacity=1000000,
        memory=dict(
            type='replay',
            random_sampling=True
        ),
        update_frequency=4,
        first_update=10000,
        repeat_update=1,
        memory_dump_suffix=".memory_pkl_zip"
    )

    def __init__(self, config, model=None):
        config.default(MemoryAgent.default_config)
        super(MemoryAgent, self).__init__(config, model)

        self.batch_size = config.batch_size
        self.memory = Memory.from_config(
            config=config.memory,
            kwargs=dict(
                capacity=config.memory_capacity,
                states_config=config.states,
                actions_config=config.actions
            )
        )
        self.update_frequency = config.update_frequency
        self.first_update = config.first_update
        self.repeat_update = config.repeat_update
        self.memory_dump_suffix = config.memory_dump_suffix

    def observe(self, reward, terminal):
        reward, terminal = super(MemoryAgent, self).observe(reward, terminal)
        self.current_reward = reward
        self.current_terminal = terminal

        self.memory.add_observation(
            state=self.current_state,
            action=self.current_action,
            reward=self.current_reward,
            terminal=self.current_terminal,
            internal=self.current_internal
        )

        if self.local_step >= self.first_update and self.local_step % self.update_frequency == 0:
            for _ in xrange(self.repeat_update):
                batch = self.memory.get_batch(batch_size=self.batch_size, next_states=True)
                _, loss_per_instance = self.model.update(batch=batch)
                self.memory.update_batch(loss_per_instance=loss_per_instance)

    def import_observations(self, observations):
        for observation in observations:
            self.memory.add_observation(
                state=observation['state'],
                action=observation['action'],
                reward=observation['reward'],
                terminal=observation['terminal'],
                internal=observation['internal']
            )

    def memory_dump_path(self, path):
        return "{}{}".format(path, self.memory_dump_suffix)

    def load_model(self, path, load_memory=True):
        self.model.load_model(path)
        if load_memory:
            memory_dump_path = self.memory_dump_path(path)
            if not os.path.exists(memory_dump_path):
                self.logger.error("load memory dump file error, file do not exist:%s", memory_dump_path)
            else:
                self.memory = pickle_load(memory_dump_path, compression=True)

    def save_model(self, path, timestep=None, save_memory=True):
        model_path = self.model.save_model(path, timestep=timestep)
        if save_memory:
            memory_dump_path = self.memory_dump_path(model_path)
            relatively_safe_pickle_dump(self.memory, memory_dump_path, compression=True)