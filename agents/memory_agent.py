from tensorforce.agents import Agent
from tensorforce.core.memories import Memory
from tensorforce.util import relatively_safe_pickle_dump, pickle_load


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
