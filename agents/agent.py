import logging
from tensorforce import util, TensorForceError
from tensorforce.core.preprocessing import Preprocessing
from tensorforce.core.explorations import Exploration
from random import random


class Agent(object):
    name = None
    model = None
    default_config = dict(
        preprocessing=None,
        exploration=None,
        reward_preprocessing=None,
        log_level='info'
    )
    multi_threads = False

    def __init__(self, config, model=None):
        assert self.__class__.name is not None and self.__class__.model is not None
        config.default(Agent.default_config)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(util.log_levels[config.log_level])

        self.preprocessing = dict()
        if 'shape' in config.states:
            config.states = dict(state=config.states)
            self.unique_state = True
            if config.preprocessing is not None:
                config.preprocessing = dict(state=config.preprocessing)
        else:
            self.unique_state = False
        for name, state in config.states:
            state.default(dict(type='float'))
            if isinstance(state.shape, int):
                state.shape = (state.shape,)
            if config.preprocessing is not None and name in config.preprocessing:
                preprocessing = Preprocessing.from_config(config=config.preprocessing[name])
                self.preprocessing[name] = preprocessing
                state.shape = preprocessing.processed_shape(shape=state.shape)
                state.type = preprocessing.processed_type(dtype=state.type)

        self.exploration = dict()
        if 'continuous' in config.actions:
            config.actions = dict(action=config.actions)
            if config.exploration is not None:
                config.exploration = dict(action=config.exploration)
            self.unique_action = True
        else:
            self.unique_action = False
        for name, action in config.actions:
            if action.continuous:
                action.default(dict(shape=(), min_value=None, max_value=None))
            else:
                action.default(dict(shape=()))
            if isinstance(action.shape, int):
                action.shape = (action.shape,)
            if config.exploration is not None and name in config.exploration:
                self.exploration[name] = Exploration.from_config(config=config.exploration[name])

        self.reward_preprocessing = None
        if config.reward_preprocessing is not None:
            self.reward_preprocessing = Preprocessing.from_config(config=config.reward_preprocessing)

        self.states_config = config.states
        self.actions_config = config.actions

        if model is None:
            self.model = self.__class__.model(config)
        else:
            if not isinstance(model, self.__class__.model):
                raise TensorForceError("Supplied model class `{}` does not match expected agent model class `{}`".format(
                    type(model).__name__, self.__class__.model.__name__
                ))
            self.model = model

        not_accessed = config.not_accessed()
        if not_accessed:
            self.logger.warning("Configuration values not accessed: {}".format(', '.join(not_accessed)))

        self.episode = -1
        self.timestep = 0
        self.local_step = 0
        self.reset()

    def __str__(self):
        return str(self.__class__.name)

    def reset(self):
        self.episode += 1
        self.current_internal = self.next_internal = self.model.reset()
        for preprocessing in self.preprocessing.values():
            preprocessing.reset()
        self.extension = None

    def last_observation(self):
        return dict(
            state=self.current_state,
            action=self.current_action,
            reward=self.current_reward,
            terminal=self.current_terminal,
            internal=self.current_internal
        )

    def load_model(self, path):
        self.model.load_model(path)

    def save_model(self, path, timestep=None):
        self.model.save_model(path, timestep=timestep)

    def act(self, state, deterministic=False):
        if not self.multi_threads:
            self.timestep += 1
        self.local_step += 1
        self.current_internal = self.next_internal

        if self.unique_state:
            self.current_state = dict(state=state)
        else:
            self.current_state = state

        # Preprocessing
        for name, preprocessing in self.preprocessing.items():
            self.current_state[name] = preprocessing.process(state=self.current_state[name])

        # Podel action
        action_result = self.model.get_action(state=self.current_state, internal=self.current_internal, deterministic=deterministic)
        self.current_action, self.next_internal = action_result[:2]
        self.extension = action_result[2:]

        # Exploration
        if not deterministic:
            for name, exploration in self.exploration.items():
                if self.actions_config[name].continuous:
                    explore = (lambda: exploration(episode=self.episode, timestep=self.timestep))
                    shape = self.actions_config[name].shape
                    exploration = np.array([explore() for _ in range(util.prod(shape))])
                    self.current_action[name] += np.reshape(exploration, shape)
                else:
                    if random() < exploration(episode=self.episode, timestep=self.timestep):
                        shape = self.actions_config[name].shape
                        num_actions = self.actions_config[name].num_actions
                        self.current_action[name] = np.random.randint(low=num_actions, size=shape)

        if self.unique_action:
            return self.current_action['action']
        else:
            return self.current_action

    def observe(self, reward, terminal):
        if self.reward_preprocessing is not None:
            reward = self.reward_preprocessing.process(reward)
        return reward, terminal
