from random import randrange
import numpy as np

from tensorforce import Memory
from tensorforce import util


class Replay(Memory):
    def __init__(self, capacity, states_config, actions_config, random_sampling=True, next_in_place=False, state_value=False):
        super(Replay).__init__(capacity, states_config, actions_config)
        self.random_sampling = random_sampling
        self.next_in_place = next_in_place
        self.states = {name: np.zeros((self.capacity,) + tuple(state.shape), dtype=util.np_dtype(state.type)) for name, state in states_config.items()}
        self.actions = {name: np.zeros((self.capacity,) + tuple(action.shape), dtype=util.np_dtype('float' if action.continuous else 'int'))
                        for name, action in actions_config.items()}
        self.rewards = np.zeros((self.capacity,), dtype=util.np_dtype('float'))
        self.terminals = np.zeros((self.capacity,), dtype=util.np_dtype('bool'))
        self.internals = None
        self.size = 0
        self.index = 0
        if self.next_in_place:
            self.next_states = {name: np.zeros((self.capacity,) + tuple(state.shape), dtype=util.np_dtype(states_config.type)) for name, state in states_config}
            self.next_internals = None
        self.state_values = np.zeros((self.capacity,), dtype=util.np_dtype('float')) if state_value else None

    def add_observation(self, state, action, reward, terminal, internal, next_state=None, next_internal=None, state_value=None):
        if self.internals is None and internal is not None:
            self.internals = [np.zeros((self.capacity,) + tuple(i.shape), dtype=util.np_dtype(i.type)) for i in internal]
        for name, state in state.items():
            self.states[name][self.index] = state
        for name, action in action.items():
            self.actions[name][self.index] = action
        self.rewards[self.index] = reward
        self.terminals[self.index] = terminal
        for i, internal in enumerate(internal):
            self.internal[self.index][i] = internal
        if self.next_in_place:
            if self.next_internals is None and next_internal is not None:
                self.next_internals = [np.zeros((self.capacity,) + tuple(i.shape), dtype=util.np_dtype(i.type))
                                       for i in next_internal]
            for name, state in next_state.items():
                self.next_states[name][self.index] = state
            for i, internal in enumerate(next_internal):
                self.next_internals[i][self.index] = internal
        if state_value is not None:
            self.state_values[self.index] = state_value
        self.index = (self.index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def get_batch(self, batch_size, next_states=False):
        if self.random_sampling:
            rand_end = (self.size - 1) if next_states else self.size
            indices = np.random.randint(rand_end, size=batch_size)
            if self.index < self.size:
                indices = (indices + self.index) % self.capacity
            states = {name: states.take(indices, axis=0) for name, states in self.states.items()}
            actions = {name: actions.take(indices, axis=0) for name, actions in self.actions.items()}
            rewards = self.rewards.take(indices)
            terminals = self.terminals.take(indices)
            internals = [internal.take(indices) for internal in self.internals]
            if self.state_values is not None:
                state_values = self.state_values.take(indices)
            if next_states:
                if self.next_in_place:
                    next_states = {name: state.take(indices, axis=0) for name, state in self.next_states.items()}
                    next_internals = [internal.take(indices) for internal in self.next_internals]
                else:
                    indices = (indices + 1) % self.capacity
                    next_states = {name: state.take(indices, axis=0) for name, state in self.states.items()}
                    next_internals = [internal.take(indices) for internal in self.internals]
        else:
            rand_start = 1 if next_states else 0
            end = (self.index - randrange(rand_start, self.size - batch_size + 1)) % self.capacity
            start = (end - batch_size) % self.capacity
            if start < end:
                states = {name: state[start:end] for name, state in self.states.items()}
                acitons = {name: action[start:end] for name, action in self.actions.items()}
                rewards = self.rewards[start:end]
                terminals = self.terminals[start:end]
                internals = [internal[start:end] for internal in self.internals]
                if next_states:
                    if self.next_in_place:
                        next_states = {name: state[start:end] for name, state in self.next_states.items}
                        next_internals = [internal[start:end] for internal in self.next_internals]
                    else:
                        next_states = {name: state[start + 1:end + 1] for name, state in self.states.items}
                        next_internals = [internal[start + 1:end + 1] for internal in self.internals]
                if self.state_values is not None:
                    state_values = self.state_values[start:end]
            else:
                states = {name: np.concatenate(state[start:], state[:end]) for name, state in self.states.items()}
                acitons = {name: np.concatenate(action[start:], action[:end]) for name, action in self.actions.items()}
                rewards = np.concatenate(self.rewards[start:], self.rewards[:end])
                terminals = np.concatenate(self.terminals[start:], self.terminals[:end])
                internals = [np.concatenate(internal[start:], internal[:end]) for internal in self.internals]
                if next_states:
                    if self.next_in_place:
                        next_states = {name: np.concatenate(state[start:], state[:end]) for name, state in self.next_states.items}
                        next_internals = [np.concatenate(internal[start:], internals[:end]) for internal in self.next_internals]
                    else:
                        next_states = {name: np.concatenate(state[start + 1:], state[:end + 1]) for name, state in self.states.items}
                        next_internals = [np.concatenate(internal[start + 1:], internals[:end + 1]) for internal in self.internals]
                if self.state_values is not None:
                    state_values = np.concatenate(self.state_values[start:], self.state_values[:end])
        batch = dict(states=states, actions=actions, rewards=rewards, terminals=terminals, internals=internals)
        if next_states:
            batch['next_states'] = next_states
            batch['next_internals'] = next_internals
        if self.state_values is not None:
            batch['state_values'] = state_values
        return batch

    def update_batch(self, loss_per_instance):
        pass

    def set_memory(self, states, actions, rewards, terminals, internals, state_values=None):
        self.size = len(states)
        if self.next_in_place:
            raise NotImplemented()
        if len(states) == self.capacity:
            for name, state in states.items():
                self.states[name] = np.asarray(state)
            for name, action in actions.items():
                self.actions[name] = np.asarray(action)
            self.rewards = np.asarray(rewards)
            self.terminals = np.asarray(terminals)
            self.internals = [np.asarray(internal) for internal in internals]
            if state_values is not None:
                self.state_values = np.asarray(state_values)
        else:
            for name, state in states.items():
                self.states[name][:len(state)] = state
            for name, action in actions.items():
                self.actions[name][:len(action)] = action
            self.rewards[:len(rewards)] = rewards
            self.terminals[:len(terminals)] = terminals
            for i, internal in enumerate(internals):
                self.internals[i][:len(internal)] = internal
            if state_values is not None:
                self.state_values[:len(state_values)] = state_values