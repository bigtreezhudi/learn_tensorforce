from random import random, randrange
import numpy as np

from tensorforce import util, TensorForceError
from tensorforce.core.memories import Memory


class PrioritizedReplay(Memory):
    def __init__(self, capacity, states_config, actions_config, prioritization_weight=1.0):
        super(PrioritizedReplay).__init(capacity, states_config, actions_config)
        self.prioritization_weight = prioritization_weight
        self.obervations = list()
        self.last_obervation = None
        self.internals_config = None
        self.batch_indices = None
        self.none_priority_index = 0

    def add_observation(self, state, action, reward, terminal, internal):
        if self.internals_config is None and internal is not None:
            self.internals_config = [(i.shape, i.dtype) for i in internal]
        if self.last_obervation is not None:
            observation = self.last_obervation + (state, internal)
            if len(self.obervations) < self.capacity:
                self.obervations.append((None, observation))
            elif self.none_priority_index > 0:
                priority, _ = self.obervations.pop(self.none_priority_index - 1)
                self.obervations.append((None, observation))
                self.none_priority_index -= 1
            else:
                raise TensorForceError("Memory contain only unseen observations")
        self.last_obervation = (state, action, reward, terminal, internal)

    def update_batch(self, loss_per_instance):
        if self.batch_indices is None:
            raise TensorForceError("Must call get_batch method before update_batch")
        if len(loss_per_instance) != len(self.batch_indices):
            raise TensorForceError("For all instance a loss is has to be provided")

        updated = list()
        for index, loss in zip(self.batch_indices, loss_per_instance):
            priority, observation = self.observations[index]
            updated.append((loss ** self.prioritization_weight, observation))
        for index in sorted(self.batch_indices, reverse=True):
            priority, _ = self.observations.pop(index)
            self.none_priority_index -= (priority is not None)
        self.batch_indices = None
        updated = sorted(updated, key=(lambda x: x[0]))

        update_priority, update_observation = updated.pop()
        index = -1
        for priority, _ in iter(self.observations):
            index += 1
            if index == self.none_priority_index:
                break
            if update_priority < priority:
                continue
            self.observations.insert(index, (update_priority, update_observation))
            index += 1
            self.none_priority_index += 1
            if not updated:
                break
            update_priority, update_observation = updated.pop()
        else:
            self.observations.insert(index, (update_priority, update_observation))
            self.none_priority_index += 1
        while updated:
            self.observations.insert(index, updated.pop())
            self.none_priority_index += 1