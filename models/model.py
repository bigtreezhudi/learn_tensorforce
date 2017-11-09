import logging
import tensorflow as tf

from tensorforce import TensorForceError, util
from tensorforce.core.optimizers import Optimizer
from tensorforce.core.explorations import Exploration

class Model(object):
    allows_discrete_actions = None
    allows_continuous_actions = None

    default_config = dict(
        discount=0.97,
        learning_rate=0.0001,
        optimizer='adam',
        device=None,
        tf_summary=None,
        tf_summary_level=0,
        tf_summary_interval=1000,
        distributed=False,
        global_model=False,
        session=None,
        checkpoint_max_to_keep=100,
        keep_checkpoint_every_n_hours=1,
        checkpoint_pad_step_number=True,
        max_grad_norm=None,
        learning_rate_decay=None
    )

    def __init__(self, config):
        assert self.__class__.allows_discrete_actions is not None and self.__class__.allows_continuous_actions is not None
        config.default(Model.default_config)

        self.discount = config.discount
        self.distributed = config.distributed
        self.session = None
        self.tf_summary = config.tf_summary

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(util.log_levels[config.log_level])

        self.graph = tf.Graph()

        graph_context = self.graph.as_default()
        graph_context.__enter__()

        if not config.distributed:
            assert not config.global_model and config.session is None
        if config.distributed and not config.global_model:
            global_config = config.copy()
            global_config.optimizer = None
            global_config.global_model = True
            global_config.device = tf.train.replica_device_setter(1, worker_device=config.device, cluster=config.cluster_spec)
            self.global_model = self.__class__(config=global_config)
            self.global_timestep = self.global_model.global_timestep
            self.global_episode = self.global_model.episode
            self.global_variables = self.global_model.variables

        self.optimizer_args = None
        with tf.device(config.device):
            if config.distributed:
                if config.global_model:
                    self.global_timestep = tf.get_variable(name='timestep', dtype=tf.int64, initializer=0, trainable=False)
                    self.episode = tf.get_variable(name='episode', dtype=tf.int64, initializer=0, trainable=False)
                    scope_context = tf.variable_scope('global')
                else:
                    scope_context = tf.variable_scope('local')
                scope = scope_context.__enter__()
            self.create_tf_operations(config)

            if config.distributed:
                self.variables = tf.contrib.framework.get_variables(scope=scope)

            assert self.optimizer or (not config.distributed or config.global_model)
            if self.optimizer:
                if config.distributed and not config.global_model:
                    self.loss = tf.add_n(inputs=tf.losses.get_losses(scope=scope.name))
                    local_grads_and_vars = self.optimizer.compute_gradients(loss=self.loss, var_list=self.variables)
                    local_gradients = [grad for grad, var in local_grads_and_vars]
                    global_gradients = list(zip(local_gradients, self.global_model.variables))
                    self.update_local = tf.group(*(v1.assign(v2) for v1, v2 in zip(self.variables, self.global_model.variables)))
                    self.optimize = tf.group(
                        self.optimizer.apply_gradients(grads_and_vars=global_gradients),
                        self.update_local,
                        self.global_timestep.assign_add(tf.shape(self.reward)[0]))
                    self.increment_global_episode = self.global_episode.assign_add(tf.count_nonzero(input_tensor=self.terminal, dtype=tf.int32))
                else:
                    self.loss = tf.losses.get_total_loss()
                    if self.optimizer_args is not None:
                        self.optimizer_args['loss'] = self.loss
                        grads_and_vars = self.optimizer.compute_gradients(**self.optimizer_args)
                    else:
                        grads_and_vars = self.optimizer.compute_gradients(self.loss)

                    if config.max_grad_norm is not None:
                        grads, params = zip(*grads_and_vars)
                        grads, grad_norm = tf.clip_by_global_norm(grads, config.max_grad_norm)
                        grads_and_vars = list(zip(grads, params))

                    self.optimize = self.optimizer.apply_gradients(grads_and_vars)

            if config.distributed:
                scope_context.__exit__(None, None, None)
        if config.learning_rate_decay:
            self.learning_rate_decay = Exploration.from_config(config=config.learning_rate_decay)
            self.init_learning_rate = config.learning_rate
        else:
            self.learning_rate_decay = None

        self.saver = tf.train.Saver(
                max_to_keep=config.checkpoint_max_to_keep,
                keep_checkpoint_every_n_hours=config.keep_checkpoint_every_n_hours,
                pad_step_number=config.checkpoint_pad_step_number)

        if config.tf_summary is not None:
            tf.summary.scalar('total-loss', self.loss)
            self.writer = tf.summary.FileWriter(config.tf_summary, graph=tf.get_default_graph())
            self.last_summary_step = -float('inf')
            if config.tf_summary_level >= 2:
                for v in tf.trainable_variables():
                    tf.summary.histogram(v.name, v)
            self.tf_summaries = tf.summary.merge_all()
            self.tf_episode_reward = tf.placeholder(tf.float32, name='episode-reward-placeholder')
            self.episode_reward_summary = tf.summary.scalar('episode-reward', self.tf_episode_reward)
        else:
            self.writer = None
            config.tf_summary_level
            config.tf_summary_interval
        self.timestep = 0
        self.summary_interval = config.tf_summary_interval

        if not config.distributed:
            gpu_options = tf.GPUOptions(allow_growth=True)
            tf_config = tf.ConfigProto(gpu_options=gpu_options)
            self.set_session(tf.Session(config=tf_config))
            self.session.run(tf.global_variables_initializer())
        graph_context.__exit__(None, None, None)
