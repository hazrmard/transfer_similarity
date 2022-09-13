from copy import deepcopy
from typing import Callable
import types

import numpy as np
import gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)


__all__ = ['XformedPolicy', 'evaluate_policy', 'learn_rl', 'transform_rl_policy']



class XformedPolicy(ActorCriticPolicy):
    
    def __init__(
        self, *args, state_xform=None, action_xform=None, learnable=False,
        unconstrained_actions=None, **kwargs):
        # Assign the variables. They will be conditionally converted to Parameters
        # in _build() so that the policy optimizer can pick them up.
        self.state_xform = state_xform
        self.action_xform = action_xform
        self.learnable_xform = learnable
        # if ==1, action space is the same
        self.unconstrained_actions = unconstrained_actions
        super().__init__(*args, **kwargs)
    
    def _build(self, *args):
        if isinstance(self.action_space, gym.spaces.Box) and \
                self.unconstrained_actions is not None:
            self.action_space = deepcopy(self.action_space)
            self.action_space.low *= self.unconstrained_actions
            self.action_space.high *= self.unconstrained_actions
        dtype=torch.float
        if self.state_xform is None:
            self.state_xform = torch.zeros(self.action_space.shape[0],
                                           self.observation_space.shape[0],
                                           dtype=dtype)
        else:
            self.state_xform =torch.tensor(self.state_xform, dtype=dtype)
        if self.action_xform is None:
            self.action_xform = torch.eye(self.action_space.shape[0],
                                          dtype=dtype)
        else:
            self.action_xform = torch.tensor(self.action_xform, dtype=dtype)
        if self.learnable_xform:
            setattr(self, 'state_xform', nn.Parameter(self.state_xform, requires_grad=True))
            setattr(self, 'action_xform', nn.Parameter(self.action_xform, requires_grad=True))
        return super()._build(*args)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # the following line was previously in `_get_action_dist_from_latent`,
        # however it has been moved here so the actions can be transformed using
        # the observation vector
        mean_action = self.action_net(latent_pi)
        mean_action = ((self.state_xform @ obs.transpose(1, 0)) + \
                       (self.action_xform @ mean_action.transpose(1, 0))).transpose(1, 0)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        # Method changed to no longer take latent vectors:
        distribution = self._get_action_dist_from_mean(mean_action, latent_pi=latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_mean(self, mean_actions: torch.Tensor, latent_pi: torch.Tensor=None):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        # mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
            """
            Evaluate actions according to the current policy,
            given the observations.
            :param obs:
            :param actions:
            :return: estimated value, log likelihood of taking those actions
                and entropy of the action distribution.
            """
            # Preprocess the observation if needed
            features = self.extract_features(obs)
            latent_pi, latent_vf = self.mlp_extractor(features)
            # the following line was previously in `_get_action_dist_from_latent`,
            # however it has been moved here so the actions can be transformed using
            # the observation vector
            mean_action = self.action_net(latent_pi)
            mean_action = ((self.state_xform @ obs.transpose(1, 0)) + \
                           (self.action_xform @ mean_action.transpose(1, 0))).transpose(1, 0)
            distribution = self._get_action_dist_from_mean(mean_action, latent_pi=latent_pi)
            log_prob = distribution.log_prob(actions)
            values = self.value_net(latent_vf)
            return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: torch.Tensor):
        """
        Get the current policy distribution given the observations.
        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        # the following line was previously in `_get_action_dist_from_latent`,
        # however it has been moved here so the actions can be transformed using
        # the observation vector
        mean_action = self.action_net(latent_pi)
        mean_action = ((self.state_xform @ obs.transpose(1, 0)) + \
                       (self.action_xform @ mean_action.transpose(1, 0))).transpose(1, 0)
        return self._get_action_dist_from_mean(mean_action, latent_pi=latent_pi)



# From: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-hyperparameters
class HParamCallback(BaseCallback):
    def __init__(self):
        """
        Saves the hyperparameters and metrics at the start of the training,
        and logs them to TensorBoard.
        """
        super().__init__()

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs,
            "learnable_xform": getattr(self.model.policy,
                                       'learnable_xform', False)
        }
        # define the metrics that will appear in the `HPARAMS`
        # Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "train/explained_variance": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )
        
    def _on_step(self) -> bool:
        return True



def learn_rl(
    env, steps=50_000, tensorboard_log=None, learning_rate=1e-3, verbose=0,
    seed=None, reuse_parameters_of=None, learnable_transformation=False,
    constrained_actions=None, **kwargs
):
    # process arguments
    kwargs['n_steps'] = kwargs.get('n_steps', getattr(env, 'period', 2048))
    kwargs['policy_kwargs'] = kwargs.get('policy_kwargs', {})
    kwargs['policy_kwargs']['learnable'] = learnable_transformation
    kwargs['policy_kwargs']['learnable'] = constrained_actions
    # if transformation is learnable, then the attributes are Parameters,
    # otherwise they are tensors
    
    model = PPO(XformedPolicy, env, verbose=verbose,
                tensorboard_log=None if tensorboard_log is None else \
                './tensorboard/' + env.__class__.__name__,
                learning_rate=learning_rate,
                seed=seed, **kwargs)
    if reuse_parameters_of is not None:
        params = reuse_parameters_of.policy.state_dict()
        # if reused policy already has been transformed with
        # learnable transform parameters, then
        # transform this policy too
        state_xform = params.get(
            'state_xform',
            getattr(reuse_parameters_of.policy, 'state_xform', None))
        action_xform = params.get(
            'action_xform',
            getattr(reuse_parameters_of.policy, 'action_xform', None))
        if state_xform is not None and action_xform is not None:
            model = transform_rl_policy(
                model, state_xform, action_xform,
                learnable=learnable_transformation
            )
            # if not learnable, ensure xforms are not in params dict
            # since the static tensors have been assigned as attributes
            # already by transform_rl_policy
            if learnable_transformation:
                params['state_xform'] = state_xform
                params['action_xform'] = action_xform
            else:
                if 'state_xform' in params:
                    del params['state_xform']
                if 'action_xform' in params:
                    del params['action_xform']
        # now params dict contains *xform iff learnable_transformation=True,
        # The transform function has already created & assigned values to the
        # *xform tensors in model.policy
        model.policy.load_state_dict(params)
    model.learn(total_timesteps=steps, tb_log_name=tensorboard_log,
                callback=HParamCallback())
    return model



def transform_rl_policy(
    agent, state_xform, action_xform, learnable=False, copy=True
):
    if copy:
        # deepcopy cannot copy _thread_lock objects which
        # the logger uses
        logger = agent.__dict__.get('_logger')
        if logger is not None:
            del agent.__dict__['_logger']
        agent = deepcopy(agent)
        if logger is not None:
            agent.__dict__[ '_logger'] = logger
    dtype = next(agent.policy.parameters()).dtype
    for xform, name in zip((state_xform, action_xform),
                           ('state_xform', 'action_xform')):
        # ensure that provided xform is a tensor
        if isinstance(xform, np.ndarray):
            t = torch.from_numpy(xform).to(dtype)
        elif isinstance(xform, nn.Parameter):
            t = xform.data.clone().detach()
        elif isinstance(xform, torch.Tensor):
            t = xform.to(dtype)

        # check if xform is already a tensor/parameter
        # attribute on the policy
        existing = getattr(agent.policy, name, None)
        if learnable:
            # if parameter, copy in-place
            if isinstance(existing, nn.Parameter):
                with torch.no_grad():
                    existing.data.copy_(t)
                    existing.data.requires_grad_(True)
                    existing.requires_grad = True
            # if a tensor, or None, create a new parameter
            else:
                p = nn.Parameter(t, requires_grad=True)
                setattr(agent.policy, name, p)
        # if not learnable, assign as a tensor
        else:
            setattr(agent.policy, name, t)
        
    # Create a new optimizer. An optimizer in the middle of
    # learning may have a state. So just create a new
    # optimizer for all params (xforms and policy/value net)
    # so they have the same state wherever applicable.
    optimizer = agent.policy.optimizer_class(
        agent.policy.parameters(),
        lr=agent.learning_rate,
        **agent.policy.optimizer_kwargs)
    agent.policy.old_optimizer = agent.policy.optimizer
    agent.policy.optimizer = optimizer

    return agent



def replace_instance_fn(instance, fn_name: str, replace_with: Callable):
    new_fn = types.MethodType(replace_with, instance)
    setattr(instance, fn_name, new_fn)
    return instance