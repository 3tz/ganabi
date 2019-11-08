import os, sys

PATH_FILE = os.path.realpath(__file__)
PATH_WRAPPER = os.path.dirname(PATH_FILE)
PATH_REINF = os.path.dirname(PATH_WRAPPER)
PATH_EXPERTS = os.path.dirname(PATH_REINF)
PATH_GANABI = os.path.dirname(PATH_EXPERTS)
sys.path.append(PATH_GANABI)
sys.path.append(PATH_REINF)

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rainbow Agent."""

import numpy as np
from hanabi_env.rl_env import Agent as _Agent
from experts.reinf_trainer.wrapper.rainbow_agent import RainbowAgent as _RainbowAgent
from experts.reinf_trainer.run_experiment import format_legal_moves
from experts.reinf_trainer.third_party.dopamine import checkpointer

config = {
  'observation_size': 658,
  'num_moves': 20,
  'players': 2
}
path = '/tmp/reinf_checkpoints/outer-49/checkpoints'

def checkpoint_stuff(path):
  """
  Args:
    - path: str
        Path to the checkpointer directory
  """
  exp_checkpointer = checkpointer.Checkpointer(path, 'ckpt')
  checkpoint_version = checkpointer.get_latest_checkpoint_number(path)

  assert checkpoint_version >=0
  dqn_dictionary = exp_checkpointer.load_checkpoint(checkpoint_version)
  return path, checkpoint_version, dqn_dictionary

class Agent(_Agent):
  """Agent that loads and applies a pretrained rainbow model."""
  def __init__(self, config, path, *args, **kwargs):
    """Initialize the agent.
    Args:
      - config: dict
            Dictionary with the configuration of the game.
            EX:
            config = {
              'observation_size': 658,
              'num_moves': 20,
              'players': 2
            }
      - path: str
            Path to the checkpoint directory.

    """
    self.config = config
    self.agent = _RainbowAgent(
        observation_size=self.config['observation_size'],
        num_actions=self.config['num_moves'],
        num_players=self.config['players'])

    self.agent.eval_mode = True

    checkpoint_dir, checkpoint_version, dqn_dictionary = checkpoint_stuff(path)

    assert self.agent.unbundle(checkpoint_dir, checkpoint_version, dqn_dictionary),\
          'agent was unable to unbundle'
    assert 'logs' in dqn_dictionary # FIXME: necessary?
    assert 'current_iteration' in dqn_dictionary # FIXME: necessary?

  def _parse_observation(self, current_player_observation):
    legal_moves = current_player_observation['legal_moves_as_int']
    legal_moves = format_legal_moves(legal_moves, self.config['num_moves'])
    observation_vector = np.array(current_player_observation['vectorized']) #FIXME: this may need to be cast as np.float64

    return legal_moves, observation_vector

  def act(self, observation):
    """Act based on the observation of the current player."""
    #import pdb; pdb.set_trace()

    # Make sure that this player is the current player
    if observation['current_player_offset'] != 0:
      return None

    legal_moves, observation_vector = self._parse_observation(observation)
    action = self.agent._select_action(observation_vector, legal_moves)
    action = observation['legal_moves'][observation['legal_moves_as_int'].index(action)]

    return action
