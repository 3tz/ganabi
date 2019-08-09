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
from experts.rainbow_models.run_experiment import format_legal_moves

class Agent(_Agent):
  """
  path_to_my_model - file that saved the model
  """
  def __init__(self, path_to_my_model):
    """Initialize the agent."""
    pre_trained = keras.models.load_model('path_to_my_model.h5')

  def _parse_observation(self, current_player_observation):
    observation_vector = np.array(current_player_observation['vectorized']) #FIXME: this may need to be cast as np.float64
    return observation_vector

  def act(self, observation):
    if observation['current_player_offset'] != 0:
      return None

    observation_vector = self._parse_observation(observation)
    #FIX ME
    action = self.pre_trained.predict(observation_vector)

    return action
