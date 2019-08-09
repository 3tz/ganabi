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
    action = self.pre_trained.predict(observation_vector)

    return action
