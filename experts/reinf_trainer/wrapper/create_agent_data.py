import os, sys

PATH_FILE = os.path.realpath(__file__)
PATH_WRAPPER = os.path.dirname(PATH_FILE)
PATH_REINF = os.path.dirname(PATH_WRAPPER)
PATH_EXPERTS = os.path.dirname(PATH_REINF)
PATH_GANABI = os.path.dirname(PATH_EXPERTS)
sys.path.append(PATH_GANABI)
sys.path.append(PATH_REINF)

# from utils import dir_utils, parse_args
from collections import defaultdict
import agent_wrapper as rainbow
import pickle
from hanabi_env import rl_env
import gin
import tensorflow as tf # version 1.x #FIX ME are we using tensorflow 2.
import importlib
import argparse

def import_agents(config, agent_path):
    return rainbow.Agent(config, agent_path)

def one_hot_vectorized_action(agent, num_moves, obs):
    action = agent.act(obs)
    one_hot_action_vector = [0]*num_moves
    action_idx = obs['legal_moves_as_int'][obs['legal_moves'].index(action)]
    one_hot_action_vector[action_idx] = 1

    return one_hot_action_vector, action

class DataCreator(object):
    def __init__(self, num_games, path_model_0, path_model_1 ):
        config = {
          'observation_size': 658,
          'num_moves': 20,
          'colors': 5,
          'ranks': 5,
          'players': 2,
          'hand_size': 5,
          'max_information_tokens': 8,
          'max_life_tokens': 3,
          'seed': 1,
          'observation_type': 1,  # FIXME: NEEDS CONFIRMATION
          'random_start_player': False}
        self.num_players = 2
        self.num_games = num_games
        self.environment = rl_env.HanabiEnv(config)
        self.agent_object = [
            import_agents(config, path_model_0),
            import_agents(config, path_model_1)
        ]

    def create_data(self):
        raw_data = []
        scores = []
        for game_num in range(self.num_games):
            raw_data.append([[],[]])
            observations = self.environment.reset()
            game_done = False

            while not game_done:
                for agent_id in range(self.num_players):
                    observation = observations['player_observations'][agent_id]
                    one_hot_action_vector, action = one_hot_vectorized_action(
                            self.agent_object[agent_id],
                            self.environment.num_moves(),
                            observation)
                    if observation['current_player'] == agent_id:
                        assert action is not None
                        current_player_action = action
                    else:
                        assert action is None

                    observations, _, game_done, _ = self.environment.step(
                            current_player_action)
                    if game_done:
                        scores.append(self.environment.state.score())
                        break
        return scores
