# this script creates data using python 2 and rainbow agents

# here we add the repo's root directory to the path variable; everything
# is imported relative to that to avoid problems
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils import dir_utils, parse_args
from collections import defaultdict
from agent_model import agent_wrapper as model
import pickle
from hanabi_env import rl_env
import gin
import tensorflow as tf # version 1.x #FIX ME are we using tensorflow 2.
import importlib
import argparse

def import_agents(agent_path):
    return model.agent(agent_path)

def one_hot_vectorized_action(agent, num_moves, obs):
    '''
    Inputs:
        agent: agent object we imported and initialized with agent_config
        num_moves: length of the action vector
        obs: observation object (has lots of good info, run print(obs.keys()) to see)
    Returns:
        one_hot_action_vector: one hot action vector
        action: action in the form recognizable by the Hanabi environment
                (idk something like {'discard': 5})
    '''
    action = agent.act(obs)
    one_hot_action_vector = [0]*num_moves
    action_idx = obs['legal_moves_as_int'][obs['legal_moves'].index(action)]
    one_hot_action_vector[action_idx] = 1

    return one_hot_action_vector, action

class DataCreator(object):
    def __init__(self, args):
        self.agent_object = import_agents(agent_path)


    def create_data(self):
        '''Create and return a list of games. Each game has the following structure:
            [ [[obs_0], [obs_1], ..., [obs_n]], [[act_0], [act_1], ..., [act_n]] ]
        where each obs_i and act_i are the observation and resultant action that
        an agent took at game step i. Each game round consists of num_players game
        steps. A game can have a variable amount of rounds--you can lose early.
        '''
        raw_data = []

        for game_num in range(self.num_games):
            raw_data.append([[],[]])
            observations = self.environment.reset()
            game_done = False

            while not game_done:
                for agent_id in range(self.num_players):
                    observation = observations['player_observations'][agent_id]
                    one_hot_action_vector, action = one_hot_vectorized_action(
                            self.agent_object,
                            self.environment.num_moves(),
                            observation)
                    raw_data[game_num][0].append(
                            observation['vectorized'])
                    raw_data[game_num][1].append(one_hot_action_vector)

                    if observation['current_player'] == agent_id:
                        assert action is not None
                        current_player_action = action
                    else:
                        assert action is None

                    observations, _, game_done, _ = self.environment.step(
                            current_player_action)
                    if game_done:
                        break

        return raw_data

# FIX ME all the code below need to change accordingly
#
# def parse():
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--agent_name',
#                       default='rainbow1')
#
#   parser.add_argument('--num_players',
#                       type=int)
#
#   parser.add_argument('--num_games',
#                       type=int)
#
#   parser.add_argument('--datapath')
#
#   parser.add_argument('--rainbowdir') #FIXME
#
#   args = parser.parse_args()
#   return args


def main(args):
    data_creator = DataCreator()
    rainbow_data = data_creator.create_data()
    pickle.dump(rainbow_data, open(args.datapath, "wb"))

if __name__ == '__main__':
    main()
