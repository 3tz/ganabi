import csv
import os, sys
import numpy as np
import glob

from create_agent_data import *

def run_game(agents, n=100):
    for ag in agents:
        print(ag)

    num_games = n
    full_score = 25
    score_matrix = []
    avg_score = []
    for a0 in agents:
        print(a0)
        scores_list = []
        avg_list = []
        for a1 in agents:
            print 'vs', a1
            scores = DataCreator(num_games, a0, a1).create_data()
            scores_list.append(scores)
            avg_list.append(sum(scores) * 1.0 / n)
        score_matrix.append(scores_list)
        avg_score.append(avg_list)

    return score_matrix, avg_score, agents
