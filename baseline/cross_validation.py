import pickle
import numpy as np
import random

# TODO:
#   - Add option for choosing int/binary/one-hot; currently just binary
#     Problem: one-hot or int will be too large b/c it's s 658 bit integer.

SIZE_OBS_VEC = 658
SIZE_ACT_VEC = 20

def CV(pkl_path='./data/Hanabi-Full_2_6_150.pkl', agent='rainbow_agent_6',
       type_obs='binary', shuffle=False, seed=1234):
    """ Generate training sets with a size of 10 games and validation sets
        containing the remaining games.

    Arguments:
        - pkl_path: str, default './data/Hanabi-Full_2_6_150.pkl'
            Path to the pickle file that contains the output generated from
            create_data.py. First 10 games are chosen for training, and the
            rest will be for validation.
        - agent: str, default 'rainbow_agent_6'
            Name of the agent to use.
        - type_obs: str, default binary
            Type of the observations. Must be in {'binary', 'int'}.
        - shuffle: boolean, default false
            If true, 10 games are randomly chosen instead of simply choosing
            the first 10.
        - seed: int, default 1234
            Seed for shuffling. Use None to set current time as seed.

    Returns:
        - X_tr: np.matrix
            Training atrix that contains the observations in following format:
            [[observatoin of round 1 in game 1],
             [observatoin of round 2 in game 1],
             ...
             [observatoin of round 1 in game 2],
             ...
             [observatoin of round N in game 10]]
        - Y_tr: np.matrix
            Training matrix that contains the actions in following format:
            [[action of round 1 in game 1],
             [action of round 2 in game 1],
             ...
             [action of round 1 in game 2],
             ...
             [action of round N in game 10]]
        - X_va: np.matrix
            Validation matrix that contains the observations.
        - Y_va: np.matrix
            Validation matrix that contains the actions.
    """

    def bin2int(bin_list):
        """ Converts an binary integer list into an integer.
            Ex: [0,1,0] -> 2
        """
        return int("".join(str(x) for x in bin_list), 2)

    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    lst = raw[agent] # list of info for the chosen agent
    ind = list(range(len(lst))) # game indices

    if shuffle:
        random.seed(seed)
        random.shuffle(ind)

    print("Training game indices:", ind[:10])

    # Determine the size of the matrices
    n_rows = 0
    for n, i in enumerate(ind): # [n]umber of games gone thru & [i]dx of game
        n_rnds =  len(lst[i][0])
        n_rows += n_rnds
        # Save the row num for the training set cutoff
        if n == 9:
            n_tr = n_rows

    X = np.zeros([n_rows, SIZE_OBS_VEC])
    Y = np.zeros([n_rows, SIZE_ACT_VEC])

    cur_idx = 0
    for n, i in enumerate(ind):
        obs = np.matrix(lst[i][0])
        act = np.matrix(lst[i][1])
        X[cur_idx:(cur_idx + obs.shape[0]), :] = obs
        Y[cur_idx:(cur_idx + act.shape[0]), :] = act
        cur_idx += act.shape[0]
        assert(obs.shape[0] == act.shape[0])

    return X[:n_tr, :], Y[:n_tr, :], X[n_tr:, :], Y[n_tr:, :]
