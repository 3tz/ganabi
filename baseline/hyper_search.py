import csv
import sys

from datetime import datetime
from mlp import *
from tensorflow.keras.layers import ReLU, PReLU
from tensorflow.keras.activations import selu
# AGENTS = ['rainbow_agent_' + str(i) for i in range(1, 7)]
AGENTS = ('rainbow_agent_1', 'rainbow_agent_6')
PKL = 'pkl/cvout_15_{}_current.pkl'
FOLDS = (0, )

def mean_acc_agents(n_epoch, folds=FOLDS, agents=AGENTS, pkl_format=PKL,
                    **hps):
    """ Return mean accuracy across all agents with input hyperparameters.

    Arguments:
        - n_epoch: int
            Number of training epochs.
        - folds: tuple
            Indices of folds to be used.
        - agents: tuple
            Names of the agents to be used to calculate the mean accuracy.
        - pkl_format: str
            Format of the naming of CV .pkl outputs. Brackets indicate position
            of agent's name.
            Ex: 'pkl/cvout_15_{}_current.pkl'
        - hps: any
            Hyperparameters to be passed into Mlp object. They should be all
            arguments in Mlp.__init__() besides X_tr, Y_tr, X_va, Y_va,
            io_sizes, out_activation, loss, metrics, and verbose.

    Returns:
        - The average validation accuracy across all agents' best epoch.
    """

    # Avg k-fold accuracies of all agents
    accs = []

    # Build MLP with *args and get best acc for each agent
    for agent in agents:
        # Best accuracies of all folds
        fold_accs = []

        print(agent, ": Fold Accs:", end='', sep='', flush=True)

        with open(PATH+pkl_format.format(agent), 'rb') as f:
            X, Y, masks, ind, cutoffs = pickle.load(f)

        # Use only the specified folds
        masks = [masks[i] for i in folds]

        # For each fold
        for mask in masks:
            X_tr, Y_tr = X[mask], Y[mask]
            X_va, Y_va = X[~mask], Y[~mask]

            m = Mlp(X_tr, Y_tr, X_va, Y_va,
                    io_sizes=(SIZE_OBS_VEC, SIZE_ACT_VEC),
                    out_activation=Softmax, loss='categorical_crossentropy',
                    metrics=['accuracy'], **hps, verbose=0)
            m.construct_model()
            m.train_model(n_epoch=n_epoch, verbose=False)
            best_fold_acc = max(m.hist.history['val_accuracy'])
            fold_accs.append(best_fold_acc)
            print(" {:0.2f}".format(best_fold_acc), end='', sep='', flush=True)

        avg_acc = np.mean(fold_accs)
        accs.append(avg_acc)
        print("; Avg Acc: {:0.3f}".format(avg_acc))

    total_avg_acc = np.mean(accs)
    print("Total Avg Acc:", total_avg_acc)

    return total_avg_acc

def gen_rands(n=2000, range_lr=(0, 1), range_bs=(32, 256), range_nl=(1,3),
              range_ls=(25, 500), act_funcs=(LeakyReLU, ReLU, ELU, PReLU),
              range_decay=(0, 0.01), seed=1234):
    """ Generate random hyperparameters and save them in a .pkl file. A log
        file containing the arguments and the corresponding timestamp of the
        generated .pkl will be also created.

    Arguments:
        - n: int, default 2000
            Number of combinations to try.
        - range_lr: tuple, default (0, 1)
            Range of learning rate (inclusive).
        - range_bs: tuple, default (32, 256)
            Range of batch size (inclusive).
        - range_nl: tuple, default (1, 3)
            Range of number of layers (inclusive).
        - range_ls: tuple, default (25, 500)
            Range of layer sizes, a.k.a, # of neurons / layer (inclusive).
        - act_funcs: tuple, default (LeakyReLU, ReLU, ELU, PReLU)
            Activation functions to choose from for hidden layers.
        - range_decay: tuple, default(0, 0.01)
            Range of learning rate decay.
        - seed: int, default 1234
            Seed for Numpy RNG.
    """
    np.random.seed(seed)

    # Generate hyperparams for each iteration.
    lr = np.random.uniform(range_lr[0], range_lr[1], size=n)
    decay = np.random.uniform(range_decay[0], range_decay[1], size=n)
    bs = np.random.randint(range_bs[0], range_bs[1]+1, size=n)
    nl = np.random.randint(range_nl[0], range_nl[1]+1, size=n)
    hl_acts = np.random.choice(act_funcs, size=n)
    # `layer_sizes` are generated based on value for each iteration in `nl`.
    ls = np.random.randint(range_ls[0], range_ls[1], size=(n, nl.max()))
    reg = np.random.choice([l2(), None], size=n)

    # Dropout shouldn't be on with BatchNorm. Also, search less for cases where
    #   neither is used. np.random.choice() only allows 1-d array, so sample w/
    #   ints first and decode later.
    dec = [(True, False), (False, True), (False, False)]
    cod = np.random.choice([0, 1, 2], size=n, p=[0.45, 0.45, 0.1])
    tups = [dec[i] for i in cod] # in the form of [(F, T), (T, F), ...]
    dropout = [tup[0] for tup in tups]
    bNorm = [tup[1] for tup in tups]


    # List of hyperparams to be saved for all iterations
    list_hypers = []

    for i in range(n):
        cur_nl = nl[i] # number of hiddern layers for current iteration
        cur_acts = (hl_acts[i], ) * cur_nl

        hypers = {'lr': lr[i],
                  'batch_size': bs[i],
                  'hl_activations': cur_acts,
                  'hl_sizes': ls[i, :cur_nl],
                  'decay': decay[i],
                  'bNorm': bNorm[i],
                  'dropout': dropout[i],
                  'regularizer': reg[i]}

        list_hypers.append(hypers)

    # Time-stamp for saving to avoid replacing existing file.
    ts = hex(int((datetime.now()).timestamp()))[4:]
    fn = PATH+'pkl/randparams_{}_{}.pkl'.format(n, ts)
    with open(fn, 'wb') as f:
        pickle.dump(list_hypers, f)

    log_fn = PATH+'pkl/logs_randparams.txt'
    write_header = not os.path.exists(log_fn)

    with open(log_fn, 'a') as f:
        if write_header:
            f.write('Timestamp, n, range_lr, range_bs, range_nl, range_ls, '
                    'act_funcs, range_decay, seed\n')

        f.write(ts + ', ')
        f.write(str(n) + ', ')
        f.write(str(range_lr) + ', ')
        f.write(str(range_bs) + ', ')
        f.write(str(range_nl) + ', ')
        f.write(str(range_ls) + ', ')
        f.write(str([x.__name__ for x in act_funcs]) + ', ')
        f.write(str(range_decay) + ', ')
        f.write(str(seed) + '\n')

def random_search(start, end, num_epoch, path_randparams,
                  path_out=PATH+'output/hyper_search'):
    """ Run models with hyperparams in given .pkl file and save the accuracy to
        @path_out.

    Arguments:
        - start: int
            Starting index of the hyperparams sets to be run.
        - end: int
            Ending index of the hyperparams sets to be run.
            Ex: start=0; end=5; # then only the first five sets will be run.
        - path_randparams: str
            Path to the .pkl file that contains the hyperparams.
        - path_out: str
            Path to the pre-existing directory where the results will be saved.

    """
    with open(path_randparams, 'rb') as f:
        list_hypers = pickle.load(f)

    # Time-stamp for saving to avoid replacing existing file.
    ts = hex(int((datetime.now()).timestamp()))[4:]
    fn = '{}/{}_{}_{}.csv'.format(path_out, start, end, ts)


    with open(fn, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        header = ['idx', 'acc', 'lr', 'batch_size', 'hl_acts', 'hl_sizes',
                  'decay', 'bNorm', 'dropout', 'reg']
        writer.writerow(header)

        for i in range(start, end):
            print('{}:'.format(i))
            hypers = list_hypers[i]
            acc = mean_acc_agents(num_epoch, **hypers)
            # Generate hyperparams values into readable .csv supported format
            vals = [*hypers.values()]
            hl_sizes = vals[3].tolist()
            str_hl_sizes = '-'.join([str(x) for x in hl_sizes])
            if vals[7] is None:
                str_reg = str(None)
            else:
                str_reg = vals[7]._keras_api_names[0]

            row = [i, acc]                # set index & accuracy
            row += vals[0:2]              # lr & batch_size
            row += [vals[2][0].__name__]  # name of acti. func.
            row += [str_hl_sizes]         # num of neurons in hidden layers
            row += vals[4:7]              # decay, bNorm, dropout
            row += [str_reg]              # regularizer

            writer.writerow(row)
            f.flush()

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Must follow the following format:')
        print('python3 hyper_search.py '
              '{start} {end} {epoch} {path to randparams}')
    else:
        random_search(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
                      sys.argv[4])
