import pandas as pd
import glob
from hyper_search import *
from os import system

#ID_PC = [35, 38, 40, 41, 42,  43, 44, 47, 48, 49,
#         50, 51, 52, 53, 54,  56, 57, 58, 59, 60]
#PATH_RANDPARAMS = 'pkl/randparams_2000_2264b0.pkl'

def para_csif(username, pc, path_randparams):
    """ Run a random search for every set in a given .pkl file generated by
        hyper_search:gen_rands() on multiple CSIF computers simultaneously.
        Results are saved under `ganabi/baseline/output/hyper_search` on the
        remote disk. Guest machine must have `XQuartz`.

    Before using this function, Three things have to be done:
        1. Have the repository cloned onto a CSIF machine with .pkl files
           created.
        2. Have setup keyless login. See 'https://goo.gl/9xFJTA'
        3. Make sure the computers that are going to be used are functional.
           Check 'https://goo.gl/fa7jS7' for which CSIF computers are up.
    Arguments:
        - username: str
            CSIF login username.
        - pc: list
            A list of IDs of PCs to be used.
        - path_randparams: str
            Path to the .pkl file that contains the hyperparams.
    """
    with open(path_randparams, 'rb') as f:
        list_hypers = pickle.load(f)

    n = len(list_hypers)

    # Evenly divide the work to each remote machine
    splits = np.array_split(np.arange(n), len(pc))

    for i, split in enumerate(splits):
        start, end = split[0], (split[-1]+1)

        # Message to be printed in each external window
        msg = "PC: {}; Split: {} {}".format(pc[i], start, end)
        # Command to be executed on remote machine
        cmd = "ssh {}@pc{}.cs.ucdavis.edu ".format(username, pc[i])
        cmd += ("python3 -u ganabi/baseline/hyper_search.py "
                "{} {} '{}'".format(start, end, path_randparams))

        system("xterm -e \"echo \\\"{}\\\"; {}; $SHELL\" &".format(msg, cmd))

def fetch_results(username, pc=60,
                  path_results='~/ganabi/baseline/output/hyper_search'):
    """ Fetch the output files from the remote CSIF disk.

    Arguments:
        - username: str
            CSIF login username.
        - path_randparams: str
            Path to the directory that contains the outputs on the remote disk.
        - pc: int, default 60
            PC ID to be used to download the outputs. One is enough.
    """
    cmd = ("scp -r {}@pc{}.cs.ucdavis.edu:{}/* "
           "output/hyper_search/".format(username, pc, path_results))
    system(cmd)

def merge_results(path='output/hyper_search'):
    """ Merge the fetched results into one single .csv.

    Arguments:
        - path: str, default 'output/hyper_search'
            Path to the directory on local machine containing the fetched
            results.
    """
    # Get a list of non-empty .csv
    files = list(filter(lambda f: os.path.getsize(f) > 0,
                        glob.glob(path + '/*.csv')))
    path_merge = 'output/hs_merged.csv'
    # Merge to existing `hs_merged.csv`
    try:
        df = pd.read_csv(path_merge)
        df = pd.concat([df] + [pd.read_csv(f) for f in files])
    except:
        df = pd.concat([pd.read_csv(f) for f in files])

    df = df.sort_values(by='acc', ascending=False)
    df = df.drop_duplicates().reset_index(drop=True)

    df.to_csv(path_merge, index=False)
    print("# of combinations: {}".format(df.shape[0]))