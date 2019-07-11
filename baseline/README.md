# ganabi/baseline

A directory with everything related to constructing, training, and testing the baseline model.

## Sample Workflow:

1. Use `ganabi/create_data.py` to generate `n` number of games where `n` must be a multiple of 10 and save the `.pkl` under `ganabi/data`. Let the saved `.pkl` file be `Hanabi-Full_2_6_150.pkl`.

2. Use `cross_validation.py:CV()` to generate training and testing sets for all of the six agents (assume absolute path of `ganabi` is `~/ganabi`):

``` py
for i in range(1, 7):
  CV('~/ganabi/data/Hanabi-Full_2_6_150.pkl', 'rainbow_agent_' + str(i))
```

`.pkl` datasets are generated with different timestamps with this method. For now, simply change the timestamp for each of them to `current` to indicate they are for the current use. They should now look like

```
baseline/pkl/cvout_15_rainbow_agent_1_current.pkl
baseline/pkl/cvout_15_rainbow_agent_2_current.pkl
...
```

3. Now that training and testing data are generated, we can start searching for hyper-parameters. First, generate some sets of hyper-parameters to test on with `hyper_search.py:gen_rands()`. The number of sets and range for each parameter can be changed by adjusting the arguments. Let's generate 100 sets with the default ranges except only searching with activations LeakyReLU and ELU:

``` py
gen_rands(100, act_funcs=(LeakyReLU, ELU))
```

Once this line is executed, a Pickle with the 100 sets of hyper-parameters will be generated under `baseline/pkl`. Let the timestamp be `244ad1` so the filename will look like `randparams_100_244ad1.pkl`, and a new row will be appended to `baseline/pkl/logs_randparams.txt` that stores the hyper-parameters' ranges used for generating this Pickle with its timestamp.

4. Next, we can start building a model for each of the set of the hyper-parameters on the previously created k-fold datasets with `hyper_search.py:random_search()`. Say we want to only train and test the first 50 sets of hyper-parameters with a training epoch of 150, we will do

``` py
path = '~/ganabi/baseline/pkl/randparams_100_244ad1.pkl'
random_search(start=0, end=50, num_epoch=150,
              path_randparams=path)
```

By default, only agents `rainbow_agent_1` & `rainbow_agent_6` and fold 0 are used for calculating the validation accuracy to reduce runtime. These can be manually changed by passing in the desired arguments.

The results will be saved under `baseline/output/hyper_search/`.

5. Alternatively, we can divide the hyper-parameters evenly to multiple remote CSIF machines with `para_csif.py`. This is a naive implementation and only works on MacOS. After reading the descriptions and finishing the setups in `para_csif.py`, let's say we want to divide the work to 5 CSIF machines, we can do

``` py
path = '~/ganabi/baseline/pkl/randparams_100_244ad1.pkl'
para_csif('my_kerbros_username', pc=[1,2,3,4,5], path_randparams=path,
          n_epoch=150)
```

After finishing all of the work, results can be fetched from the remote machines and merged to the existing datasets with

``` py
fetch_results('my_kerbros_username')
merge_results()
```

and now individual results are downloaded to `baseline/output/hyper_search` and a merged results sorted by accuracy can be found in
`baseline/output/hs_merged.csv`.

*Note: CSIFs are not known to be stable, so some machines might not be able to finish all of the requested sets of hyper-parameters; however, since random search is performed here, rows are independent and lack of certain rows do not affect others. Row indices are also saved, so a search on the rows whose indices are missing from here can be performed at a later time if desired. But notice that doing so is no different from performing a search on a newly generated `randparams*.pkl` with a different seed.*

## Files

### `cross_validation.py`

Implementation of a variation of k-fold. [# of Games]/10 pairs of training and validation sets are generated where training set in each pair has a size of 10. Returns can be saved as `.pkl` under `baseline/pkl/`.

### `hyper_search.py`

Contains functions used for hyper-parameters search. Search method uses random search where each hyper-parameter is generated uniformly randomly between a user-defined range.

### `mlp.py`

A class for a more organized building process of MLP w/ Keras.

### `para_csif.py`

A naive implementation of parallel remote computing on CSIF using SSH. Results are saved on remote machines and can be retrieved and merged with functions within this file.

## Sub-directories

### `ganabi/baseline/output`

Hyper-parameter search output results are saved here in `.csv` with naming scheme and columns described below.



**Naming Scheme**:

`{start_row_index}_{end_row_index}_{timestamp}.csv`


**Columns**:

`idx,acc,lr,batch_size,hl_acts,hl_sizes,decay,bNorm,dropout,reg`




### `ganabi/baseline/pkl`

Pickle files of cross-validation datasets and randomly generated hyper-parameters.

**Naming Scheme**:

*cross_validation:* `cvout_{number of train/test pairs}_{agent name}_{"current" | timestamp}.pkl`


*hyper_search:gen_rands:* `randparams_{n}_{timestamp}.pkl`

and details of parameters are automatically stored in `logs_randparams.txt` for the corresponding timestamp after each parameter Pickle is generated.


