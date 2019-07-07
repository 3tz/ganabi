This is a directory where `.pkl` files are saved for baseline.

Outputs for `cross_validation.py` are named in the following format:

`cvout_{number of train/test pairs}_{agent name}_{"current" | timestamp}.pkl`

Outputs for `hyper_search:gen_rands()` are named in the following format, and
details of parameters are stored in `logs_randparams.txt` for corresponding
timestamp:

`randparams_{n}_{timestamp}.pkl`
