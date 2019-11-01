# $1: file name of the weights without extension. Ex: WTFWT-32
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/tmp/cuda/lib64
export PYTHONPATH=/tmp/ganabi-pkg:$PYTHONPATH
cd /tmp/ganabi/experts/reinf_trainer
python -um train --base_dir ~/reinf_checkpoints/$1 --gin_files configs/hanabi_rainbow.gin --weights /tmp/$1.h5
