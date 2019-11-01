cd /tmp
mkdir ganabi-pkg
# Install required packages
pip install tensorflow-gpu==1.14 gin-config cffi -t /tmp/ganabi-pkg/
# Get Cuda & best model weights
wget https://storage.googleapis.com/ganabi/cudnn-10.0-linux-x64-v7.6.2.24.tgz
wget https://storage.googleapis.com/best_models/$1.h5
tar -zxvf cudnn-10.0-linux-x64-v7.6.2.24.tgz
# make
cd ganabi/hanabi_env
cmake .
make
