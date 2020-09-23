#!/usr/bin/env bash
pyrep_src="../PyRep"
env_name="catalystenv"
tensorboard_port=6380
useGPU=0
useDBPORT=13013
logs_folder=/home/$USER/Desktop
series_name=tutorial_training
config=./configs/config.yml
seed=42

if [ ! -d $env_name ]
then
  echo "Have not found existing $env_name, creating a new one"
  pip install virtualenv
  virtualenv --python=python3.6 $env_name
  . ./$env_name/bin/activate && echo "successfully activated $env_name python is $(which python)"
  cd $pyrep_src
  pip install -r requirements.txt
  python setup.py install
  cd -
  pip install -r requirements.txt
fi
EXP_CONFIG=$config LOGDIR=$logs_folder/$series_name DBPORT=$useDBPORT . ./scripts/prepare_configs.sh
if [[ -z "$series_name" ]]; then
  tb_logdir=${CUR_TB_LOGDIR}; else
  tb_logdir=$logs_folder/$series_name;
fi
tmux new-session \; \
  send-keys 'source '$env_name'/bin/activate' C-m \; \
  send-keys 'tensorboard --logdir='${tb_logdir}' --port='$tensorboard_port' --bind_all' C-m \; \
  split-window -v \;\
  send-keys 'mongod --config configs/_mongod.conf' C-m \; \
  split-window -h \; \
  send-keys 'source '$env_name'/bin/activate' C-m \; \
  send-keys 'CUDA_VISIBLE_DEVICES='$useGPU' catalyst-rl run-trainer --config '$config C-m \; \
  split-window -h -t 0 \; \
  send-keys 'source '$env_name'/bin/activate' C-m \; \
  send-keys 'sleep 20s' C-m \; \
  send-keys 'CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers --seed '$seed' --config '$config C-m \; \
