#!/bin/sh
pyrep_src="../PyRep"
env_name="catalystenv"
useGPU=0
logs_folder=/home/$USER/Desktop
config=./configs/config_inference.yml
seed=42

EXP_CONFIG=$config LOGDIR=$logs_folder DBPORT=$useDBPORT ./bin/prepare_configs.sh
tmux new-session \; \
  send-keys 'source '$env_name'/bin/activate' C-m \; \
  send-keys 'CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers --seed '$seed' --config '$config C-m \; \




