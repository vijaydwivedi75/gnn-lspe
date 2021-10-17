#!/bin/bash


# bash script_tensorboard.sh





tmux new -s tensorboard -d
tmux send-keys "source activate gnn_lspe" C-m
tmux send-keys "tensorboard --logdir out/ --port 6006" C-m









