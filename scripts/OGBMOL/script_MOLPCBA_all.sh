#!/bin/bash


############
# Usage
############

# bash script_MOLPCBA_all.sh


####################################
# MOLPCBA - 4 SEED RUNS OF EACH EXPTS
####################################

seed0=41
seed1=95
seed2=12
seed3=35
code=main_OGBMOL_graph_classification.py 
dataset=OGBG-MOLPCBA
tmux new -s gnn_lspe_PCBA -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "source activate gnn_lspe" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GatedGCN_MOLPCBA_NoPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GatedGCN_MOLPCBA_NoPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GatedGCN_MOLPCBA_NoPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GatedGCN_MOLPCBA_NoPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GatedGCN_MOLPCBA_LapPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GatedGCN_MOLPCBA_LapPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GatedGCN_MOLPCBA_LapPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GatedGCN_MOLPCBA_LapPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GatedGCN_MOLPCBA_LSPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GatedGCN_MOLPCBA_LSPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GatedGCN_MOLPCBA_LSPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GatedGCN_MOLPCBA_LSPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/PNA_MOLPCBA_NoPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/PNA_MOLPCBA_NoPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/PNA_MOLPCBA_NoPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/PNA_MOLPCBA_NoPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/PNA_MOLPCBA_LSPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/PNA_MOLPCBA_LSPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/PNA_MOLPCBA_LSPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/PNA_MOLPCBA_LSPE.json' &
wait" C-m
tmux send-keys "tmux kill-session -t gnn_lspe_PCBA" C-m











