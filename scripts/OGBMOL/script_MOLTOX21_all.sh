#!/bin/bash


############
# Usage
############

# bash script_MOLTOX21_all.sh


####################################
# MOLTOX21 - 4 SEED RUNS OF EACH EXPTS
####################################

seed0=41
seed1=95
seed2=12
seed3=35
code=main_OGBMOL_graph_classification.py 
dataset=OGBG-MOLTOX21
tmux new -s gnn_lspe_TOX21 -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "source activate gnn_lspe" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GatedGCN_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GatedGCN_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GatedGCN_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GatedGCN_MOLTOX21_NoPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GatedGCN_MOLTOX21_LapPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GatedGCN_MOLTOX21_LapPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GatedGCN_MOLTOX21_LapPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GatedGCN_MOLTOX21_LapPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GatedGCN_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GatedGCN_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GatedGCN_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GatedGCN_MOLTOX21_LSPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/PNA_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/PNA_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/PNA_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/PNA_MOLTOX21_NoPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/PNA_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/PNA_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/PNA_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/PNA_MOLTOX21_LSPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/PNA_MOLTOX21_LSPE_withLapEigLoss.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/PNA_MOLTOX21_LSPE_withLapEigLoss.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/PNA_MOLTOX21_LSPE_withLapEigLoss.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/PNA_MOLTOX21_LSPE_withLapEigLoss.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SAN_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SAN_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SAN_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SAN_MOLTOX21_NoPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SAN_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SAN_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SAN_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SAN_MOLTOX21_LSPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GraphiT_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GraphiT_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GraphiT_MOLTOX21_NoPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GraphiT_MOLTOX21_NoPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GraphiT_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GraphiT_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GraphiT_MOLTOX21_LSPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GraphiT_MOLTOX21_LSPE.json' &
wait" C-m
tmux send-keys "tmux kill-session -t gnn_lspe_TOX21" C-m











