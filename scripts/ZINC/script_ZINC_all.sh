#!/bin/bash


############
# Usage
############

# bash script_ZINC_all.sh


####################################
# ZINC - 4 SEED RUNS OF EACH EXPTS
####################################

seed0=41
seed1=95
seed2=12
seed3=35
code=main_ZINC_graph_regression.py 
dataset=ZINC
tmux new -s gnn_lspe_ZINC -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "source activate gnn_lspe" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GatedGCN_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GatedGCN_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GatedGCN_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GatedGCN_ZINC_NoPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GatedGCN_ZINC_LapPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GatedGCN_ZINC_LapPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GatedGCN_ZINC_LapPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GatedGCN_ZINC_LapPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GatedGCN_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GatedGCN_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GatedGCN_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GatedGCN_ZINC_LSPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GatedGCN_ZINC_LSPE_withLapEigLoss.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GatedGCN_ZINC_LSPE_withLapEigLoss.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GatedGCN_ZINC_LSPE_withLapEigLoss.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GatedGCN_ZINC_LSPE_withLapEigLoss.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/PNA_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/PNA_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/PNA_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/PNA_ZINC_NoPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/PNA_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/PNA_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/PNA_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/PNA_ZINC_LSPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SAN_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SAN_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SAN_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SAN_ZINC_NoPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SAN_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SAN_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SAN_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SAN_ZINC_LSPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GraphiT_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GraphiT_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GraphiT_ZINC_NoPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GraphiT_ZINC_NoPE.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/GraphiT_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/GraphiT_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/GraphiT_ZINC_LSPE.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/GraphiT_ZINC_LSPE.json' &
wait" C-m
tmux send-keys "tmux kill-session -t gnn_lspe_ZINC" C-m













