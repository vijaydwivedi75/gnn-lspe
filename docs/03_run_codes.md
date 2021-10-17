# Reproducibility


<br>

## 1. Usage


<br>

### In terminal

```
# Run the main file (at the root of the project)
python main_ZINC_graph_regression.py --config 'configs/GatedGCN_ZINC_LSPE.json' # for CPU
python main_ZINC_graph_regression.py --gpu_id 0 --config 'configs/GatedGCN_ZINC_LSPE.json' # for GPU
```
The training and network parameters for each experiment is stored in a json file in the [`configs/`](../configs) directory.




<br>

## 2. Output, checkpoints and visualizations

Output results are located in the folder defined by the variable `out_dir` in the corresponding config file (eg. [`configs/GatedGCN_ZINC_LSPE.json`](../configs/GatedGCN_ZINC_LSPE.json) file).  

If `out_dir = 'out/GatedGCN_ZINC_LSPE_noLapEigLoss/'`, then 

#### 2.1 To see checkpoints and results
1. Go to`out/GatedGCN_ZINC_LSPE_noLapEigLoss/results` to view all result text files.
2. Directory `out/GatedGCN_ZINC_LSPE_noLapEigLoss/checkpoints` contains model checkpoints.

#### 2.2 To see the training logs in Tensorboard on local machine
1. Go to the logs directory, i.e. `out/GatedGCN_ZINC_LSPE_noLapEigLoss/logs/`.
2. Run the commands
```
source activate gnn_lspe
tensorboard --logdir='./' --port 6006
```
3. Open `http://localhost:6006` in your browser. Note that the port information (here 6006 but it may change) appears on the terminal immediately after starting tensorboard.


#### 2.3 To see the training logs in Tensorboard on remote machine
1. Go to the logs directory, i.e. `out/GatedGCN_ZINC_LSPE_noLapEigLoss/logs/`.
2. Run the [script](../scripts/TensorBoard/script_tensorboard.sh) with `bash script_tensorboard.sh`.
3. On your local machine, run the command `ssh -N -f -L localhost:6006:localhost:6006 user@xx.xx.xx.xx`.
4. Open `http://localhost:6006` in your browser. Note that `user@xx.xx.xx.xx` corresponds to your user login and the IP of the remote machine.



<br>

## 3. Reproduce results 


```
# At the root of the project 

bash scripts/ZINC/script_ZINC_all.sh 
bash scripts/OGBMOL/script_MOLTOX21_all.sh 
bash scripts/OGBMOL/script_MOLPCBA_all.sh

```

Scripts are [located](../scripts/) at the `scripts/` directory of the repository.

 

















<br><br><br>