# OpenPAV Linux Quick Start

## 4) Run the full pipeline
```
python run_all.py --benchmark_model idm


python run_all.py --data_csv data/MicroSimACC/step3.csv --fav_id 0
python run_all.py --data_csv data/CATS/step3_ACC.csv --fav_id 0
python run_all.py --data_csv data/OpenACC/step3_Casale.csv --fav_id 0
python run_all.py --data_csv data/OpenACC/step3_ASta.csv --fav_id 0
python run_all.py --data_csv data/Ohio/step3_single_vehicle.csv --fav_id 0
python run_all.py --data_csv data/Vanderbilt/step3_two_vehicle_ACC.csv --fav_id 0

```

tmux new -s job1
conda activate myenv
python train.py

保存
Ctrl + b 然后按 d

查看
tmux ls
tmux attach -t job1

关闭进程
exit


### Common optional flags
```
python run_all.py \
  --data_csv data/MicroSimACC/step3.csv \
  --fav_id 0 \
  --seq_len 1 \
  --train_epochs 50 \
  --mc_iterations 20000 \
  --mc_interval 5000
```

## Output layout
```
results/<data_folder>_FAV<id>/
  behavior_modeling/
  benchmark/
  Markov/
```

## Notes
- Run commands from the project root, or relative paths will fail.
- CSV must include columns: `Trajectory_ID, Time_Index, ID_FAV, Spatial_Gap, Speed_LV, Speed_FAV, Acc_FAV`.
- The evaluation stage uses the trained model saved in the run’s `behavior_modeling/` folder.
