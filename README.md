# 
This notebook accompanies the technical report, "<a href="https://arxiv.org/abs/2108.06491">DQN Control Solution for KDD Cup 2021 City Brain Challenge</a>" by Yitian Chen, Kunlong Chen, Kunjin Chen, and Li Wang presented at KDD 2021 ,KDD cup Workshop City Brain Challenge

The notebook provides codes of DQN solution for City Brain Challenge 2021, which  achieved the 8th place in the final leaderboard and top3 of the DQN solutions (Most of winning solutions were based on heuristic strategies and parameter search).

## Preliminary
   * For competition background, simulation environment configuration, users can refer the document https://kddcup2021-citybrainchallenge.readthedocs.io/en/latest/city-brain-challenge.html as the starter kit.
   * Our DQN model is implemented in pytorch, users can install the pytorch by the simple command ``pip3 install torch'' 

## Model training and submit
   * Run 'python3 DQNSimulation120.py' for simulation.
   * You can get model parameter file in the fold 'agent/Params'. e.g., 'epoch_DQN1002_48_metrics_-47428.740.param'.
   * Load the model file in the file agent.py 
