# V1-A

I here want to organize the code necessary for this test, see overleaf for docs.
This folder contains:

1. Environment file
    1. contains the simulation definition along with action space, observation space, reward function and termination function.
2. Script to collect data to DP
    1. requires a spacemounse and auto resets at an end of a demo, which is determined based on the termination state from env
3. Script to train DP
    1. train DP and logs to `./.runs/data_mocap/`
4. Script to train IBRL
    1. the main event, here we train the IBRL on the data.
5. Experiment configuration file