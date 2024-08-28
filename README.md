# gaze-fixation

## Structure
- Actions:
    - Acceleration in radial and lateral direction
    - Without Gaze Fixation: rotational acceleration
- Observations:
    - Orientation offset to target
    - Robot rotational velocity
    - Robot radial velocity
    - Robot lateral velocity
    - Robot distance error
    - State estimate: Orientation offset velocity
- Reward:
    - Increasing reward 1/(error + 1) within margin, 0 else, multiplied by timestep
    - Every second spent at optimal distance to target yields reward 1

## ToDo
- think about which observations are relavant to solving the problem
- unfair: without gaze fixation, x/y-control may be more efficient at solving the task --> test this
- Results:
    - document difficulties in generating useful behavior
    - include baseline into wandb
- Time-variant objective:
    - make distance to target time-variant and include in observations / exclude from config
    - make target move through space
- Obstacles
    - include in representation
    - long run: make robot be unable to see past obstacles
- rework load_and_run
