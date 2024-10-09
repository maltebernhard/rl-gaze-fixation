# gaze-fixation

## Structure
- Actions:
    - Acceleration in radial and lateral direction
    - Without Gaze Fixation: rotational acceleration
- Observations:
    - (1 - Orientation offset to target)
        - Optional: only if Gaze Fixation is not used
    - 2 - Robot rotational velocity
    - 3 - Robot radial velocity
    - 4 - Robot lateral velocity
    - (5 - Robot distance error)
        - Optional: can be turned off to make state partially observed
    - 6 - State estimate: Orientation offset derivative
        - only needed for gaze fixation
    - (7 - obstacles (orientation offset, relative size in camera image, optional: distance))

- Reward:
    - Increasing reward 1/(error + 1) within margin, 0 else, multiplied by timestep
    - Acceleration (energy waste) penalizes the reward slightly
    - Every second spent at optimal distance to target yields reward 1

## ToDo

- new environment
    - how about classics?
        - drone landing --> look into lunar lander

- manifold representation
    - define space
    - visualize?

- mixture of experts
    - work with mean and stddev mixture model?

- contingencies
    - include mixture mode of simply deciding between actions --> only lacks time variant actions to fit into options framework
    - max radial velocity contingency for obstacles
        - could be relevant to include some understanding of which contingency is active into model

- observations
    - reduce obstacle distance measurement by robot size
    - rethink what's necessary

- Baseline:
    - reimplement?

- Results:
    - include current model into wandb -> how to log arbitrary data?
- Time-variant objective:
    - make target move through space
