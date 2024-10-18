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

- play around with reward and penalty margins / obstacle proximity vs. collision penalty?

- Important considerations
    - additional skills (go sideways, fixate elsewhere, ...)
        - maybe apply gaussians here?
    - how to put sensorimotor contingencies into mixture model?

- apply wandb logging

- contingencies
    - max radial velocity contingency for obstacles
        - could be relevant to include some understanding of which contingency is active into model

- Time-variant objective:
    - make target move through space

- new environment
    - how about classics?
        - drone landing --> look into lunar lander
