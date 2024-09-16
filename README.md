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

- manifold representation
    - define space
    - visualize

- mixture of experts
    - decouple agent and environment concept, so that an agent actually represents a decision maker (controller or policy)
    - implement mixture model
        - each agent outputs mean and stddev for each action space dimension,
        - weighting function learns matrix for each dimension / agent

- multi contingencies
    - make mixture of obstacle avoidance and gf possible
    - contingency:
        - the closer, the slower the radial velocity
            - does not conflict with gaze fixation, as different action space dimensions are adressed
        - what if radial velocity is negative?
        - what if distance is large --> we don't need to approach obstacle
        - IF radial velocity is positive, reduce it by size of obstacle in fov?
        

- think about which observations are relavant to solving the problem
    - should no history be included whatsoever?

- Baseline:
    - improve baseline distance estimation by finding correct alphas
    - alternative baseline: reduce all velocities to 0 when close to target

- Results:
    - document difficulties in generating useful behavior -> too high learning rate
    - include baseline into wandb -> how to log data?
- Time-variant objective:
    - make distance to target time-variant and include in observations / exclude from config
    - make target move through space
- Obstacles
    - rethink representation
