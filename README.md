# Gaze-Fixation: Reinforcement Learning for Target Reaching and Obstacle Avoidance

This project implements a 2D simulation environment for training a holonomic robot using reinforcement learning. The robot is tasked with reaching a target while avoiding obstacles. The framework supports modular agent architectures, including policy-based, contingency-based, and mixture-of-experts models, and leverages the `stable-baselines3` library for training.

## Features

- **Environment**: 
  - 2D holonomic robot with configurable dynamics (e.g., max velocity, acceleration, sensor angle).
  - Targets and obstacles are dynamically generated.
  - Customizable reward structure for proximity to the target, obstacle avoidance, energy efficiency, and collision penalties.

- **Agent Architectures**:
  - **Policy Agents**: Predefined behaviors such as moving towards the target or avoiding obstacles.
  - **Contingency Agents**: Reactive agents that adapt based on specific conditions.
  - **Mixture-of-Experts (MoE)**: Combines multiple sub-agents to handle complex tasks.

- **Reinforcement Learning**:
  - Supports training with PPO (Proximal Policy Optimization) and other RL algorithms.
  - Modular callbacks for logging, plotting, and monitoring training progress.

- **Visualization**:
  - Real-time rendering of the environment using `pygame`.
  - Tools for visualizing action fields, training progress, and agent behavior.

## Project Structure

- **`environment/`**: Contains the simulation environment (`GazeFixEnv`) and its components (robot, obstacles, target).
- **`agent/`**: Implements agent architectures, including policy, contingency, and mixture models.
- **`utils/`**: Utility scripts for logging, plotting, and user interaction.
- **`config/`**: YAML configuration files for environment and agent setups.
- **`train_model.py`**: Script for training agents.
- **`test_env.py`**: Script for testing the environment and agent behavior.

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies: Install via `pip install -r requirements.txt` (ensure `stable-baselines3`, `pygame`, `matplotlib`, `numpy`, `wandb`, etc.).

### Running the Simulation

1. **Train an Agent**:
   ```bash
   python train_model.py
   ```
   Modify the agent and environment configurations in config as needed.

2. **Test the Environment**:
   ```bash
   python test_env.py
   ```

3. **Load and Run a Pretrained Agent**:
   ```bash
   python load_run.py
   ```

### Visualization

- Training progress and agent behavior can be visualized using the plotting utilities in plotting.py.

## Configuration

- **Environment**: Configure parameters like timestep, world size, number of obstacles, and robot dynamics in env.
- **Agents**: Define agent types, observation keys, and reward indices in agent.

## License

This project is for academic and research purposes. Please contact the author for usage in other contexts.
