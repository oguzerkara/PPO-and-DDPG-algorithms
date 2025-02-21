# Investigation of DDPG/PPO Performance and Adaptability in Lunar Lander Environment

## Overview
This project investigates the performance and adaptability of the Proximal Policy Optimization (PPO) algorithm in the Lunar Lander environment. The primary goal is to explore the limits of the PPO algorithm under various noise conditions and determine potential methods to improve its robustness.

## Experimental Setup
- **Training Steps:** 1 million
- **Environment Parameters:**
  - Gravity: -10
  - Wind: 20
  - Turbulence: 2
  - Initial random force range: 1500 and 2000 (two separate models trained)

### Noise and Environmental Variations
The following environmental conditions were simulated:
- **Wind Speeds:** Up to 20
- **Turbulence Levels:** Up to 2
- **Random Initial Force Ranges:** -2000 to 2000
- **Extreme Forces:** Upward and downward forces were simulated
- **Dynamic Random Force:** Applied every 100 steps

**Key Finding:** The models showed strong adaptability except when gravity increased from -10 to -1, where performance declined significantly.

## Random Force Impact Analysis
A modified version of the Lunar Lander environment was developed where the random force increased over time:
- **Training Variants:**
  - Three models trained with a force range increasing up to 1000
  - Three models trained with a force range increasing up to 1500

**Results:**
- Models with a maximal force range of 1000 performed better than those with a range of 1500.

## Solutions for Handling Extreme Noises
### 1. Entropy Coefficient Adjustment
- Increasing the entropy coefficient resulted in a slight improvement in negative reward values.
- However, the adjustment alone was insufficient to significantly enhance performance.

### 2. Stacked Frames Approach
Implemented using `VecStackFrame` from Stable Baselines3:
- **DummyVecEnv:** Sequential execution of multiple environments
- **SubprocVecEnv:** True parallel execution of multiple environments

**Findings:**
- **SubprocVecEnv performed significantly better**:
  - Faster training (12 min vs. 18 min)
  - Higher rewards (~100 more compared to DummyVecEnv)

### 3. Stacked Observations
Stacking multiple state representations to provide better temporal understanding.
- **4-State Stacking:** Improved reward values significantly
- **8-State Stacking:** Slightly better performance than 4-state stacking
- **15-State Stacking:** Caused inefficiency and resulted in negative rewards for lower gravity values

**Best Configuration:** 8 stacked observations with SubprocVecEnv for optimal balance between performance and efficiency.

Results are found in Wiki: https://github.com/oguzerkara/PPO-and-DDPG-algorithms/wiki/Outcomes
## Conclusion
- PPO performance is highly affected by environmental randomness and gravity shifts.
- **Stacked observations and frame stacking provide a robust solution** for improving PPO in noisy environments.
- Increasing entropy coefficient alone does not provide significant benefits.
- **Parallel processing (`SubprocVecEnv`) is the preferred approach** for optimizing PPO training efficiency.
