"""
patient_simulator.py

A Gymnasium Reinforcement Learning environment to simulate a virtual patient
with type 2 diabetes. The goal is to learn an optimal drug dosage policy.
"""

from typing import Tuple, Dict, Any
import gymnasium
import numpy as np
from scipy.integrate import solve_ivp
from gymnasium import spaces

class PersonalizedMedicineEnv(gymnasium.Env):
    """
    A custom Gymnasium environment for simulating personalized medicine for a
    patient with type 2 diabetes.

    The environment models the patient's glucose level response to two different
    drugs. An RL agent's goal is to administer daily dosages to keep the
    patient's glucose within a healthy range.

    Attributes:
        metadata (dict): Metadata for rendering.
        patient_profile (dict): Parameters defining the patient's physiology.
        action_space (gymnasium.spaces.Box): The space of possible actions (drug dosages).
        observation_space (gymnasium.spaces.Box): The space of possible observations.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, patient_profile: Dict[str, float]):
        """
        Initializes the environment.

        Args:
            patient_profile (dict): A dictionary containing patient-specific
                parameters like 'metabolism_rate' and 'insulin_resistance'.
        """
        super().__init__()

        # --- Patient-specific parameters ---
        self.patient_profile = patient_profile
        self.metabolism_rate = patient_profile.get('metabolism_rate', 1.0)
        self.insulin_resistance = patient_profile.get('insulin_resistance', 1.0)

        # --- Simulation & Model Constants ---
        self.time_step_duration = 24  # Each step simulates 24 hours
        self.max_episode_days = 90    # Truncate after 90 days
        self.initial_glucose_mean = 160.0  # mg/dL
        self.initial_glucose_std = 10.0 # Add some randomness to starting glucose

        # --- Define action and observation spaces ---
        # Action: Dosage for Drug A and Drug B (e.g., in mg)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([50.0, 50.0]),
            dtype=np.float32
        )

        # Observation: [glucose, drug_a_conc, drug_b_conc, time_of_day]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([600.0, 100.0, 100.0, 23.99]), # Increased glucose high for safety
            dtype=np.float32
        )

        # --- Initialize state ---
        self.state = None
        self.current_day = 0

    def _get_obs(self) -> np.ndarray:
        """
        Helper method to return the current state as a NumPy array.
        The time_of_day is fixed at the beginning of the 24h cycle (e.g., 8 AM).
        """
        obs = np.array([
            self.state[0],  # Glucose
            self.state[1],  # Drug A concentration
            self.state[2],  # Drug B concentration
            8.0             # Time of day (fixed at 8 AM for dosage)
        ], dtype=np.float32)
        return obs

    def _calculate_reward(self, glucose_history: np.ndarray, dosage_a: float, dosage_b: float) -> float:
        """
        Calculates the reward for a 24-hour glucose trajectory.

        Args:
            glucose_history (np.array): Time-series of glucose values over 24 hours.
            dosage_a (float): The dosage of drug A administered.
            dosage_b (float): The dosage of drug B administered.

        Returns:
            float: The calculated reward.
        """
        # --- Define glucose ranges ---
        safe_range = (60, 350)
        target_range = (80, 140)

        # --- Severe penalty for leaving the safe range ---
        if np.any(glucose_history < safe_range[0]) or np.any(glucose_history > safe_range[1]):
            return -100.0

        # --- Reward for time in target range ---
        time_in_target = np.sum(
            (glucose_history >= target_range[0]) & (glucose_history <= target_range[1])
        )
        reward = float(time_in_target) # +1 for each hour in range

        # --- Penalty for drug dosage (to encourage efficiency) ---
        dosage_penalty = 0.1 * (dosage_a + dosage_b)
        reward -= dosage_penalty

        return reward

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): The seed for the random number generator.
            options (dict, optional): Additional options for resetting.

        Returns:
            tuple: A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed)

        # Reset to morning on day 0 with no drugs in the system
        self.current_day = 0
        # Add some randomness to the initial glucose level for better generalization
        initial_glucose = self.np_random.normal(self.initial_glucose_mean, self.initial_glucose_std)
        self.state = np.array([initial_glucose, 0.0, 0.0]) # [glucose, drug_a_conc, drug_b_conc]

        observation = self._get_obs()
        info = {}

        return observation, info

    def _system_dynamics(self, t: float, y: np.ndarray) -> list[float]:
        """
        Defines the patient's physiological dynamics via a system of ODEs.

        Args:
            t (float): Current time in the simulation step (from 0 to 24).
            y (np.ndarray): The state vector [glucose, conc_a, conc_b].

        Returns:
            list[float]: The derivatives [d_glucose_dt, d_conc_a_dt, d_conc_b_dt].
        """
        glucose, conc_a, conc_b = y

        # --- Glucose Dynamics ---
        # Glucose production varies sinusoidally (peaks mimicking meals)
        glucose_production = 10 * (1 + np.sin(np.pi * t / 12))
        # Insulin effect is reduced by the patient's specific resistance level
        insulin_effect = 5 * (glucose / 100) / self.insulin_resistance

        # --- Drug Dynamics ---
        effect_a = 0.5 * conc_a
        effect_b = 0.8 * conc_b
        interaction_effect = 0.1 * conc_a * conc_b # Synergistic/antagonistic term

        # --- Clearance Rates ---
        # Drug clearance is affected by the patient's metabolism rate
        clearance_rate_a = 0.1 * self.metabolism_rate
        clearance_rate_b = 0.08 * self.metabolism_rate

        # --- Differential Equations ---
        d_glucose_dt = glucose_production - insulin_effect - (effect_a + effect_b - interaction_effect)
        d_conc_a_dt = -clearance_rate_a * conc_a # Absorption is instant at step start
        d_conc_b_dt = -clearance_rate_b * conc_b

        return [d_glucose_dt, d_conc_a_dt, d_conc_b_dt]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step in the environment.

        Args:
            action (np.array): The action to take (dosages for Drug A and B).

        Returns:
            tuple: A tuple containing the next observation, reward, terminated flag,
                   truncated flag, and an info dictionary.
        """
        dosage_a, dosage_b = action

        # --- Solve the ODEs for the next 24 hours ---
        t_span = [0, self.time_step_duration]
        t_eval = np.linspace(t_span[0], t_span[1], self.time_step_duration + 1)

        # The administered dose is modeled as an immediate increase in concentration
        y0 = self.state.copy()
        y0[1] += dosage_a
        y0[2] += dosage_b

        sol = solve_ivp(
            fun=self._system_dynamics,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
        )

        # --- Update state and day ---
        self.state = sol.y[:, -1]
        self.current_day += 1

        # --- Calculate reward ---
        glucose_history = sol.y[0, :]
        reward = self._calculate_reward(glucose_history, dosage_a, dosage_b)

        # --- Check for termination and truncation ---
        terminated = (np.any(glucose_history < 60) or np.any(glucose_history > 350))
        truncated = (self.current_day >= self.max_episode_days)

        # --- Get next observation ---
        observation = self._get_obs()
        info = {'glucose_history': glucose_history}

        return observation, reward, terminated, truncated, info


if __name__ == '__main__':
    """
    Example usage and environment validation.
    """
    from gymnasium.utils.env_checker import check_env

    # 1. Define a sample patient profile
    patient_profile = {
        'metabolism_rate': 0.8,   # Slower metabolism
        'insulin_resistance': 1.5 # Higher resistance
    }
    print(f"Creating environment for patient with profile: {patient_profile}")

    # 2. Instantiate the environment
    env = PersonalizedMedicineEnv(patient_profile)

    # 3. Check if the environment conforms to the Gymnasium API
    print("Running Gymnasium environment checker...")
    try:
        check_env(env)
        print("\n✅ Success! The environment is valid and conforms to the Gymnasium API.")
    except Exception as e:
        print(f"\n❌ Error! The environment failed the check: {e}")

    # --- Example of a few steps ---
    print("\n--- Running a short example episode ---")
    obs, info = env.reset()
    for i in range(5):
        action = env.action_space.sample() # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Day {i+1}:")
        print(f"  Action (Dosages): Drug A={action[0]:.2f}, Drug B={action[1]:.2f}")
        print(f"  Next State (Obs): Glucose={obs[0]:.2f}, Conc A={obs[1]:.2f}, Conc B={obs[2]:.2f}")
        print(f"  Reward: {reward:.2f}")
        if terminated or truncated:
            print("Episode finished.")
            break
