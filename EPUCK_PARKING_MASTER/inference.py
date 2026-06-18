import os
import gymnasium as gym
from stable_baselines3 import PPO
import EPuckLidarParkingEnv


def main() -> None:
    base_dir = os.path.join(os.getcwd(), "logs")
    new_run_dir = os.path.join(base_dir, "new_model")

    # Must match trainingPPO.py save location exactly
    model_path = os.path.join(new_run_dir, "final_model.zip")

    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}")
        return

    print(f"Loading model from: {model_path}")
    env = gym.make("EPuckLidarParkingEnv-v0")
    model = PPO.load(model_path, env=env)

    obs, _info = env.reset()
    print("Inference started. Press Ctrl+C to stop.")

    try:
        while True:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _info = env.step(action)

            if terminated or truncated:
                print(f"Episode Finished. Resetting... (Final Reward: {reward:.2f})")
                obs, _info = env.reset()

    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        env.close()


if __name__ == '__main__':
    main()