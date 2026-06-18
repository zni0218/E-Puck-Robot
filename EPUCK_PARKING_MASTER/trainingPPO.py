import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import EPuckLidarParkingEnv


class EarlyStopCallback(BaseCallback):
    def __init__(self, env: gym.Env, verbose: int = 1):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        if getattr(self.env, "should_stop", False):
            if self.verbose:
                print("\n[EarlyStopCallback] Early stop triggered.")
            return False
        return True


def main() -> None:
    base_dir: str = os.path.join(os.getcwd(), "logs")
    new_run_dir: str = os.path.join(base_dir, "new_model")
    os.makedirs(new_run_dir, exist_ok=True)

    raw_env = gym.make("EPuckLidarParkingEnv-v0")
    env = Monitor(raw_env, filename=os.path.join(new_run_dir, "monitor.csv"))

    # Check new_model first, then fall back to best_model, then fresh start
    new_checkpoint = os.path.join(new_run_dir, "final_model.zip")
    old_checkpoint = os.path.join(base_dir, "final_model.zip")

    if os.path.exists(new_checkpoint):
        print(f"\n[TRAINING] RESUMING new model from {new_checkpoint}")
        model = PPO.load(new_checkpoint, env=env)
        model.ent_coef = 0.01
        model.learning_rate = 3e-4
        reset_timesteps = False
    elif os.path.exists(old_checkpoint):
        print(f"\n[TRAINING] LOADING old model from {old_checkpoint}")
        model = PPO.load(old_checkpoint, env=env)
        model.ent_coef = 0.01
        model.learning_rate = 3e-4
        reset_timesteps = False
    else:
        print("\n[TRAINING] FRESH START")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            ent_coef=0.01,
            learning_rate=3e-4,
            n_steps=1024,  # mais responsivo
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[64, 64])
        )
        reset_timesteps = True

    eval_callback = EvalCallback(
        env,
        n_eval_episodes=5,
        eval_freq=5000,
        best_model_save_path=os.path.join(new_run_dir, "best_model"),
        log_path=os.path.join(new_run_dir, "eval_logs"),
        deterministic=True,
        render=False
    )

    unwrapped_env = raw_env.unwrapped
    early_stop_callback = EarlyStopCallback(unwrapped_env, verbose=1)

    time_str = time.strftime("%Y%m%d-%H%M%S")

    try:
        model.learn(
            total_timesteps=900_000,
            log_interval=2,
            tb_log_name=time_str,
            callback=[eval_callback, early_stop_callback],
            reset_num_timesteps=reset_timesteps,
        )
    except KeyboardInterrupt:
        print("\n[TRAINING] Interrupted. Saving...")

    model.save(os.path.join(new_run_dir, "final_model"))
    print(f"[TRAINING] Done. Model saved to {new_run_dir}/")


if __name__ == '__main__':
    main()