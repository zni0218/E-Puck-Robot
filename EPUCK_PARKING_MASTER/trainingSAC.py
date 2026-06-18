import os
import time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import EPuckLidarParkingEnvSAC


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

    # COMPLETAMENTE SEPARADO DO PPO
    base_dir = os.path.join(os.getcwd(), "logs_sac")
    run_dir = os.path.join(base_dir, "parking_sac")

    os.makedirs(run_dir, exist_ok=True)

    raw_env = gym.make("EPuckLidarParkingEnvSAC-v0")
    env = Monitor(
        raw_env,
        filename=os.path.join(run_dir, "monitor.csv")
    )

    checkpoint = os.path.join(run_dir, "final_model.zip")

    if os.path.exists(checkpoint):

        print(f"\n[TRAINING] RESUMING SAC model from {checkpoint}")

        model = SAC.load(
            checkpoint,
            env=env
        )

        reset_timesteps = False

    else:

        print("\n[TRAINING] FRESH START (SAC)")

        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=os.path.join(run_dir, "tensorboard_logs"),

            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,

            gamma=0.99,
            tau=0.02,

            train_freq=1,
            gradient_steps=1,

            policy_kwargs=dict(
                net_arch=[256, 256]
            )
        )

        reset_timesteps = True

    eval_callback = EvalCallback(
        env,
        n_eval_episodes=5,
        eval_freq=20_000,

        best_model_save_path=os.path.join(
            run_dir,
            "best_model"
        ),

        log_path=os.path.join(
            run_dir,
            "eval_logs"
        ),

        deterministic=True,
        render=False
    )

    unwrapped_env = raw_env.unwrapped

    early_stop_callback = EarlyStopCallback(
        unwrapped_env,
        verbose=1
    )

    time_str = time.strftime("%Y%m%d-%H%M%S")

    try:

        model.learn(
            total_timesteps=900_000,
            log_interval=10,
            tb_log_name=time_str,

            callback=[
                eval_callback,
                early_stop_callback
            ],

            reset_num_timesteps=reset_timesteps,
        )

    except KeyboardInterrupt:

        print("\n[TRAINING] Interrupted. Saving...")

    model.save(
        os.path.join(
            run_dir,
            "final_model"
        )
    )

    print(f"\n[TRAINING] Done.")
    print(f"[TRAINING] Model saved to: {run_dir}")


if __name__ == "__main__":
    main()