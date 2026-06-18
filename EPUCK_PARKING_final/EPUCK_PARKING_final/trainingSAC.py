import os
import time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import EPuckLidarParkingEnvSAC


# Callback simples para parar o treino caso o ambiente sinalize isso
class EarlyStopCallback(BaseCallback):

    def __init__(self, env: gym.Env, verbose: int = 1):
        super().__init__(verbose)
        self.env = env  # referência ao ambiente para verificar flags internas

    def _on_step(self) -> bool:
        # chamado a cada passo do treino
        if getattr(self.env, "should_stop", False):

            # imprime mensagem caso esteja ativo o modo verbose
            if self.verbose:
                print("\n[EarlyStopCallback] Early stop triggered.")

            # False = interrompe o treino imediatamente
            return False

        return True


def main() -> None:

    # função principal responsável por iniciar / continuar treino SAC

    # diretórios onde tudo será guardado
    base_dir = os.path.join(os.getcwd(), "logs_sac")  # pasta principal do SAC
    run_dir = os.path.join(base_dir, "parking_sac")  # pasta desta execução específica

    # garante que a pasta existe
    os.makedirs(run_dir, exist_ok=True)

    # criação do ambiente de simulação (Webots + Gymnasium)
    raw_env = gym.make("EPuckLidarParkingEnvSAC-v0")

    # wrapper que regista recompensas e estatísticas do treino
    env = Monitor(
        raw_env,
        filename=os.path.join(run_dir, "monitor.csv")
    )

    # caminho do modelo final guardado
    checkpoint = os.path.join(run_dir, "final_model.zip")

    # =========================================================
    # LÓGICA DE CARREGAMENTO / INICIALIZAÇÃO DO MODELO SAC
    # =========================================================

    if os.path.exists(checkpoint):

        # se já existe modelo treinado anteriormente, retoma treino
        print(f"\n[TRAINING] RESUMING SAC model from {checkpoint}")

        model = SAC.load(
            checkpoint,
            env=env
        )

        reset_timesteps = False  # continua contagem de steps anterior

    else:

        # se não existe modelo, inicia treino do zero
        print("\n[TRAINING] FRESH START (SAC)")

        model = SAC(
            "MlpPolicy",  # rede neural fully connected (MLP)

            env,  # ambiente de treino

            verbose=1,  # logs detalhados

            # logs para TensorBoard
            tensorboard_log=os.path.join(run_dir, "tensorboard_logs"),

            # parâmetros principais do SAC
            learning_rate=3e-4,
            buffer_size=300_000,  # replay buffer
            batch_size=256,  # tamanho dos batches de treino

            gamma=0.99,  # desconto de recompensa futura
            tau=0.02,  # atualização suave da target network

            # frequência de treino
            train_freq=1,
            gradient_steps=1,

            # arquitetura da rede neural (2 camadas de 256 neurónios)
            policy_kwargs=dict(
                net_arch=[256, 256]
            )
        )

        reset_timesteps = True  # inicia contagem de treino do zero

    # =========================================================
    # CALLBACK DE AVALIAÇÃO DO MODELO
    # =========================================================

    eval_callback = EvalCallback(
        env,  # ambiente usado para avaliação

        n_eval_episodes=5,  # número de episódios de avaliação
        eval_freq=20_000,  # frequência de avaliação durante treino

        best_model_save_path=os.path.join(
            run_dir,
            "best_model"
        ),  # guarda automaticamente o melhor modelo

        log_path=os.path.join(
            run_dir,
            "eval_logs"
        ),  # logs da avaliação

        deterministic=True,  # ações determinísticas na avaliação
        render=False  # sem renderização para acelerar
    )

    # ambiente sem wrappers para permitir leitura de flags internas
    unwrapped_env = raw_env.unwrapped

    # callback que permite parar treino manualmente via variável interna
    early_stop_callback = EarlyStopCallback(
        unwrapped_env,
        verbose=1
    )

    # timestamp usado para identificar runs no TensorBoard
    time_str = time.strftime("%Y%m%d-%H%M%S")

    # =========================================================
    # TREINO PRINCIPAL DO MODELO
    # =========================================================

    try:

        model.learn(
            total_timesteps=900_000,  # número total de passos de treino

            log_interval=10,  # frequência de logs no terminal

            tb_log_name=time_str,  # nome do run no TensorBoard

            # callbacks ativos durante treino
            callback=[
                eval_callback,
                early_stop_callback
            ],

            reset_num_timesteps=reset_timesteps  # reinicia ou continua treino
        )

    except KeyboardInterrupt:
        # permite interromper treino manualmente sem crash
        print("\n[TRAINING] Interrupted. Saving...")

    # =========================================================
    # GUARDAR MODELO FINAL
    # =========================================================

    model.save(
        os.path.join(
            run_dir,
            "final_model"
        )
    )

    print(f"\n[TRAINING] Done.")
    print(f"[TRAINING] Model saved to: {run_dir}")


# ponto de entrada obrigatório do script
if __name__ == "__main__":
    main()