import os
import time
import gymnasium as gym

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback
)

import EPuckLidarParkingEnvSAC


# CALLBACK PARA PARAR O TREINO AUTOMATICAMENTE
# --------------------------------------------
# Este callback permite interromper o treino caso o ambiente
# defina uma variável interna "should_stop = True"
class EarlyStopCallback(BaseCallback):

    def __init__(self, env, verbose=1):
        super().__init__(verbose)
        self.env = env  # referência ao ambiente (para ler flags internas)

    def _on_step(self):

        # chamado a cada passo do algoritmo de treino
        if getattr(self.env, "should_stop", False):

            if self.verbose:
                print("\n[EarlyStopCallback] Early stop triggered.")

            # False = interrompe imediatamente o treino
            return False

        return True


# FUNÇÃO PRINCIPAL DO TREINO TD3
# -------------------------------
def main():

    # diretório base onde serão guardados logs e modelos
    base_dir = os.path.join(
        os.getcwd(),
        "logs_td3"
    )

    # pasta específica desta execução de treino
    run_dir = os.path.join(
        base_dir,
        "parking_td3"
    )

    # cria a pasta caso não exista
    os.makedirs(run_dir, exist_ok=True)

    # criação do ambiente Gym + Webots
    raw_env = gym.make(
        "EPuckLidarParkingEnvSAC-v0"
    )

    # wrapper que regista estatísticas (rewards, episódios, etc.)
    env = Monitor(
        raw_env,
        filename=os.path.join(
            run_dir,
            "monitor.csv"
        )
    )

    # caminho do modelo guardado (checkpoint final)
    checkpoint = os.path.join(
        run_dir,
        "final_model.zip"
    )

    # =========================================================
    # CARREGAMENTO OU INICIALIZAÇÃO DO MODELO TD3
    # =========================================================

    if os.path.exists(checkpoint):

        # se já existe modelo treinado, continua o treino
        print(
            f"\n[TRAINING] RESUMING TD3 model "
            f"from {checkpoint}"
        )

        model = TD3.load(
            checkpoint,
            env=env
        )

        reset_timesteps = False  # continua contagem anterior

    else:

        # se não existe modelo, começa treino do zero
        print(
            "\n[TRAINING] FRESH START (TD3)"
        )

        model = TD3(
            "MlpPolicy",  # rede neural tipo MLP (fully connected)

            env,  # ambiente de treino

            verbose=1,  # logs no terminal

            # logs para TensorBoard (visualização de treino)
            tensorboard_log=os.path.join(
                run_dir,
                "tensorboard_logs"
            ),

            # hiperparâmetros principais do TD3
            learning_rate=3e-4,  # taxa de aprendizagem
            buffer_size=300_000,  # replay buffer
            batch_size=256,  # tamanho do batch de treino

            gamma=0.99,  # desconto de recompensas futuras
            tau=0.02,  # atualização suave das redes alvo

            train_freq=1,  # frequência de treino
            gradient_steps=1,  # updates por interação

            # arquitetura da rede neural (2 camadas de 256 neurónios)
            policy_kwargs=dict(
                net_arch=[256, 256]
            )
        )

        reset_timesteps = True  # inicia treino do zero

    # =========================================================
    # CALLBACK DE AVALIAÇÃO
    # =========================================================

    eval_callback = EvalCallback(
        env,

        n_eval_episodes=5,  # episódios usados para avaliar performance

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
        render=False  # sem renderização (mais rápido)
    )

    # versão "limpa" do ambiente para aceder a variáveis internas
    early_stop_callback = EarlyStopCallback(
        raw_env.unwrapped,
        verbose=1
    )

    # timestamp para identificar esta execução no TensorBoard
    time_str = time.strftime(
        "%Y%m%d-%H%M%S"
    )

    # =========================================================
    # LOOP DE TREINO PRINCIPAL
    # =========================================================

    try:

        model.learn(
            total_timesteps=900_000,  # número total de passos de treino

            log_interval=10,  # frequência de logs no terminal

            tb_log_name=time_str,  # nome do run no TensorBoard

            # callbacks usados durante o treino
            callback=[
                eval_callback,
                early_stop_callback
            ],

            reset_num_timesteps=reset_timesteps  # continua ou reinicia contagem
        )

    except KeyboardInterrupt:

        # permite parar treino manualmente sem crash
        print(
            "\n[TRAINING] Interrupted. Saving..."
        )

    # =========================================================
    # GUARDAR MODELO FINAL
    # =========================================================

    model.save(
        os.path.join(
            run_dir,
            "final_model"
        )
    )

    print(
        f"\n[TRAINING] TD3 saved to {run_dir}"
    )


# ponto de entrada do script
if __name__ == "__main__":
    main()