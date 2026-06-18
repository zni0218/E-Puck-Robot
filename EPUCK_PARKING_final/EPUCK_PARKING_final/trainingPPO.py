import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import EPuckLidarParkingEnv


# Callback personalizado para parar o treino caso uma condição interna do ambiente seja ativada
class EarlyStopCallback(BaseCallback):

    def __init__(self, env: gym.Env, verbose: int = 1):
        super().__init__(verbose)
        self.env = env  # guardamos referência ao ambiente para ler flags internas (ex: should_stop)

    def _on_step(self) -> bool:
        # a cada passo de treino, verificamos se o ambiente pediu paragem
        if getattr(self.env, "should_stop", False):

            # se verbose ativo, mostramos mensagem no terminal
            if self.verbose:
                print("\n[EarlyStopCallback] Early stop triggered.")

            # retorna False para parar o treino imediatamente
            return False

        # continua o treino normalmente
        return True


def main() -> None:
    # função principal responsável por iniciar ou continuar o treino do modelo PPO

    # definição das pastas onde os logs e modelos são guardados
    base_dir: str = os.path.join(os.getcwd(), "logs")  # pasta base de logs
    new_run_dir: str = os.path.join(base_dir, "new_model")  # pasta da execução atual

    # garante que a pasta existe (evita erros ao guardar ficheiros)
    os.makedirs(new_run_dir, exist_ok=True)

    # criação do ambiente de simulação (Webots + Gymnasium)
    raw_env = gym.make("EPuckLidarParkingEnv-v0")

    # wrapper Monitor: regista recompensas, episódios e estatísticas
    env = Monitor(raw_env, filename=os.path.join(new_run_dir, "monitor.csv"))

    # caminhos possíveis para modelos já treinados
    new_checkpoint = os.path.join(new_run_dir, "final_model.zip")  # modelo mais recente
    old_checkpoint = os.path.join(base_dir, "final_model.zip")  # fallback para modelo antigo

    # =========================================================
    # LÓGICA DE CARREGAMENTO DO MODELO
    # =========================================================

    if os.path.exists(new_checkpoint):
        # caso exista modelo recente, continua treino a partir dele
        print(f"\n[TRAINING] RESUMING new model from {new_checkpoint}")

        model = PPO.load(new_checkpoint, env=env)

        # ajusta parâmetros importantes após load (exploração)
        model.ent_coef = 0.01
        model.learning_rate = 3e-4

        reset_timesteps = False

    elif os.path.exists(old_checkpoint):
        # caso não exista novo, mas exista antigo, usa esse
        print(f"\n[TRAINING] LOADING old model from {old_checkpoint}")

        model = PPO.load(old_checkpoint, env=env)

        model.ent_coef = 0.01
        model.learning_rate = 3e-4

        reset_timesteps = False

    else:
        # caso não exista nenhum modelo, começa treino do zero
        print("\n[TRAINING] FRESH START")

        model = PPO(
            "MlpPolicy",  # rede neural totalmente ligada (MLP)
            env,  # ambiente de treino

            verbose=1,  # mostra logs durante treino

            # parâmetros de exploração e aprendizagem
            ent_coef=0.01,  # incentiva exploração
            learning_rate=3e-4,

            # parâmetros de otimização PPO
            n_steps=1024,  # passos por rollout (mais pequeno = mais responsivo)
            batch_size=64,  # tamanho dos mini-batches
            n_epochs=10,  # número de passes por batch
            gamma=0.99,  # fator de desconto

            # arquitetura da rede neural (2 camadas de 64 neurónios)
            policy_kwargs=dict(net_arch=[64, 64])
        )

        reset_timesteps = True

    # =========================================================
    # CALLBACK DE AVALIAÇÃO
    # =========================================================

    eval_callback = EvalCallback(
        env,  # ambiente usado para avaliação

        n_eval_episodes=5,  # número de episódios de avaliação
        eval_freq=5000,  # frequência de avaliação (a cada X passos)

        best_model_save_path=os.path.join(new_run_dir, "best_model"),  # guarda melhor modelo
        log_path=os.path.join(new_run_dir, "eval_logs"),  # logs de avaliação

        deterministic=True,  # ações determinísticas na avaliação
        render=False  # não renderizar durante avaliação
    )

    # =========================================================
    # CALLBACK DE EARLY STOP
    # =========================================================

    unwrapped_env = raw_env.unwrapped  # acesso direto ao ambiente sem wrappers
    early_stop_callback = EarlyStopCallback(unwrapped_env, verbose=1)

    # timestamp usado para logs no tensorboard
    time_str = time.strftime("%Y%m%d-%H%M%S")

    # =========================================================
    # LOOP DE TREINO PRINCIPAL
    # =========================================================

    try:
        model.learn(
            total_timesteps=900_000,  # número total de passos de treino
            log_interval=2,  # frequência de logs
            tb_log_name=time_str,  # nome dos logs no tensorboard

            callback=[eval_callback, early_stop_callback],  # callbacks ativos

            reset_num_timesteps=reset_timesteps  # reinicia ou continua contagem de steps
        )

    except KeyboardInterrupt:
        # permite parar treino manualmente sem perder progresso
        print("\n[TRAINING] Interrupted. Saving...")

    # =========================================================
    # GUARDA FINAL DO MODELO
    # =========================================================

    model.save(os.path.join(new_run_dir, "final_model"))

    print(f"[TRAINING] Done. Model saved to {new_run_dir}/")


# ponto de entrada do programa
if __name__ == '__main__':
    main()