import os
import gymnasium as gym
from stable_baselines3 import PPO
import EPuckLidarParkingEnv


# FUNÇÃO PRINCIPAL DE INFERÊNCIA (TESTE DO MODELO TREINADO)

def main() -> None:


    # DEFINIÇÃO DOS CAMINHOS DO MODELO TREINADO

    base_dir = os.path.join(os.getcwd(), "logs")  # pasta principal de logs
    new_run_dir = os.path.join(base_dir, "new_model")  # pasta do treino específico

    # caminho exato onde o modelo foi guardado
    model_path = os.path.join(new_run_dir, "final_model.zip")


    # VERIFICA SE O MODELO EXISTE

    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}")  # erro se não encontrar modelo
        return  # termina o programa

    # confirmação de carregamento
    print(f"Loading model from: {model_path}")


    # CRIAÇÃO DO AMBIENTE DE SIMULAÇÃO

    env = gym.make("EPuckLidarParkingEnv-v0")

    # carregamento do modelo PPO treinado
    model = PPO.load(model_path, env=env)


    # RESET INICIAL DO AMBIENTE

    obs, _info = env.reset()

    print("Inference started. Press Ctrl+C to stop.")


    # LOOP PRINCIPAL DE INFERÊNCIA (AGENTE A INTERAGIR COM O AMBIENTE)

    try:
        while True:

            # modelo escolhe ação com base na observação atual
            action, _states = model.predict(obs, deterministic=False)

            # aplica ação no ambiente e avança simulação
            obs, reward, terminated, truncated, _info = env.step(action)

            # se episódio terminou (sucesso, colisão ou timeout)
            if terminated or truncated:
                print(f"Episode Finished. Resetting... (Final Reward: {reward:.2f})")

                # reinicia ambiente para novo episódio
                obs, _info = env.reset()

    # INTERRUPÇÃO MANUAL (CTRL + C)

    except KeyboardInterrupt:
        print("\nStopping inference...")


    # LIMPEZA FINAL

    finally:
        env.close()  # fecha corretamente o ambiente



# EXECUÇÃO DO SCRIPT

if __name__ == '__main__':
    main()