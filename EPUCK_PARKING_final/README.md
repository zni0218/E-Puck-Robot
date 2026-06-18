# WebotsRL - Planeamento de Caminho e Estacionamento Autónomo com Reinforcement Learning

## Descrição

Este projeto implementa navegação autónoma e estacionamento de um robô E-Puck equipado com LiDAR no simulador Webots.

O sistema combina:
- Aprendizagem por reforço (PPO, SAC e TD3)
- Planeamento de caminho com D* Lite
- Ambiente dinâmico com lugares de estacionamento aleatórios (livres e ocupados)
- Observações baseadas em LiDAR, distância ao objetivo e ângulo relativo

O objetivo é o robô encontrar um lugar livre, navegar até ele e estacionar de forma segura, evitando colisões com paredes e outros robôs.

---

## Requisitos

- Python 3.10 ou 3.11
- Webots R2021b ou superior
- Dependências listadas em `requirements.txt`

---

## Instalação

### Criar ambiente virtual

Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Windows CMD:
python -m venv .venv
.\.venv\Scripts\activate

### Atualizar pip
python -m pip install --upgrade pip

### Instalar dependências
pip install -r requirements.txt

---

## Estrutura do projeto

project_root/
├─ EPUCK_PARKING_final/
│  ├─ dstar_parking/
│  │  ├─ d_star_lite.py
│  │  ├─ dstar_parking.py
│  │  ├─ grid.py
│  │  ├─ iri_utils.py
│  │  ├─ navigator.py
│  │  ├─ parking_map.py
│  │  ├─ parking_other_epucks.py
│  │  ├─ path_follower.py
│  │  ├─ priority_queue.py
│  │  ├─ utils.py
│  │  └─ worlds/
│  │     ├─ .Project_Dlite.wbproj
│  │     ├─ .Scenario1.jpg
│  │     ├─ .Scenario1.wbproj
│  │     ├─ Scenario1.wbt
│  │     ├─ Scenario1.png
│  │     ├─ Scenario1_config.yaml
│  │     ├─ Scenario1_points.csv
│  │     └─ base_map.wbt
│  ├─ controllers/
│  │  ├─ create_map.py
│  │  ├─ localization_utils.py
│  │  ├─ print_devices.py
│  │  ├─ transformations.py
│  │  ├─ utils.py
│  │  ├─ eval_logs/
│  │  ├─ tensorboard_logs/
│  │  └─ monitor.csv
│  ├─ logs/
│  │  ├─ new_model/
│  │  ├─ logs_sac/
│  │  └─ logs_td3/
│  ├─ models/
│  ├─ EPuckLidarParkingEnv.py
│  ├─ EPuckLidarParkingEnvSAC.py
│  ├─ inference.py
│  ├─ positions.py
│  ├─ trainingPPO.py
│  ├─ trainingSAC.py
│  ├─ trainingTD3.py
│  ├─ transformations.py
│  └─ utils.py
└─ models/

---

## Execução no Webots

Abrir o Webots

Ir a File → Open World

Selecionar:
worlds/Scenario1.wbt

Confirmar no Scene Tree que o robô usa o controller correto (dstar_parking)

Clicar em Play para iniciar a simulação

---

## Execução do controlador

O ficheiro principal é:

dstar_parking.py

Este é executado automaticamente pelo Webots como controller (Supervisor).

---

## Treino dos modelos (PPO, SAC, TD3)

Executar os scripts:

python trainingPPO.py
python trainingSAC.py
python trainingTD3.py

Notas:
- Ajustar timesteps conforme hardware
- Usar checkpoints para evitar perda de progresso
- Monitorizar com TensorBoard:

tensorboard --logdir logs/

---

## Inferência

python inference.py --model models/ppo/latest.zip --env Scenario1Env --render True

Substituir PPO por SAC ou TD3 conforme necessário.

---

## Troubleshooting

- FileNotFoundError → verificar caminhos relativos
- Webots não usa venv → configurar Preferences → Python
- Treino lento → reduzir complexidade ou acelerar simulação
- Dependências falham → instalar individualmente

---

## Boas práticas

- Usar sempre caminhos relativos (__file__)
- Fixar seeds para reprodutibilidade
- Guardar versões das dependências:

pip freeze > requirements_freeze.txt

---

## Autores

Hugo Sousa
Tiago Silva
Zhixu Ni