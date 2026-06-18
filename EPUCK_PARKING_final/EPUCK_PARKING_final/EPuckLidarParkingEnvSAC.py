import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from controller import Supervisor  # interface principal do Webots para controlar o mundo
from utils import cmd_vel, warp_robot  # funções auxiliares para mover o robô e reposicionar
from positions import get_positions  # função que escolhe spawn e target no cenário
import numpy as np
import math
import random


# CONFIGURAÇÃO DO AMBIENTE
# -------------------------
# Aqui definimos como o "mundo" do problema está estruturado

PARKED_BAY_MAP = {
    "PARKED_0": 0.74,
    "PARKED_1": 1.19,
    "PARKED_2": 0.97,
    "PARKED_3": 0.51,
    "PARKED_4": 0.28,
    "PARKED_5": 1.65,
    "PARKED_6": 1.88,
    "PARKED_7": 1.42,
}

# lista auxiliar com todas as bays
ALL_BAYS = list(PARKED_BAY_MAP.items())

# distância mínima para considerar que chegou ao objetivo
DIST_THRESHOLD = 0.1

# limite máximo de passos por episódio (evita loops infinitos)
MAX_STEPS = 800


# REGISTO DO AMBIENTE NO GYMNASIUM
# ---------------------------------
ENV_ID = "EPuckLidarParkingEnvSAC-v0"

# evita registar duas vezes o mesmo ambiente
if ENV_ID not in gym.envs.registration.registry:
    register(
        id=ENV_ID,
        entry_point="EPuckLidarParkingEnvSAC:EPuckLidarParkingEnvSAC",
        max_episode_steps=MAX_STEPS,
    )


class EPuckLidarParkingEnvSAC(gym.Env):

    def __init__(self):
        super().__init__()

        # INICIALIZAÇÃO DO SIMULADOR (WEBOTS)
        # ------------------------------------
        # cria o supervisor que controla o robô e o ambiente
        self.supervisor = Supervisor()
        timestep = int(self.supervisor.getBasicTimeStep())

        # SENSORES DO ROBÔ
        # -----------------
        # LIDAR: perceção do ambiente à volta do robô
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enablePointCloud()
        self.lidar.enable(timestep)

        # sensor de colisão (toque físico)
        self.touch_sensor = self.supervisor.getDevice("touch sensor")
        self.touch_sensor.enable(timestep)

        # GPS: posição do robô no mundo
        self.gps = self.supervisor.getDevice("gps")
        self.gps.enable(timestep)

        # bússola: orientação do robô
        self.compass = self.supervisor.getDevice("compass")
        self.compass.enable(timestep)

        # ESPAÇO DE AÇÕES (SAC = CONTÍNUO)
        # ---------------------------------
        # o agente controla apenas rotação (ângulo), enquanto velocidade é fixa
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # ESPAÇO DE OBSERVAÇÃO
        # ---------------------
        # 9 valores do LIDAR + distância ao alvo + ângulo + velocidade estimada (x,y)
        self.observation_space = spaces.Box(
            low=np.array([0]*9 + [0, -1, -1, -1], dtype=np.float32),
            high=np.array([1]*9 + [1,  1,  1,  1], dtype=np.float32),
            dtype=np.float32
        )

        # ESTADO INTERNO DO EPISÓDIO
        # --------------------------
        self.trainmode = "all"  # modo de treino (define distribuição de spawns/targets)
        self.steps = 0  # contador de passos no episódio
        self.same_spot_steps = 0  # deteta se o agente está parado/estagnado

        self.prev_distance = None  # distância anterior ao target (para reward de progresso)
        self.prev_pos = None  # posição anterior (para estimar movimento)

        self.target_x = None  # posição x do objetivo
        self.target_y = 2.34  # posição y fixa (linha das bays)

        # dá um passo inicial para estabilizar o simulador
        self.supervisor.step(1)

        # guarda referências aos carros estacionados no cenário
        self.parked_nodes = {}

        # recolhe todos os nós definidos no mundo (PARKED_0, PARKED_1, etc.)
        for def_name in PARKED_BAY_MAP:
            node = self.supervisor.getFromDef(def_name)
            if node is not None:
                self.parked_nodes[def_name] = node


    # ESCOLHA DINÂMICA DAS BAYS
    # -------------------------
    # simula estacionamento aleatório: algumas vagas ficam livres
    def _toggle_bays(self):

        all_bays = list(PARKED_BAY_MAP.keys())

        # escolhe aleatoriamente 2 lugares livres
        free_bays = random.sample(all_bays, 2)

        for bay_name, node in self.parked_nodes.items():

            bay_x = PARKED_BAY_MAP[bay_name]
            trans = node.getField("translation")

            # se a bay estiver livre, "remove" o carro
            if bay_name in free_bays:
                trans.setSFVec3f([bay_x, 0.05, 0.0])
            else:
                # caso contrário mantém o carro estacionado
                trans.setSFVec3f([bay_x, 2.34, 0.0])

            # reinicia física para evitar glitches de simulação
            node.resetPhysics()

        return free_bays


    # RESET DO EPISÓDIO
    # -----------------
    # prepara um novo cenário para treino do agente
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.same_spot_steps = 0

        # define quais vagas estão livres neste episódio
        free_bays = self._toggle_bays()

        # todas as outras são consideradas ocupadas
        occupied = [
            b for b in PARKED_BAY_MAP
            if b not in free_bays
        ]

        # escolhe posição inicial e objetivo com base no cenário atual
        spawn, target = get_positions(self.trainmode, occupied)
        spawn_x, spawn_y = spawn
        _, target_x = target

        self.target_x = float(target_x)
        self.target_y = 2.34

        # coloca o robô na posição inicial
        warp_robot(self.supervisor, "EPUCK", (spawn_x, spawn_y))
        self.supervisor.step(1)

        # guarda posição inicial para calcular movimento
        gps = self.gps.getValues()
        self.prev_pos = np.array(gps[:2], dtype=np.float32)

        # calcula distância inicial ao objetivo
        self.prev_distance = self._distance()

        return self._obs(), {}


    # PASSO DO AMBIENTE
    # -----------------
    # executa uma ação e avança a simulação
    def step(self, action):

        self.steps += 1

        # ação contínua: valor entre -1 e 1 controla rotação
        w = float(action[0]) * 1.5
        v = 0.08  # velocidade linear constante

        cmd_vel(self.supervisor, v, w)

        # avança simulação no Webots
        self.supervisor.step(200)

        # obtém novo estado
        obs = self._obs()

        # calcula recompensa e se terminou episódio
        reward, done = self._reward(obs)

        # termina se atingir limite de passos
        truncated = self.steps >= MAX_STEPS

        return obs, reward, done, truncated, {}


    # OBSERVAÇÃO DO ESTADO
    # --------------------
    # transforma sensores em vetor para a rede neural
    def _obs(self):

        pc = np.array(self.lidar.getPointCloud())
        lidar = self._lidar(pc)

        gps = self.gps.getValues()
        rx, ry = gps[0], gps[1]

        dx = self.target_x - rx
        dy = self.target_y - ry

        # distância ao objetivo (normalizada)
        dist = math.sqrt(dx*dx + dy*dy)
        dist_norm = min(dist / 4.0, 1.0)

        # ângulo relativo ao objetivo
        theta = self._orientation()
        angle = math.atan2(dy, dx) - theta
        angle = (angle + math.pi) % (2*math.pi) - math.pi
        angle_norm = angle / math.pi

        # estima velocidade baseada no movimento entre passos
        curr = np.array([rx, ry], dtype=np.float32)

        if self.prev_pos is None:
            vel = np.array([0.0, 0.0], dtype=np.float32)
        else:
            vel = curr - self.prev_pos

        self.prev_pos = curr

        vx = float(np.clip(vel[0] * 10.0, -1, 1))
        vy = float(np.clip(vel[1] * 10.0, -1, 1))

        # junta tudo num único vetor
        return np.array(
            np.concatenate([lidar, [dist_norm, angle_norm, vx, vy]]),
            dtype=np.float32
        )


    # FUNÇÃO DE RECOMPENSA
    # --------------------
    # define o comportamento que o agente deve aprender
    def _reward(self, obs):

        dist = self._distance()

        # colisão: penalização forte e fim do episódio
        if self.touch_sensor.getValue() == 1.0:
            return -100.0, True

        # sucesso: chegou ao objetivo
        if dist < DIST_THRESHOLD:
            return 100.0, True

        reward = 0.0

        # recompensa por aproximação ao alvo
        progress = self.prev_distance - dist
        reward += 8.0 * progress

        # penaliza afastamento
        if progress < 0:
            reward += 6.0 * progress

        # penalização por direção errada
        gps = self.gps.getValues()
        rx, ry = gps[0], gps[1]

        dx = self.target_x - rx
        dy = self.target_y - ry

        desired_angle = math.atan2(dy, dx)
        robot_angle = self._orientation()

        angle_error = abs(
            (desired_angle - robot_angle + math.pi) % (2*math.pi) - math.pi
        )

        reward -= 1.5 * angle_error

        # penalização por demorar muito
        reward -= 0.01

        # deteta se o agente está preso no mesmo local
        if abs(progress) < 0.001:
            self.same_spot_steps += 1
        else:
            self.same_spot_steps = 0

        if self.same_spot_steps > 8:
            reward -= 40
            self.same_spot_steps = 0

        self.prev_distance = dist

        return reward, False


    # FUNÇÕES AUXILIARES
    # ------------------
    def _distance(self):
        gps = self.gps.getValues()
        rx, ry = gps[0], gps[1]
        return math.sqrt((self.target_x - rx)**2 + (self.target_y - ry)**2)

    def _orientation(self):
        c = self.compass.getValues()
        return math.atan2(c[0], c[1])

    def _lidar(self, pc):
        # reduz nuvem de pontos do lidar para 9 medições úteis
        ids = [0, 11, 24, 36, 49, 61, 74, 86, 99]

        out = []
        for i in ids:
            p = pc[i]
            d = math.sqrt(p.x**2 + p.y**2)
            out.append(min(d, 2.0))

        return np.clip(np.array(out, dtype=np.float32) / 2.0, 0, 1)