import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from controller import Supervisor, TouchSensor, GPS, Compass
from utils import cmd_vel, warp_robot
from positions import get_positions
import numpy as np
import math
import random

# =========================================================
# Configuração geral do ambiente
# =========================================================

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

ALL_BAYS = list(PARKED_BAY_MAP.items())
DIST_THRESHOLD = 0.1
MAX_STEPS = 800

ENV_ID = "EPuckLidarParkingEnv-v0"

if ENV_ID not in gym.envs.registration.registry:
    register(
        id=ENV_ID,
        entry_point="EPuckLidarParkingEnv:EPuckLidarParkingEnv",
        max_episode_steps=MAX_STEPS,
    )


class EPuckLidarParkingEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # =====================================================
        # Inicialização do Webots
        # =====================================================
        self.supervisor = Supervisor()
        timestep = int(self.supervisor.getBasicTimeStep())

        # =====================================================
        # Sensores do robô
        # =====================================================
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enablePointCloud()
        self.lidar.enable(timestep)

        self.touch_sensor = self.supervisor.getDevice("touch sensor")
        self.touch_sensor.enable(timestep)

        self.gps = self.supervisor.getDevice("gps")
        self.gps.enable(timestep)

        self.compass = self.supervisor.getDevice("compass")
        self.compass.enable(timestep)

        # =====================================================
        # Ação e observação
        # =====================================================
        self.action_space = spaces.Discrete(3)

        # 9 valores do lidar + distância + ângulo + velocidade x/y
        self.observation_space = spaces.Box(
            low=np.array([0]*9 + [0, -1, -1, -1], dtype=np.float32),
            high=np.array([1]*9 + [1,  1,  1,  1], dtype=np.float32),
            dtype=np.float32
        )

        # =====================================================
        # Estado interno do episódio
        # =====================================================
        self.trainmode = "all"
        self.steps = 0
        self.same_spot_steps = 0

        self.prev_distance = None
        self.prev_pos = None

        self.target_x = None
        self.target_y = 2.34

        # Primeiro passo para a simulação estabilizar
        self.supervisor.step(1)

        # Guardar os nós dos EPucks estacionados
        self.parked_nodes = {}
        for def_name in PARKED_BAY_MAP:
            node = self.supervisor.getFromDef(def_name)
            if node is not None:
                self.parked_nodes[def_name] = node
                print(f"[INIT] Found {def_name}")
            else:
                print(f"[INIT] Missing {def_name}")

    # =========================================================
    # Escolhe 2 bays livres e move os restantes EPucks
    # =========================================================
    def _toggle_bays(self):
        all_bays = list(PARKED_BAY_MAP.keys())

        # Duas bays ficam livres em cada episódio
        free_bays = random.sample(all_bays, 2)

        for bay_name, node in self.parked_nodes.items():
            bay_x = PARKED_BAY_MAP[bay_name]
            trans = node.getField("translation")

            if bay_name in free_bays:
                # Bay livre: o robô é movido para fora da zona útil
                trans.setSFVec3f([bay_x, 0.05, 0.0])
                print(f"[FREE] {bay_name}")
            else:
                # Bay ocupada: mantém o robô estacionado no lugar
                trans.setSFVec3f([bay_x, 2.34, 0.0])
                print(f"[OCCUPIED] {bay_name}")

            node.resetPhysics()

        return free_bays

    # =========================================================
    # Reset do episódio
    # =========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.same_spot_steps = 0

        # Define quais bays ficam livres neste episódio
        free_bays = self._toggle_bays()

        # Todas as outras são consideradas ocupadas
        occupied = [
            bay
            for bay in PARKED_BAY_MAP
            if bay not in free_bays
        ]

        # Escolhe spawn e target com base nas bays ocupadas
        spawn, target = get_positions(
            self.trainmode,
            occupied
        )
        spawn_x, spawn_y = spawn
        _, target_x = target

        self.target_x = float(target_x)
        self.target_y = 2.34

        # Reposiciona o robô no spawn inicial do episódio
        warp_robot(self.supervisor, "EPUCK", (spawn_x, spawn_y))
        self.supervisor.step(1)

        # Guarda posição inicial para calcular movimento
        gps = self.gps.getValues()
        self.prev_pos = np.array(gps[:2], dtype=np.float32)

        # Distância inicial ao target
        self.prev_distance = self._distance()

        return self._obs(), {}

    # =========================================================
    # Passo do ambiente
    # =========================================================
    def step(self, action):
        self.steps += 1

        # Ações discretas: frente, esquerda, direita
        if action == 0:
            v, w = 0.10, 0.0
        elif action == 1:
            v, w = 0.08, 0.8
        elif action == 2:
            v, w = 0.08, -0.8

        # Envia o comando ao robô
        cmd_vel(self.supervisor, v, w)

        # Avança a simulação
        self.supervisor.step(200)

        # Lê o novo estado
        obs = self._obs()

        # Calcula recompensa e se terminou por sucesso/colisão
        reward, done = self._reward(obs)

        # Termina por limite de passos
        truncated = self.steps >= MAX_STEPS

        return obs, reward, done, truncated, {}

    # =========================================================
    # Construção da observação
    # =========================================================
    def _obs(self):
        # Lê o lidar em nuvem de pontos
        pc = np.array(self.lidar.getPointCloud())
        lidar = self._lidar(pc)

        # Posição atual do robô
        gps = self.gps.getValues()
        rx, ry = gps[0], gps[1]

        dx = self.target_x - rx
        dy = self.target_y - ry

        # Distância até ao target normalizada
        dist = math.sqrt(dx*dx + dy*dy)
        dist_norm = min(dist / 4.0, 1.0)

        # Ângulo relativo ao target normalizado para [-1, 1]
        theta = self._orientation()
        angle = math.atan2(dy, dx) - theta
        angle = (angle + math.pi) % (2*math.pi) - math.pi
        angle_norm = angle / math.pi

        # Estimação da deslocação entre passos
        curr = np.array([rx, ry], dtype=np.float32)

        if self.prev_pos is None:
            vel = np.array([0.0, 0.0], dtype=np.float32)
        else:
            vel = curr - self.prev_pos

        self.prev_pos = curr

        vx = float(np.clip(vel[0] * 10.0, -1, 1))
        vy = float(np.clip(vel[1] * 10.0, -1, 1))

        # Junta tudo num único vetor de observação
        return np.array(
            np.concatenate([lidar, [dist_norm, angle_norm, vx, vy]]),
            dtype=np.float32
        )

    # =========================================================
    # Recompensa
    # =========================================================
    def _reward(self, obs):
        dist = self._distance()

        # Colisão: termina logo com penalização forte
        if self.touch_sensor.getValue() == 1.0:
            return -100.0, True

        # Sucesso: chegou suficientemente perto do target
        if dist < DIST_THRESHOLD:
            return 100.0, True

        reward = 0.0

        # Recompensa principal: aproximar-se do target
        progress = self.prev_distance - dist
        reward += 8.0 * progress

        # Penaliza mais se estiver a afastar-se
        if progress < 0:
            reward += 6.0 * progress

        # Penalização por orientação incorreta
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

        # Pequena penalização por passo, para incentivar rapidez
        reward -= 0.01

        # Anti-stuck: se quase não progride durante vários passos, penaliza
        if abs(progress) < 0.001:
            self.same_spot_steps += 1
        else:
            self.same_spot_steps = 0

        if self.same_spot_steps > 8:
            reward -= 40
            self.same_spot_steps = 0

        self.prev_distance = dist

        return reward, False

    # =========================================================
    # Helpers geométricos
    # =========================================================
    def _distance(self):
        gps = self.gps.getValues()
        rx, ry = gps[0], gps[1]
        return math.sqrt((self.target_x - rx)**2 + (self.target_y - ry)**2)

    def _orientation(self):
        c = self.compass.getValues()
        return math.atan2(c[0], c[1])

    def _lidar(self, pc):
        # Seleciona 9 pontos do lidar e reduz para um vetor pequeno
        ids = [0, 11, 24, 36, 49, 61, 74, 86, 99]
        out = []

        for i in ids:
            p = pc[i]
            d = math.sqrt(p.x**2 + p.y**2)
            out.append(min(d, 2.0))

        out = np.array(out, dtype=np.float32)
        return np.clip(out / 2.0, 0, 1)