import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from controller import Supervisor
from utils import cmd_vel, warp_robot
from positions import get_positions
import numpy as np
import math
import random

# ---------------- CONFIG ----------------
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

ENV_ID = "EPuckLidarParkingEnvSAC-v0"

if ENV_ID not in gym.envs.registration.registry:
    register(
        id=ENV_ID,
        entry_point="EPuckLidarParkingEnvSAC:EPuckLidarParkingEnvSAC",
        max_episode_steps=MAX_STEPS,
    )


class EPuckLidarParkingEnvSAC(gym.Env):

    def __init__(self):
        super().__init__()

        self.supervisor = Supervisor()
        timestep = int(self.supervisor.getBasicTimeStep())

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
        # SAC ACTION SPACE (CONTINUOUS)
        # =====================================================
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # observation unchanged
        self.observation_space = spaces.Box(
            low=np.array([0]*9 + [0, -1, -1, -1], dtype=np.float32),
            high=np.array([1]*9 + [1,  1,  1,  1], dtype=np.float32),
            dtype=np.float32
        )

        self.trainmode = "all"

        self.steps = 0
        self.same_spot_steps = 0

        self.prev_distance = None
        self.prev_pos = None

        self.target_x = None
        self.target_y = 2.34

        self.supervisor.step(1)

        self.parked_nodes = {}

        for def_name in PARKED_BAY_MAP:
            node = self.supervisor.getFromDef(def_name)
            if node is not None:
                self.parked_nodes[def_name] = node

    # =========================================================
    def _toggle_bays(self):

        all_bays = list(PARKED_BAY_MAP.keys())

        free_bays = random.sample(all_bays, 2)

        for bay_name, node in self.parked_nodes.items():

            bay_x = PARKED_BAY_MAP[bay_name]
            trans = node.getField("translation")

            if bay_name in free_bays:
                trans.setSFVec3f([bay_x, 0.05, 0.0])
            else:
                trans.setSFVec3f([bay_x, 2.34, 0.0])

            node.resetPhysics()

        return free_bays

    # =========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.same_spot_steps = 0

        free_bays = self._toggle_bays()

        occupied = [
            b for b in PARKED_BAY_MAP
            if b not in free_bays
        ]

        spawn, target = get_positions(self.trainmode, occupied)
        spawn_x, spawn_y = spawn
        _, target_x = target

        self.target_x = float(target_x)
        self.target_y = 2.34

        warp_robot(self.supervisor, "EPUCK", (spawn_x, spawn_y))
        self.supervisor.step(1)

        gps = self.gps.getValues()
        self.prev_pos = np.array(gps[:2], dtype=np.float32)

        self.prev_distance = self._distance()

        return self._obs(), {}

    # =========================================================
    def step(self, action):

        self.steps += 1

        # =========================
        # SAC ACTION (CONTINUOUS)
        # =========================
        w = float(action[0]) * 1.5   # scale steering

        v = 0.08

        cmd_vel(self.supervisor, v, w)
        self.supervisor.step(200)

        obs = self._obs()
        reward, done = self._reward(obs)

        truncated = self.steps >= MAX_STEPS

        return obs, reward, done, truncated, {}

    # =========================================================
    def _obs(self):

        pc = np.array(self.lidar.getPointCloud())
        lidar = self._lidar(pc)

        gps = self.gps.getValues()
        rx, ry = gps[0], gps[1]

        dx = self.target_x - rx
        dy = self.target_y - ry

        dist = math.sqrt(dx*dx + dy*dy)
        dist_norm = min(dist / 4.0, 1.0)

        theta = self._orientation()
        angle = math.atan2(dy, dx) - theta
        angle = (angle + math.pi) % (2*math.pi) - math.pi
        angle_norm = angle / math.pi

        curr = np.array([rx, ry], dtype=np.float32)

        if self.prev_pos is None:
            vel = np.array([0.0, 0.0], dtype=np.float32)
        else:
            vel = curr - self.prev_pos

        self.prev_pos = curr

        vx = float(np.clip(vel[0] * 10.0, -1, 1))
        vy = float(np.clip(vel[1] * 10.0, -1, 1))

        return np.array(
            np.concatenate([lidar, [dist_norm, angle_norm, vx, vy]]),
            dtype=np.float32
        )

    # =========================================================
    def _reward(self, obs):

        dist = self._distance()

        if self.touch_sensor.getValue() == 1.0:
            return -100.0, True

        if dist < DIST_THRESHOLD:
            return 100.0, True

        reward = 0.0

        progress = self.prev_distance - dist
        reward += 8.0 * progress

        if progress < 0:
            reward += 6.0 * progress

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
        reward -= 0.01

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
    def _distance(self):
        gps = self.gps.getValues()
        rx, ry = gps[0], gps[1]
        return math.sqrt((self.target_x - rx)**2 + (self.target_y - ry)**2)

    def _orientation(self):
        c = self.compass.getValues()
        return math.atan2(c[0], c[1])

    def _lidar(self, pc):
        ids = [0, 11, 24, 36, 49, 61, 74, 86, 99]
        out = []

        for i in ids:
            p = pc[i]
            d = math.sqrt(p.x**2 + p.y**2)
            out.append(min(d, 2.0))

        return np.clip(np.array(out, dtype=np.float32) / 2.0, 0, 1)