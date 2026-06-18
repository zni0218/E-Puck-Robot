"""
dstar_parking.py  --  CONTROLLER PRINCIPAL
==========================================

Este ficheiro é o "orquestrador" do sistema de estacionamento.

Ele junta várias partes:
- mapa do estacionamento (imagem + grelha)
- planeamento de caminho (D* Lite)
- controlo do robô (seguir caminho + estacionar)
- calibração da bússola
- entrada final no lugar de estacionamento

A ideia geral:
1) encontrar lugar livre
2) planear até à entrada do lugar
3) alinhar o robô
4) entrar em linha reta até ao fundo
"""

import os
import math
from controller import Supervisor

from parking_map import ParkingMap
from navigator import DStarNavigator
from path_follower import PathFollower
from iri_utils import cmd_vel, stop, gps_xy, raw_compass
from parking_other_epucks import parked_other_cars


# caminho base para ficheiros do mapa
HERE = os.path.dirname(__file__)
PATH = os.path.join(HERE, "worlds")

# parâmetros geométricos do robô e estacionamento
ROBOT_RADIUS = 0.037
ENTRANCE_GAP = 0.25


def _ang_diff(a, b):
    # calcula diferença angular normalizada entre dois ângulos
    # resultado está sempre entre [-pi, pi]
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def calibrate_heading_sign(supervisor, compass, dt):
    """
    A bússola pode estar invertida dependendo do mundo/simulação.

    Esta função:
    - roda o robô para um lado
    - roda para o outro lado
    - compara leitura da bússola
    - descobre automaticamente o sinal correto (+1 ou -1)
    """

    h0 = raw_compass(compass)

    # rotação para testar direção positiva
    for _ in range(20):
        cmd_vel(supervisor, 0.0, 1.0)
        supervisor.step(dt)

    h1 = raw_compass(compass)

    # rotação para testar direção negativa
    for _ in range(20):
        cmd_vel(supervisor, 0.0, -1.0)
        supervisor.step(dt)

    stop(supervisor)

    # determina se bússola está invertida ou não
    sign = 1.0 if _ang_diff(h1, h0) > 0 else -1.0

    print(f"[calib] sinal da bússola = {sign:+.0f}")
    return sign


def subsample(path, every=3):
    """
    Reduz número de waypoints do caminho.

    Isto simplifica o controlo:
    - menos pontos = movimento mais suave
    - evita micro-ajustes desnecessários
    """

    if not path:
        return path

    pts = path[::every]

    # garante que o último ponto nunca é perdido
    if pts[-1] != path[-1]:
        pts.append(path[-1])

    return pts


def main():
    """
    PIPELINE PRINCIPAL DO SISTEMA DE PARKING
    """

    supervisor = Supervisor()
    dt = int(supervisor.getBasicTimeStep())

    # =====================================================
    # 1) definir ambiente inicial e escolher lugar livre
    # =====================================================
    _img_shape, goal, _slots = parked_other_cars(supervisor)

    if goal is None:
        print("[D*Lite] ERRO: parked_other_cars não devolveu goal.")
        return

    goal_xy = (goal[0], goal[1])
    print(f"[D*Lite] lugar livre (goal) = {goal_xy}")

    # sensores base do robô
    gps = supervisor.getDevice("gps")
    gps.enable(dt)

    compass = supervisor.getDevice("compass")
    compass.enable(dt)

    supervisor.step(dt)

    # =====================================================
    # 2) calibração da bússola
    # =====================================================
    sign = calibrate_heading_sign(supervisor, compass, dt)

    # função local que já devolve heading corrigido
    def heading():
        return sign * raw_compass(compass)

    # =====================================================
    # 3) criação do mapa de estacionamento
    # =====================================================
    pm = ParkingMap(
        os.path.join(PATH, "Scenario1.png"),
        os.path.join(PATH, "Scenario1_config.yaml"),
        cell_size=0.02,
        inflate_m=0.07
    )

    # calcula orientação do lugar de estacionamento (direção de entrada)
    inward, back = pm.slot_geometry(*goal_xy)
    ix, iy = inward
    inward_heading = math.atan2(iy, ix)

    print(
        f"[D*Lite] entrar na direção {inward} "
        f"(heading {math.degrees(inward_heading):.0f}°), "
        f"fundo a {back*100:.1f}cm"
    )

    # ponto de entrada (antes do lugar real)
    ex = goal_xy[0] - ix * ENTRANCE_GAP
    ey = goal_xy[1] - iy * ENTRANCE_GAP

    entrance_cell = pm.world_to_cell(ex, ey)

    # ajusta entrada caso esteja em zona ocupada
    step_out = 0
    while not pm.is_free_cell(entrance_cell) and step_out < 20:

        step_out += 1

        ex = goal_xy[0] - ix * (ENTRANCE_GAP + 0.02 * step_out)
        ey = goal_xy[1] - iy * (ENTRANCE_GAP + 0.02 * step_out)

        entrance_cell = pm.world_to_cell(ex, ey)

    # posição inicial do robô
    x, y = gps_xy(gps)
    start_cell = pm.world_to_cell(x, y)

    print(f"[D*Lite] start_cell={start_cell}  entrance_cell={entrance_cell}")

    # =====================================================
    # 4) planeamento D* Lite até à entrada
    # =====================================================
    nav = DStarNavigator(pm, start_cell, entrance_cell)
    path = subsample(nav.world_waypoints(), every=3)

    print(f"[D*Lite] caminho até à entrada: {len(path)} waypoints")

    # controlador de trajetória
    follower = PathFollower(max_lin=0.10, max_ang=2.5)

    # seguir cada waypoint do caminho
    for wp in path:
        while supervisor.step(dt) != -1:

            x, y = gps_xy(gps)

            v, w, arrived = follower.control((x, y, heading()), wp)

            cmd_vel(supervisor, v, w)

            if arrived:
                break

    # =====================================================
    # 5) alinhar robô com direção do lugar
    # =====================================================
    while supervisor.step(dt) != -1:

        if abs(_ang_diff(inward_heading, heading())) < 0.02:
            break

        cmd_vel(
            supervisor,
            0.0,
            max(-2.5, min(2.5,
                2.0 * _ang_diff(inward_heading, heading())
            ))
        )

    # =====================================================
    # 6) entrada final no lugar (movimento reto)
    # =====================================================

    margin = ROBOT_RADIUS + 0.05

    tx = goal_xy[0] + ix * (back - margin)
    ty = goal_xy[1] + iy * (back - margin)

    # projeta posição ao longo do eixo do lugar
    target_in = tx * ix + ty * iy

    # avança em linha reta até ao fundo
    while supervisor.step(dt) != -1:

        x, y = gps_xy(gps)

        pos_in = x * ix + y * iy

        w = max(-1.0, min(1.0,
            2.0 * _ang_diff(inward_heading, heading())
        ))

        if pos_in < target_in:
            cmd_vel(supervisor, 0.04, w)
        else:
            break

    stop(supervisor)

    print("[D*Lite] Estacionado, dentro do lugar e orientado para o fundo.")


if __name__ == "__main__":
    main()