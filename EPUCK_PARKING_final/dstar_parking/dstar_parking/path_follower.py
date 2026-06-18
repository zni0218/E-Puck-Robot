"""
path_follower.py  --  seguir o caminho do D* Lite com controlo contínuo
========================================================================
Responde à tua Q de treino "discreto vs contínuo":
  - O D* Lite dá um caminho DISCRETO (lista de células / waypoints).
  - Este seguidor converte-o em comandos CONTÍNUOS (v, w) -> as tuas "linhas guia".
  - Não há aprendizagem aqui: é um controlador proporcional simples (tipo pure-pursuit
    leve). É a "cola" entre o planeador e os motores.
"""

import math

def _ang_diff(a, b):
    # devolve a diferença angular entre a e b no intervalo (-pi, +pi]
    # a, b: ângulos em radianos
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class PathFollower:
    def __init__(self, max_lin=0.10, max_ang=2.5,
                 wp_tol=0.04, k_ang=3.0, k_lin=1.2, slow_angle=0.5):
        # max_lin: velocidade linear máxima (m/s)
        # max_ang: velocidade angular máxima (rad/s)
        # wp_tol: tolerância para considerar waypoint atingido (m)
        # k_ang: ganho proporcional para controlo angular (rad/s por rad)
        # k_lin: ganho proporcional para controlo linear (m/s por m)
        # slow_angle: se |erro angular| > slow_angle, o robô gira no sítio (rad)
        self.max_lin = max_lin            # guarda velocidade linear máxima
        self.max_ang = max_ang            # guarda velocidade angular máxima
        self.wp_tol = wp_tol              # guarda tolerância de waypoint
        self.k_ang = k_ang                # guarda ganho angular
        self.k_lin = k_lin                # guarda ganho linear
        self.slow_angle = slow_angle      # guarda limiar para andar vs rodar

    def control(self, pose, target_xy):
        # pose: tupla (x, y, theta) com posição e heading atual (metros, metros, rad)
        # target_xy: tupla (tx, ty) com coordenadas do waypoint alvo (metros, metros)
        # devolve: (v, w, chegou)
        #   v: velocidade linear a comandar (m/s)
        #   w: velocidade angular a comandar (rad/s)
        #   chegou: booleano indicando se o waypoint foi atingido

        x, y, theta = pose                 # extrai posição e orientação atuais
        tx, ty = target_xy                 # extrai coordenadas do alvo
        dx, dy = tx - x, ty - y            # diferença vetorial entre alvo e pose
        dist = math.hypot(dx, dy)          # distância euclidiana até ao alvo

        if dist < self.wp_tol:             # se estamos suficientemente perto do waypoint
            return 0.0, 0.0, True          # parar e sinalizar "chegou"

        desired = math.atan2(dy, dx)       # ângulo desejado para apontar ao alvo
        err = _ang_diff(desired, theta)    # erro angular (desejado - atual) normalizado

        # controlo angular proporcional com saturação nos limites [-max_ang, max_ang]
        w = max(-self.max_ang, min(self.max_ang, self.k_ang * err))

        # se o erro angular for grande, roda no sítio (v = 0) para alinhar primeiro
        if abs(err) > self.slow_angle:
            v = 0.0                        # não andar enquanto o ângulo está muito errado
        else:
            # controlo linear proporcional à distância com saturação [0, max_lin]
            # evita velocidades negativas (não recua)
            v = max(0.0, min(self.max_lin, self.k_lin * dist))

        return v, w, False                  # devolve comandos e "não chegou"

    def align_heading(self, pose, target_heading, tol=0.05):
        # pose: (x, y, theta) — só o theta é usado aqui
        # target_heading: ângulo alvo (rad) para alinhar o robô
        # tol: tolerância angular (rad) para considerar alinhado
        # devolve: (v, w, alinhado) — v sempre 0 (gira no sítio), w é a velocidade angular

        _, _, theta = pose                 # ignora x,y; usa apenas theta
        err = _ang_diff(target_heading, theta)  # erro angular entre alvo e atual

        if abs(err) < tol:                 # se o erro estiver dentro da tolerância
            return 0.0, 0.0, True          # parar e sinalizar "alinhado"

        # controlo angular proporcional com saturação
        w = max(-self.max_ang, min(self.max_ang, self.k_ang * err))
        return 0.0, w, False               # roda no sítio com velocidade w
