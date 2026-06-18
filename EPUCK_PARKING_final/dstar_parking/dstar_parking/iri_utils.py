"""
iri_utils.py  --  FUNÇÕES DE BAIXO NÍVEL DO WEBOTS
==================================================
Ponte entre "andar/rodar" e os motores reais do e-puck, mais leitura de posição (GPS)
e orientação (bússola).

Parâmetros reais do e-puck:
  raio da roda  R = 0.0205 m ; entre-eixos L = 0.052 m ; vel. máx. roda = 6.28 rad/s

SOBRE A BÚSSOLA (a dor de cabeça que termina aqui):
  raw_compass() devolve um ângulo que é 0 quando o robô aponta para +X. Mas o SINAL
  (se sobe ou desce quando o robô roda à esquerda) depende da convenção do mundo e pode
  estar invertido. Por isso NÃO confiamos no sinal aqui — o controller mede-o sozinho no
  arranque (auto-calibração) e aplica +1 ou -1. Ver dstar_parking.calibrate_heading_sign.
"""

import math

WHEEL_RADIUS = 0.0205
AXLE = 0.052
MAX_WHEEL = 6.28


def cmd_vel(robot, v, w):
    # Converte velocidade linear v (m/s) e angular w (rad/s) nas velocidades das DUAS rodas.
    # NOTA: w > 0 faz a roda esquerda recuar e a direita avançar -> o robô vira à ESQUERDA
    # (sentido anti-horário). Isto é física da tração diferencial, não depende de convenções.
    left = robot.getDevice("left wheel motor")
    right = robot.getDevice("right wheel motor")
    left.setPosition(float("inf"))      # modo de velocidade
    right.setPosition(float("inf"))
    wl = (v - w * AXLE / 2.0) / WHEEL_RADIUS
    wr = (v + w * AXLE / 2.0) / WHEEL_RADIUS
    m = max(abs(wl), abs(wr), 1e-9)     # se passar do máximo, reduz ambas na mesma proporção
    if m > MAX_WHEEL:
        wl *= MAX_WHEEL / m
        wr *= MAX_WHEEL / m
    left.setVelocity(wl)
    right.setVelocity(wr)


def stop(robot):
    cmd_vel(robot, 0.0, 0.0)


def gps_xy(gps):
    # Posição (x, y) em metros no referencial do mundo.
    v = gps.getValues()
    return v[0], v[1]


def raw_compass(compass):
    # Ângulo "cru": 0 quando o robô aponta para +X. O SINAL é calibrado pelo controller.
    c = compass.getValues()
    return math.atan2(-c[0], c[1])

