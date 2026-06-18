import random  # biblioteca usada para escolhas aleatórias (spawn e target)

# Mapa das bays de estacionamento.
# Cada "PARKED_X" representa um lugar fixo no ambiente Webots
# e o valor associado é a posição em X desse lugar no mapa.
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

# Lista de todas as bays no formato (nome, posição_x)
ALL_BAYS = list(PARKED_BAY_MAP.items())


def get_positions(mode="all", occupied_bays=None):
    """
    Esta função escolhe:
    - uma posição de spawn (onde o robô começa)
    - um target (onde o robô deve ir estacionar)

    occupied_bays: lista de bays que estão ocupadas por outros robôs
    """

    # Se não for fornecida lista de ocupadas, assume lista vazia
    if occupied_bays is None:
        occupied_bays = []

    # Filtra apenas as bays livres (não ocupadas)
    free_bays = [
        (name, x)
        for name, x in ALL_BAYS
        if name not in occupied_bays
    ]

    # Se por algum motivo não houver bays livres,
    # usa todas as bays como fallback
    if len(free_bays) == 0:
        free_bays = ALL_BAYS

    # Escolhe aleatoriamente uma bay livre como target
    target_name, target_x = random.choice(free_bays)

    # Define posição inicial do robô (spawn aleatório dentro de uma zona válida)
    spawn_x = random.uniform(0.85, 1.35)
    spawn_y = random.uniform(0.40, 0.90)

    # Retorna:
    # - posição de spawn (x, y)
    # - target (nome da bay + posição x)
    return (spawn_x, spawn_y), (target_name, target_x)