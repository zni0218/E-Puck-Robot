import random

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


def get_positions(mode="all", occupied_bays=None):
    """
    Returns:
        (spawn_x, spawn_y), (target_name, target_x)
    """

    if occupied_bays is None:
        occupied_bays = []

    free_bays = [(name, x) for name, x in ALL_BAYS if name not in occupied_bays]

    if len(free_bays) == 0:
        free_bays = ALL_BAYS

    target_name, target_x = random.choice(free_bays)

    # spawn aleatório no mapa
    spawn_x = random.uniform(0.85, 1.35)
    spawn_y = random.uniform(0.40, 0.90)

    return (spawn_x, spawn_y), (target_name, target_x)