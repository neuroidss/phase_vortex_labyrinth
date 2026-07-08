# implicit_config.py
import numpy as np

COORDS_16_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14])
COORDS_16_Y = np.array([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71])

REF_16_X = 5.5
REF_16_Y = 0.0

GND_16_X = -5.49
GND_16_Y = 0.0

# === ГЕНЕРАТИВНЫЙ РЕЕСТР ИНГРЕДИЕНТОВ ===
# Каждый ингредиент - это комплексный вектор [R, G, B] единичной длины
ALCHEMY_ENTITIES_CONFIG = [
    {
        'type': 'yang_essence',
        'name': 'YANG ESSENCE',
        'freq': 80.0,
        'tq': 45.0,
        'rgb': 0, # Legacy для аудио-роутинга
        'vector': [1.0, 0.0, 0.0],
        'color': (255, 50, 0),
        'offset': [-20.0, -10.0]
    },
    {
        'type': 'yin_essence',
        'name': 'YIN ESSENCE',
        'freq': 6.0,
        'tq': -30.0,
        'rgb': 4, # Legacy для аудио-роутинга
        'vector': [0.0, 0.0, 1.0],
        'color': (0, 100, 255),
        'offset': [20.0, -10.0]
    },
    {
        'type': 'smr_catalyst',
        'name': 'SMR CATALYST',
        'freq': 14.0,
        'tq': 35.0,
        'rgb': 2, # Legacy для аудио-роутинга
        'vector': [0.0, 1.0, 0.0],
        'color': (0, 255, 50),
        'offset': [0.0, 20.0]
    }
]

# === ЭМЕРДЖЕНТНЫЕ ЦЕЛИ СИНТЕЗА (ПИЛЮЛИ) ===
# Определяются косинусным сходством в конце плавки
SEMANTIC_PILLS_DB = {
    "Foundation Pill": {
        "vector": [0.577, 0.577, 0.577], # Равномерная смесь (Yang + Yin + Catalyst)
        "color": (255, 200, 255)
    },
    "Pure Yang Core": {
        "vector": [0.894, 0.447, 0.0],   # Много Yang, немного Catalyst (2x Yang + 1 Catalyst)
        "color": (255, 100, 0)
    },
    "Deep Yin Core": {
        "vector": [0.0, 0.447, 0.894],   # Много Yin, немного Catalyst (2x Yin + 1 Catalyst)
        "color": (0, 150, 255)
    },
    "Turbulent Anomaly": {
        "vector": [0.707, 0.0, 0.707],   # Конфликт Ян и Инь без стабилизатора (Yang + Yin)
        "color": (200, 0, 255)
    }
}
