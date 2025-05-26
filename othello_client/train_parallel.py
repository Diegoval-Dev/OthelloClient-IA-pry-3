# train_parallel.py

import json
from multiprocessing import Pool, cpu_count
from ai_engine import OthelloAI

def worker_train(n_games):
    """
    Función que ejecuta n_games de self-play en un proceso separado
    y devuelve el diccionario de pesos resultante.
    """
    ai = OthelloAI()
    ai.self_play_and_train(games=n_games)
    return ai.weights

def average_weights(weights_list):
    """
    Promedia una lista de diccionarios de pesos.
    """
    avg = {}
    keys = weights_list[0].keys()
    for k in keys:
        avg[k] = sum(w[k] for w in weights_list) / len(weights_list)
    return avg

def main():
    total_games = 5000
    n_procs = min(cpu_count(), 8)  # por ejemplo, usa hasta 8 procesos
    games_per_proc = total_games // n_procs

    print(f"Entrenando en paralelo {total_games} partidas con {n_procs} procesos…")

    # Crear pool de procesos y repartir trabajo
    with Pool(processes=n_procs) as pool:
        # Cada proceso corre worker_train(games_per_proc)
        results = pool.map(worker_train, [games_per_proc] * n_procs)

    # Si hay resto de partidas, haz una pasada extra en el proceso principal
    remainder = total_games - games_per_proc * n_procs
    if remainder > 0:
        print(f"Entrenando {remainder} partidas adicionales en el proceso principal…")
        results.append(worker_train(remainder))

    # Promediar todos los vectores de pesos obtenidos
    final_weights = average_weights(results)

    print("Entrenamiento paralelo completado. Pesos promedio:")
    for feat, w in final_weights.items():
        print(f"  {feat}: {w:.4f}")

    # Guardar los pesos inicial y entrenado
    with open('weights_initial.json', 'w') as f:
        json.dump({
            'positional_sum': 10.0,
            'mobility_ratio': 78.0,
            'parity_ratio': 10.0,
            'frontier_diff': 12.0
        }, f, indent=2)

    with open('weights_trained_parallel.json', 'w') as f:
        json.dump(final_weights, f, indent=2)

if __name__ == "__main__":
    main()
