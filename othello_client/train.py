from ai_engine import OthelloAI
import json

def main():
    ai = OthelloAI()

    num_games = 600

    print(f"Entrenando en {num_games} partidas de self-play…")
    ai.self_play_and_train(games=num_games)

    print("Entrenamiento completado. Pesos finales:")
    for feat, w in ai.weights.items():
        print(f"  {feat}: {w:.4f}")
        
    with open('weights_initial.json', 'w') as f:
        json.dump({
            'positional_sum': 10.0,
            'mobility_ratio': 78.0,
            'parity_ratio': 10.0,
            'frontier_diff': 12.0
        }, f, indent=2)

    with open('weights_trained.json', 'w') as f:
        json.dump(ai.weights, f, indent=2)

if __name__ == "__main__":
    main()
