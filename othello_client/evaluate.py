import json
import argparse
from ai_engine import OthelloAI, initial_board

def load_weights(path):
    with open(path) as f:
        return json.load(f)

def play_match(ai1, ai2, time_limit):
    """
    Juega una partida completa entre ai1 (White) y ai2 (Black).
    Devuelve:  1 si White gana, -1 si Black gana, 0 empate.
    """
    board = initial_board()
    symbol = 1  # White mueve siempre primero
    passes = 0
    while True:
        ai = ai1 if symbol == 1 else ai2
        move = ai.find_best_move(board, symbol, time_limit=time_limit)
        if move is None:
            passes += 1
            if passes >= 2:
                # ambos pasaron→fin de partida
                break
        else:
            passes = 0
            r, c = move
            # aplicar movimiento al tablero
            board[r][c] = symbol
            # voltear fichas manualmente (reusa apply_move)
            my_bb, opp_bb = ai.board_to_bitboards(board) if symbol==1 else ai.board_to_bitboards([[-x for x in row] for row in board])
            new_my, new_opp = ai.apply_move(my_bb, opp_bb, r*8+c)
            # reconstruir board desde bitboards
            new_board = [[0]*8 for _ in range(8)]
            for i in range(64):
                rr,cc = divmod(i,8)
                if (new_my >> i) & 1:
                    new_board[rr][cc] = symbol
                elif (new_opp >> i) & 1:
                    new_board[rr][cc] = -symbol
            board = new_board
        symbol *= -1

    # contar discos finales
    w = sum(cell == 1 for row in board for cell in row)
    b = sum(cell == -1 for row in board for cell in row)
    if w > b: return 1
    if b > w: return -1
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained', required=True, help='Ruta a weights_trained.json')
    parser.add_argument('--baseline', required=True, help='Ruta a weights_initial.json')
    parser.add_argument('--games', type=int, default=100, help='Número de partidas a jugar')
    parser.add_argument('--time', type=float, default=0.5, help='Límite de tiempo por jugada (s)')
    args = parser.parse_args()

    # Crea dos motores y carga pesos
    ai_trained  = OthelloAI()
    ai_baseline = OthelloAI()
    ai_trained.weights  = load_weights(args.trained)
    ai_baseline.weights = load_weights(args.baseline)

    results = {1:0, 0:0, -1:0}
    for i in range(args.games):
        # Alternar quien comienza
        if i % 2 == 0:
            res = play_match(ai_trained, ai_baseline, args.time)
        else:
            # Black comienza jugando como White: intercambia motores
            res = play_match(ai_baseline, ai_trained, args.time)
            res = -res  # invierte resultado
        results[res] += 1
        print(f'Partida {i+1}/{args.games}: resultado {res}')

    print('\n=== Estadísticas ===')
    print(f'Victorias IA entrenada: {results[1]} ({results[1]/args.games:.1%})')
    print(f'Empates:               {results[0]} ({results[0]/args.games:.1%})')
    print(f'Derrotas IA entrenada:{results[-1]} ({results[-1]/args.games:.1%})')

if __name__ == '__main__':
    main()
  # python evaluate.py --trained=weights_trained.json --baseline=weights_initial.json --games=100 --time=0.5 