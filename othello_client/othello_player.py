import requests
import sys
import time
from ai_engine import OthelloAI
from requests.exceptions import JSONDecodeError

### Public IP Server / Testing Server
host_name = 'http://localhost:8000'

def post_json(url):
    resp = requests.post(url)
    if resp.status_code != 200 or not resp.text:
        print(f"[WARN] respuesta no válida {resp.status_code} de {url!r}")
        return {}
    try:
        return resp.json()
    except JSONDecodeError:
        print(f"[WARN] no pude decodificar JSON de {url!r}: {resp.text!r}")
        return {}

class OthelloPlayer:

    def __init__(self, username):
        self.username = username
        self.current_symbol = 0
        # 1) Instancia tu IA y carga sus pesos entrenados
        self.ai = OthelloAI()
        # Si quieres forzar otra ruta de pesos:
        # self.ai = OthelloAI(weights_path='weights_trained.json')

    def connect(self, session_name) -> bool:
        url = (
            f"{host_name}/player/new_player"
            f"?session_name={session_name}"
            f"&player_name={self.username}"
        )
        new_player = post_json(url)
        print(new_player.get('message', ''))
        self.session_name = session_name
        return new_player.get('status') == 200

    def play(self) -> None:
        session_info = post_json(
            f"{host_name}/game/game_info?session_name={self.session_name}"
        )

        while session_info.get('session_status') == 'active':
            try:
                if session_info.get('round_status') == 'ready':
                    match_info = post_json(
                        f"{host_name}/player/match_info"
                        f"?session_name={self.session_name}"
                        f"&player_name={self.username}"
                    )

                    # bench ...
                    while match_info.get('match_status') == 'bench':
                        print('Estás en bench esta ronda. Espera...')
                        time.sleep(2)
                        match_info = post_json(
                            f"{host_name}/player/match_info"
                            f"?session_name={self.session_name}"
                            f"&player_name={self.username}"
                        )

                    if match_info.get('match_status') == 'active':
                        self.current_symbol = match_info.get('symbol', 0)
                        color = 'White (⬤)' if self.current_symbol == 1 else 'Black (◯)'
                        print(f'Vamos a jugar! Eres {color}.')

                    # turno a turno...
                    turn_info = post_json(
                        f"{host_name}/player/turn_to_move"
                        f"?session_name={self.session_name}"
                        f"&player_name={self.username}"
                        f"&match_id={match_info.get('match','')}"
                    )

                    while not turn_info.get('game_over', False):
                        if turn_info.get('turn', False):
                            print('Puntuación', turn_info.get('score',''))
                            board = turn_info.get('board', [])
                            move = self.AI_MOVE(board)

                            if move is None:
                                print('No hay movimientos válidos. Pasando turno…')
                            else:
                                r, c = move
                                resp = post_json(
                                    f"{host_name}/player/move"
                                    f"?session_name={self.session_name}"
                                    f"&player_name={self.username}"
                                    f"&match_id={match_info.get('match','')}"
                                    f"&row={r}&col={c}"
                                )
                                print(resp.get('message',''))

                        time.sleep(1)
                        turn_info = post_json(
                            f"{host_name}/player/turn_to_move"
                            f"?session_name={self.session_name}"
                            f"&player_name={self.username}"
                            f"&match_id={match_info.get('match','')}"
                        )

                    print('Juego terminado. Ganador:', turn_info.get('winner','Desconocido'))
                    session_info = post_json(
                        f"{host_name}/game/game_info?session_name={self.session_name}"
                    )
                else:
                    print('Esperando lotería de emparejamiento...')
                    time.sleep(2)
                    session_info = post_json(
                        f"{host_name}/game/game_info?session_name={self.session_name}"
                    )

            except requests.exceptions.ConnectionError:
                continue

    def AI_MOVE(self, board):
        """
        1) Obtiene valid_moves de tu IA.
        2) Llama a find_best_move.
        3) Si el resultado NO está en valid_moves, elige un respaldo
           que maximice tu función de evaluación.
        """
        # 1) movimientos legales
        valids = self.ai.valid_moves(board, self.current_symbol)
        if not valids:
            return None

        # 2) mejor movida según tu buscador
        move = self.ai.find_best_move(board, self.current_symbol, time_limit=2.5)

        # 3) si es inválido, elige respaldo
        if move not in valids:
            best_move, best_val = None, -float('inf')
            for m in valids:
                # simula en bitboards y mide tu evaluate()
                white_bb, black_bb = self.ai.board_to_bitboards(board)
                if self.current_symbol == 1:
                    my_bb, opp_bb = white_bb, black_bb
                else:
                    my_bb, opp_bb = black_bb, white_bb
                idx = m[0]*8 + m[1]
                new_my, new_opp = self.ai.apply_move(my_bb, opp_bb, idx)
                val = self.ai.evaluate(new_my, new_opp)
                if val > best_val:
                    best_val, best_move = val, m
            move = best_move

        return move


if __name__ == '__main__':
    session_id = sys.argv[1]
    player_id  = sys.argv[2]
    print('Bienvenido', player_id)
    p = OthelloPlayer(player_id)
    if p.connect(session_id):
        p.play()
    print('Hasta luego')
