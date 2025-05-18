import requests
import random
import sys
import time
from ai_engine import OthelloAI

### Public IP Server
### Testing Server
host_name = 'http://localhost:8000'

class OthelloPlayer():

    def __init__(self, username):
        ### Player username
        self.username = username
        ### Player symbol in una partida
        self.current_symbol = 0
        ### Instancia del motor de IA
        self.ai = OthelloAI()

    def connect(self, session_name) -> bool:
        new_player = requests.post(
            f"{host_name}/player/new_player?session_name={session_name}&player_name={self.username}"
        ).json()
        self.session_name = session_name
        print(new_player.get('message', ''))
        return new_player.get('status') == 200

    def play(self) -> None:
        session_info = requests.post(
            f"{host_name}/game/game_info?session_name={self.session_name}"
        ).json()

        while session_info.get('session_status') == 'active':
            try:
                if session_info.get('round_status') == 'ready':
                    match_info = requests.post(
                        f"{host_name}/player/match_info?session_name={self.session_name}&player_name={self.username}"
                    ).json()

                    # Esperar si está en bench
                    while match_info.get('match_status') == 'bench':
                        print('Estás en bench esta ronda. Espera...')
                        time.sleep(2)
                        match_info = requests.post(
                            f"{host_name}/player/match_info?session_name={self.session_name}&player_name={self.username}"
                        ).json()

                    if match_info.get('match_status') == 'active':
                        self.current_symbol = match_info.get('symbol')
                        color = 'White (⬤)' if self.current_symbol == 1 else 'Black (◯)'
                        print(f'Vamos a jugar! Eres {color}.')

                    # Ciclo de turnos
                    turn_info = requests.post(
                        f"{host_name}/player/turn_to_move?session_name={self.session_name}&player_name={self.username}&match_id={match_info.get('match')}"
                    ).json()

                    # Asegurar que game_over exista
                    while not turn_info.get('game_over', False):
                        if turn_info.get('turn'):
                            print('Puntuación', turn_info.get('score', ''))
                            board = turn_info.get('board')
                            move = self.AI_MOVE(board)
                            if move is None:
                                print('No hay movimientos válidos. Pasando turno...')
                            else:
                                row, col = move
                                resp = requests.post(
                                    f"{host_name}/player/move?session_name={self.session_name}&player_name={self.username}&match_id={match_info.get('match')}&row={row}&col={col}"
                                ).json()
                                print(resp.get('message', ''))
                        time.sleep(1)
                        turn_info = requests.post(
                            f"{host_name}/player/turn_to_move?session_name={self.session_name}&player_name={self.username}&match_id={match_info.get('match')}"
                        ).json()

                    print('Juego terminado. Ganador:', turn_info.get('winner', 'Desconocido'))
                    session_info = requests.post(
                        f"{host_name}/game/game_info?session_name={self.session_name}"
                    ).json()
                else:
                    print('Esperando lotería de emparejamiento...')
                    time.sleep(2)
                    session_info = requests.post(
                        f"{host_name}/game/game_info?session_name={self.session_name}"
                    ).json()
            except requests.exceptions.ConnectionError:
                continue

    def AI_MOVE(self, board):
        # board: lista de listas
        return self.ai.find_best_move(board, self.current_symbol, time_limit=2.5)

if __name__ == '__main__':
    session_id = sys.argv[1]
    player_id = sys.argv[2]
    print('Bienvenido', player_id)
    player = OthelloPlayer(player_id)
    if player.connect(session_id):
        player.play()
    print('Hasta luego')