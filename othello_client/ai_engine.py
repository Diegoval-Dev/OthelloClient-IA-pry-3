import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from copy import deepcopy



# Definición de features y pesos iniciales
FEATURES = [
    'positional_sum',    # Suma de valores posicionales
    'mobility_ratio',    # (mis_movs - movs_oponente)/(total_movs)
    'parity_ratio',      # Ventaja de piezas en fase final
    'frontier_diff',     # Diferencia de discos en frontera
]

# Pesos internos para evaluación compuesta
POSITION_WEIGHTS = [
    100, -20, 10,  5,  5, 10, -20, 100,
    -20, -50, -2, -2, -2, -2, -50, -20,
     10,  -2,  0,  0,  0,  0,  -2,  10,
      5,  -2,  0,  0,  0,  0,  -2,   5,
      5,  -2,  0,  0,  0,  0,  -2,   5,
     10,  -2,  0,  0,  0,  0,  -2,  10,
    -20, -50, -2, -2, -2, -2, -50, -20,
    100, -20, 10,  5,  5, 10, -20, 100
]

# Zobrist: pre-generar claves aleatorias para [posición][jugador]
ZOBRIST_TABLE = [[random.getrandbits(64) for _ in range(2)] for _ in range(64)]

def initial_board():
    board = [[0]*8 for _ in range(8)]
    board[3][3], board[4][4] = 1, 1
    board[3][4], board[4][3] = -1, -1
    return board

class OthelloAI:
    def __init__(self):
        weights_path = "weights_trained.json"
        self.start_time = 0
        self.time_limit = 0
        self.best_move = None
        self.max_depth_reached = 0
        self.transposition_table = {}
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                self.weights = json.load(f)
            print(f"[AI] Pesos cargados desde {weights_path}")
        else:
            # pesos iniciales por defecto
            self.weights = {
                'positional_sum':  10.0,
                'mobility_ratio':  78.0,
                'parity_ratio':    10.0,
                'frontier_diff':   12.0,
            }
        # Parámetros TD-Learning
        self.gamma = 0.99
        self.alpha = 1e-4

    def valid_moves(self, board, symbol):
        """
        Escanea cada casilla; replica is_valid_move de OthelloGame
        para devolver lista de (r,c) que el servidor aceptará.
        """
        def is_valid(board, symbol, r, c):
            if board[r][c] != 0:
                return False
            opponent = -symbol
            directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            for dr,dc in directions:
                rr, cc = r+dr, c+dc
                found_op = False
                while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == opponent:
                    rr += dr; cc += dc
                    found_op = True
                if found_op and 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == symbol:
                    return True
            return False

        moves = []
        for r in range(8):
            for c in range(8):
                if is_valid(board, symbol, r, c):
                    moves.append((r, c))
        return moves
    def _ensure_valid(self, board, symbol, move):
        """
        Si `move` no está en valid_moves, elige como respaldo el que
        maximice self.evaluate().
        """
        valids = self.valid_moves(board, symbol)
        if move in valids:
            return move

        # Si por algún motivo self.best_move era None o ilegal:
        best_move, best_val = None, -float('inf')
        for (r, c) in valids:
            idx = r*8 + c
            # Reconstruye bitboards según symbol
            white_bb, black_bb = self.board_to_bitboards(board)
            if symbol == 1:
                my_bb, opp_bb = white_bb, black_bb
            else:
                my_bb, opp_bb = black_bb, white_bb
            new_my, new_opp = self.apply_move(my_bb, opp_bb, idx)
            val = self.evaluate(new_my, new_opp)
            if val > best_val:
                best_val, best_move = val, (r, c)

        return best_move


    def board_to_bitboards(self, board):
        white_bb = black_bb = 0
        for r in range(8):
            for c in range(8):
                idx = r*8 + c
                if board[r][c] == 1:
                    white_bb |= 1 << idx
                elif board[r][c] == -1:
                    black_bb |= 1 << idx
        return white_bb, black_bb
    
    def extract_features(self, my_bb, opp_bb):
        # Positional sum
        pos_sum = sum((POSITION_WEIGHTS[idx] if (my_bb>>idx)&1 else -POSITION_WEIGHTS[idx])
                      for idx in range(64) if ((my_bb|opp_bb)>>idx)&1)
        # Mobility
        my_moves = bin(self.generate_moves(my_bb, opp_bb)).count('1')
        opp_moves = bin(self.generate_moves(opp_bb, my_bb)).count('1')
        mobility = ((my_moves - opp_moves)/(my_moves + opp_moves)
                    if my_moves+opp_moves>0 else 0)
        # Parity
        total = bin(my_bb|opp_bb).count('1')
        if total>48:
            m = bin(my_bb).count('1'); o = bin(opp_bb).count('1')
            parity = (m-o)/(m+o)
        else:
            parity = 0
        # Frontier
        frontier = 0
        occ = my_bb|opp_bb
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for idx in range(64):
            if (occ>>idx)&1:
                r,c = divmod(idx,8)
                for dr,dc in dirs:
                    rr,cc = r+dr, c+dc
                    if 0<=rr<8 and 0<=cc<8 and not ((occ>>(rr*8+cc))&1):
                        frontier += -1 if (my_bb>>idx)&1 else 1
                        break
        pos_sum  /= 100
        mobility /= 1  
        parity   /= 1     
        frontier /= 10  
        return {
            'positional_sum': pos_sum,
            'mobility_ratio': mobility,
            'parity_ratio':   parity,
            'frontier_diff':  frontier
        }
        
    def evaluate(self, my_bb, opp_bb):
        feats = self.extract_features(my_bb, opp_bb)
        return sum(self.weights[f] * feats[f] for f in feats)
    
    def td_update(self, feats_t, reward, feats_tp1=None):
        v_t = sum(self.weights[f]*feats_t[f] for f in feats_t)
        v_tp1 = sum(self.weights[f]*feats_tp1[f] for f in feats_tp1) if feats_tp1 else 0
        delta = reward + self.gamma*v_tp1 - v_t
        delta = max(min(delta, 1.0), -1.0)
        for f in self.weights:
            self.weights[f] += self.alpha * delta * feats_t[f]

    def zobrist_hash(self, my_bb, opp_bb):
        h = 0
        for idx in range(64):
            if (my_bb>>idx)&1: h ^= ZOBRIST_TABLE[idx][0]
            elif (opp_bb>>idx)&1: h ^= ZOBRIST_TABLE[idx][1]
        return h

    def generate_moves(self, my_bb, opp_bb):
        empty = ~(my_bb|opp_bb) & ((1<<64)-1)
        moves = 0
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for r in range(8):
            for c in range(8):
                idx=r*8+c
                if not (empty>>idx)&1: continue
                for dr,dc in dirs:
                    rr,cc=r+dr,c+dc; path=0
                    while 0<=rr<8 and 0<=cc<8 and ((opp_bb>>(rr*8+cc))&1):
                        path |= 1<<(rr*8+cc); rr+=dr; cc+=dc
                    if 0<=rr<8 and 0<=cc<8 and ((my_bb>>(rr*8+cc))&1):
                        moves |= 1<<idx; break
        return moves

    def apply_move(self, my_bb, opp_bb, idx):
        r,c=divmod(idx,8)
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        flip=0
        for dr,dc in dirs:
            rr,cc=r+dr,c+dc; path=0
            while 0<=rr<8 and 0<=cc<8 and ((opp_bb>>(rr*8+cc))&1):
                path|=1<<(rr*8+cc); rr+=dr; cc+=dc
            if 0<=rr<8 and 0<=cc<8 and ((my_bb>>(rr*8+cc))&1): flip|=path
        return (my_bb | (1<<idx) | flip, opp_bb & ~flip)

    def order_moves(self, moves, my_bb, opp_bb):
        return sorted(moves, key=lambda mv: POSITION_WEIGHTS[mv[0]*8+mv[1]], reverse=True)


    def moves_list(self, moves_bb):
        moves=[]
        while moves_bb:
            lsb=moves_bb & -moves_bb; idx=lsb.bit_length()-1
            moves.append((idx//8, idx%8)); moves_bb&=moves_bb-1
        return moves

    def evaluate(self, my_bb, opp_bb):
        feats=self.extract_features(my_bb,opp_bb)
        return sum(self.weights[f]*feats[f] for f in feats)

    def negamax(self, my_bb, opp_bb, depth, alpha, beta, color):
        if time.time()-self.start_time>self.time_limit: raise TimeoutError
        # Transposition table omitted for brevity
        moves_bb=self.generate_moves(my_bb,opp_bb)
        if depth==0 or moves_bb==0:
            return self.evaluate(my_bb,opp_bb)
        value=-float('inf')
        for r,c in self.order_moves(self.moves_list(moves_bb),my_bb,opp_bb):
            idx=r*8+c
            new_my,new_opp=self.apply_move(my_bb,opp_bb,idx)
            score=-self.negamax(new_opp,new_my,depth-1,-beta,-alpha,-color)
            value=max(value,score); alpha=max(alpha,score)
            if alpha>=beta: break
        if depth==self.max_depth_reached: self.best_move=(r,c)
        return value
    
    def find_best_move(self, board, symbol, time_limit=2.5):
        white_bb,black_bb=self.board_to_bitboards(board)
        my_bb,opp_bb=(white_bb,black_bb) if symbol==1 else (black_bb,white_bb)
        self.start_time=time.time(); self.time_limit=time_limit
        self.best_move=None; depth=1
        while True:
            try:
                self.max_depth_reached=depth
                self.negamax(my_bb,opp_bb,depth,-float('inf'),float('inf'),1)
                depth+=1
            except TimeoutError:
                break
        final = self._ensure_valid(board, symbol, self.best_move)
        return final

    def self_play_and_train(self, games=100):
        """
        Ejecuta partidas de auto-juego y aplica TD(0) para ajustar pesos.
        """
        for _ in range(games):
            board = initial_board()
            history = []  # lista de (feats, symbol)
            symbol = 1
            # jugar hasta el final
            while True:
                my_bb, opp_bb = self.board_to_bitboards(board) if symbol==1 else self.board_to_bitboards([[ -x for x in row] for row in board])
                feats = self.extract_features(my_bb, opp_bb)
                move = self.find_best_move(board, symbol, time_limit=0.01)
                if move is None:
                    # pase, sin jugadas
                    history.append((feats, symbol))
                    symbol = -symbol
                    continue
                history.append((feats, symbol))
                idx = move[0]*8 + move[1]
                my_bb, opp_bb = self.apply_move(my_bb, opp_bb, idx)
                # reconstruir board de bitboards
                board = [[0]*8 for _ in range(8)]
                for i in range(64):
                    r,c=divmod(i,8)
                    if (my_bb>>i)&1: board[r][c] = symbol
                    elif (opp_bb>>i)&1: board[r][c] = -symbol
                symbol = -symbol
                # verificar fin: no movimientos para ambos
                if self.generate_moves(my_bb,opp_bb)==0 and self.generate_moves(opp_bb,my_bb)==0:
                    # determinar ganador
                    wb = bin(my_bb).count('1'); ob = bin(opp_bb).count('1')
                    if wb>ob: reward_map = {1:1,-1:-1}
                    elif ob>wb: reward_map = {1:-1,-1:1}
                    else: reward_map={1:0,-1:0}
                    # actualización TD
                    for i in range(len(history)):
                        feats_t, sym = history[i]
                        feats_tp1 = history[i+1][0] if i+1<len(history) else None
                        self.td_update(feats_t, reward_map[sym], feats_tp1)
                    break
