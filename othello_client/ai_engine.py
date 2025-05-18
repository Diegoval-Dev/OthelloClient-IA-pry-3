import time
import random

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

class OthelloAI:
    def __init__(self):
        self.start_time = 0
        self.time_limit = 0
        self.best_move = None
        self.max_depth_reached = 0
        self.transposition_table = {}  # {zobrist_key: (depth, value, flag)}

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

    def zobrist_hash(self, my_bb, opp_bb):
        h = 0
        for idx in range(64):
            if (my_bb >> idx) & 1:
                h ^= ZOBRIST_TABLE[idx][0]
            elif (opp_bb >> idx) & 1:
                h ^= ZOBRIST_TABLE[idx][1]
        return h

    def generate_moves(self, my_bb, opp_bb):
        empty = ~(my_bb | opp_bb) & ((1<<64)-1)
        moves = 0
        directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for r in range(8):
            for c in range(8):
                idx = r*8 + c
                if not (empty >> idx) & 1: continue
                for dr,dc in directions:
                    rr,cc = r+dr, c+dc
                    path=0
                    while 0<=rr<8 and 0<=cc<8 and ((opp_bb>>(rr*8+cc))&1):
                        path |= 1<<(rr*8+cc)
                        rr+=dr; cc+=dc
                    if 0<=rr<8 and 0<=cc<8 and ((my_bb>>(rr*8+cc))&1):
                        moves |= 1<<idx
                        break
        return moves

    def apply_move(self, my_bb, opp_bb, idx):
        r,c = divmod(idx,8)
        directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        flip=0
        for dr,dc in directions:
            rr,cc=r+dr,c+dc; path=0
            while 0<=rr<8 and 0<=cc<8 and ((opp_bb>>(rr*8+cc))&1):
                path|=1<<(rr*8+cc); rr+=dr; cc+=dc
            if 0<=rr<8 and 0<=cc<8 and ((my_bb>>(rr*8+cc))&1):
                flip |= path
        return (my_bb | (1<<idx) | flip, opp_bb & ~flip)

    def order_moves(self, moves, my_bb, opp_bb):
        # heurística simple: posiciones con mayor valor posicional primero
        scored = []
        for r,c in moves:
            idx=r*8+c
            scored.append(((POSITION_WEIGHTS[idx]), (r,c)))
        scored.sort(reverse=True)
        return [mv for _,mv in scored]

    def moves_list(self, moves_bb):
        moves=[]
        while moves_bb:
            lsb=moves_bb & -moves_bb; idx=lsb.bit_length()-1
            moves.append((idx//8, idx%8)); moves_bb&=moves_bb-1
        return moves

    def evaluate(self, my_bb, opp_bb):
        pos=mob=parity=front=0
        # posicional
        for idx,w in enumerate(POSITION_WEIGHTS):
            if (my_bb>>idx)&1: pos+=w
            elif (opp_bb>>idx)&1: pos-=w
        # movilidad
        my_m=bin(self.generate_moves(my_bb,opp_bb)).count('1')
        op_m=bin(self.generate_moves(opp_bb,my_bb)).count('1')
        if my_m+op_m>0: mob=100*(my_m-op_m)/(my_m+op_m)
        # paridad late game
        total=bin(my_bb|opp_bb).count('1')
        if total>48:
            m_cnt=bin(my_bb).count('1'); o_cnt=bin(opp_bb).count('1')
            parity=100*(m_cnt-o_cnt)/(m_cnt+o_cnt)
        # frontera
        dirs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        board_occ=my_bb|opp_bb
        for idx in range(64):
            if (board_occ>>idx)&1:
                r,c=divmod(idx,8)
                for dr,dc in dirs:
                    rr,cc=r+dr,c+dc
                    if 0<=rr<8 and 0<=cc<8 and not ((board_occ>>(rr*8+cc))&1):
                        front += -1 if (my_bb>>idx)&1 else 1
                        break
        # composición
        return 10*pos + 78*mob + 10*parity + 12*front

    def negamax(self, my_bb, opp_bb, depth, alpha, beta, color):
        # color=1 o -1 para manejo TT entries
        if time.time()-self.start_time>self.time_limit: raise TimeoutError
        zob_key=self.zobrist_hash(my_bb,opp_bb)
        if zob_key in self.transposition_table:
            d,val,flag=self.transposition_table[zob_key]
            if d>=depth:
                if flag=='EXACT': return val
                if flag=='LOWER' and val>alpha: alpha=val
                if flag=='UPPER' and val<beta: beta=val
                if alpha>=beta: return val
        moves_bb=self.generate_moves(my_bb,opp_bb)
        if depth==0 or moves_bb==0:
            return self.evaluate(my_bb,opp_bb)
        value=-float('inf')
        best_local=None
        moves=self.order_moves(self.moves_list(moves_bb),my_bb,opp_bb)
        for r,c in moves:
            idx=r*8+c
            new_my,new_opp=self.apply_move(my_bb,opp_bb,idx)
            score=-self.negamax(new_opp,new_my,depth-1,-beta,-alpha,-color)
            if score>value:
                value=score; best_local=(r,c)
            alpha=max(alpha,score)
            if alpha>=beta: break
        # store TT
        flag='EXACT'
        if value<=alpha_orig: flag='UPPER'
        elif value>=beta: flag='LOWER'
        self.transposition_table[zob_key]=(depth,value,flag)
        if depth==self.max_depth_reached: self.best_move=best_local
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
        return self.best_move
