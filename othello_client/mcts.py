# ==== mcts.py ====

import math
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

from ai_engine import OthelloAI, initial_board

class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        # state = (board, symbol)
        self.state = state
        self.parent = parent
        self.children = {}      # move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MCTS:
    def __init__(self, net, c_puct=1.0, n_sim=800, device='cuda'):
        # neural net (policy+value), exploración, simulaciones, dispositivo
        self.net = net.to(device)
        self.device = device
        self.c_puct = c_puct
        self.n_sim = n_sim
        # helper para bitboards y apply_move
        self.helper_ai = OthelloAI()

    def board_to_tensor(self, board, symbol):
        """Convierte un tablero y símbolo a tensor [2,8,8] en device."""
        arr = np.zeros((2, 8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                if board[r][c] == symbol:
                    arr[0, r, c] = 1.0
                elif board[r][c] == -symbol:
                    arr[1, r, c] = 1.0
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)  # agrega batch dim

    def get_legal_moves(self, board, symbol):
        """Devuelve lista de (r,c) válidos para `symbol`."""
        if symbol == 1:
            my_bb, opp_bb = self.helper_ai.board_to_bitboards(board)
        else:
            inv = [[-x for x in row] for row in board]
            my_bb, opp_bb = self.helper_ai.board_to_bitboards(inv)
        moves_bb = self.helper_ai.generate_moves(my_bb, opp_bb)
        return self.helper_ai.moves_list(moves_bb)

    def apply_move(self, board, symbol, move):
        """
        Aplica `move` a `board` para `symbol` y devuelve (nuevo_board, -symbol).
        Usa la lógica de bitboards de helper_ai.
        """
        r, c = move
        idx = r * 8 + c
        if symbol == 1:
            my_bb, opp_bb = self.helper_ai.board_to_bitboards(board)
        else:
            inv = [[-x for x in row] for row in board]
            my_bb, opp_bb = self.helper_ai.board_to_bitboards(inv)
        new_my, new_opp = self.helper_ai.apply_move(my_bb, opp_bb, idx)

        # reconstruye tablero 2D
        new_board = [[0] * 8 for _ in range(8)]
        for i in range(64):
            rr, cc = divmod(i, 8)
            if (new_my >> i) & 1:
                new_board[rr][cc] = symbol
            elif (new_opp >> i) & 1:
                new_board[rr][cc] = -symbol

        return new_board, -symbol

    def select(self, node):
        """Selecciona el hijo con UCB más alto."""
        total = sum(child.visit_count for child in node.children.values())
        best_score = -float('inf')
        best_child = None
        for move, child in node.children.items():
            u = (self.c_puct * child.prior *
                 math.sqrt(total) / (1 + child.visit_count))
            score = child.value() + u
            if score > best_score:
                best_score, best_child = score, child
        return best_child

    def expand_and_evaluate(self, node):
        """
        Expande `node` generando sus hijos y evalúa la posición con la red.
        Retorna el valor (float) para backprop.
        """
        board, symbol = node.state

        # 1) forward pass red
        state_tensor = self.board_to_tensor(board, symbol)
        with torch.no_grad():
            logits, value = self.net(state_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            v = value.item()

        # 2) expand children con priors de la política
        legal_moves = self.get_legal_moves(board, symbol)
        for (r, c) in legal_moves:
            idx = r * 8 + c
            prior = float(probs[idx])
            next_board, next_symbol = self.apply_move(board, symbol, (r, c))
            node.children[(r, c)] = MCTSNode(
                (next_board, next_symbol),
                parent=node,
                prior=prior
            )

        return v

    def backpropagate(self, path, value):
        """Actualiza visit_count y value_sum a lo largo de `path` inverso."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # intercambia signo para oponente

    def select_action(self, root):
        """Devuelve el move cuyo child tenga mayor visit_count."""
        best_move, best_child = None, None
        for move, child in root.children.items():
            if best_child is None or child.visit_count > best_child.visit_count:
                best_move, best_child = move, child
        return best_move

    def run(self, root_board, symbol):
        """
        Corre MCTS desde la posición inicial (root_board, symbol)
        retorna el mejor move.
        """
        root = MCTSNode((root_board, symbol))
        self.root = root 
        for _ in range(self.n_sim):
            node = root
            path = [node]

            # Selection
            while node.children:
                node = self.select(node)
                path.append(node)

            # Expansion + Evaluation
            v = self.expand_and_evaluate(node)

            # Backpropagation
            self.backpropagate(path, v)

        # Elige acción tras MCTS
        return self.select_action(root)
