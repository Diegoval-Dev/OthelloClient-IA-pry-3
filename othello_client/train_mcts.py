# ==== train_mcts.py ====

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from policy_value_net import OthelloNet
from mcts import MCTS
from ai_engine import initial_board, OthelloAI

# Instancia auxiliar para acceder a bitboards y apply_move
helper_ai = OthelloAI()

def board_to_tensor(board, symbol):
    """
    Convierte un tablero 8×8 y símbolo en un tensor [batch,2,8,8]:
    canal 0 = posiciones propias, canal 1 = posiciones del oponente.
    """
    arr = np.zeros((2, 8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            if board[r][c] == symbol:
                arr[0, r, c] = 1.0
            elif board[r][c] == -symbol:
                arr[1, r, c] = 1.0
    return torch.from_numpy(arr)

def get_legal_moves(board, symbol):
    """
    Retorna la lista de movimientos legales para `symbol` usando la instancia helper_ai.
    """
    if symbol == 1:
        my_bb, opp_bb = helper_ai.board_to_bitboards(board)
    else:
        inv = [[-x for x in row] for row in board]
        my_bb, opp_bb = helper_ai.board_to_bitboards(inv)
    moves_bb = helper_ai.generate_moves(my_bb, opp_bb)
    return helper_ai.moves_list(moves_bb)

def apply_move_and_next(board, symbol, move):
    """
    Aplica `move` a `board` para `symbol` y devuelve (nuevo_board, siguiente_symbol).
    """
    r, c = move
    idx = r * 8 + c
    if symbol == 1:
        my_bb, opp_bb = helper_ai.board_to_bitboards(board)
    else:
        inv = [[-x for x in row] for row in board]
        my_bb, opp_bb = helper_ai.board_to_bitboards(inv)
    new_my, new_opp = helper_ai.apply_move(my_bb, opp_bb, idx)

    # Reconstruye tablero desde bitboards
    new_board = [[0] * 8 for _ in range(8)]
    for i in range(64):
        rr, cc = divmod(i, 8)
        if (new_my >> i) & 1:
            new_board[rr][cc] = symbol
        elif (new_opp >> i) & 1:
            new_board[rr][cc] = -symbol

    return new_board, -symbol

def game_over(board):
    """
    True si ninguno de los dos jugadores tiene movimientos legales.
    """
    return not get_legal_moves(board, 1) and not get_legal_moves(board, -1)

def compute_outcome(board):
    """
    +1 si White (1) gana, -1 si Black (-1) gana, 0 empate.
    """
    w = sum(cell == 1 for row in board for cell in row)
    b = sum(cell == -1 for row in board for cell in row)
    if w > b: return 1
    if b > w: return -1
    return 0

def get_pi_from_root(root_node):
    """
    Construye la distribución π(a|s) a partir de visit_count de cada hijo.
    Devuelve lista de 64 floats.
    """
    counts = [0] * 64
    for (r, c), child in root_node.children.items():
        counts[r*8 + c] = child.visit_count
    total = sum(counts)
    return [cnt/total if total>0 else 1/64 for cnt in counts]

def self_play(net, n_games=1000, n_sim=200):
    """
    Genera [(state_tensor, pi, z), …] mediante self-play con MCTS+red.
    """
    examples = []
    for _ in range(n_games):
        mcts = MCTS(net, n_sim=n_sim, device=device)
        board = initial_board()
        symbol = 1
        history = []

        # Jugar hasta el final
        while True:
            move = mcts.run(board, symbol)
            pi = get_pi_from_root(mcts.root)
            state_tensor = board_to_tensor(board, symbol)
            history.append((state_tensor, pi, symbol))

            board, symbol = apply_move_and_next(board, symbol, move)
            if game_over(board):
                break

        z_final = compute_outcome(board)
        for state_tensor, pi, sym in history:
            # Ajustamos signo de z según perspectiva
            z = z_final if sym == 1 else -z_final
            examples.append((state_tensor, pi, z))

    return examples

if __name__ == '__main__':
    # Dispositivo: GPU si está disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Entrenando MCTS+Red en dispositivo: {device}")

    # 1) Inicializar red y optimizador
    net = OthelloNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # 2) Generar datos de self-play
    n_games = 500
    n_sim   = 100
    print(f"Generando {n_games} partidas de self-play con {n_sim} simulaciones cada una…")
    data = self_play(net, n_games=n_games, n_sim=n_sim)

    # 3) Preparar DataLoader
    states = torch.stack([s for s,_,_ in data]).to(device)
    pis    = torch.stack([torch.tensor(pi, dtype=torch.float32) for _,pi,_ in data]).to(device)
    zs     = torch.tensor([z for _,_,z in data], dtype=torch.float32).view(-1,1).to(device)
    dataset = TensorDataset(states, pis, zs)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    # 4) Entrenamiento
    epochs = 10
    for epoch in range(1, epochs+1):
        net.train()
        total_loss = 0.0
        for state, target_pi, target_z in loader:
            optimizer.zero_grad()
            logits, value = net(state)
            value = value.view(-1)              
            target_z = target_z.view(-1)        
            loss_p = - (target_pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss_v = F.mse_loss(value, target_z)
            loss = loss_p + loss_v
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

    # 5) Guardar modelo
    torch.save(net.state_dict(), 'othello_net.pth')
    print("Modelo guardado en othello_net.pth")
