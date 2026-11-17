
#Tic Tac Toe Player
import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    
    #Returns starting state of the board.
    
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    
    #Returns player who has the next turn on a board.
  
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count == o_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return {(i, j) for i in range(3) for j in range(3)
            if board[i][j] == EMPTY}


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise ValueError("Invalid action")

    i, j = action
    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    lines = []
    lines.extend(board)                                              # rows
    lines.extend([[board[i][j] for i in range(3)] for j in range(3)])  # columns
    lines.append([board[i][i] for i in range(3)])                    # diagonal
    lines.append([board[i][2 - i] for i in range(3)])                # anti-diagonal

    for line in lines:
        if line[0] is not EMPTY and line[0] == line[1] == line[2]:
            return line[0]
    return None


def terminal(board):
   
    #Returns True if game is over, False otherwise.
   
    return winner(board) is not None or not actions(board)


def utility(board):
    
    #Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
   
    win = winner(board)
    if win == X:
        return 1
    if win == O:
        return -1
    return 0


def max_value(board, alpha, beta):
 
    #Returns the best utility X (maximizer) can guarantee from this board.
  
    if terminal(board):
        return utility(board)

    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action), alpha, beta))
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return v


def min_value(board, alpha, beta):
   
    #Returns the best utility O (minimizer) can guarantee from this board.
    
    if terminal(board):
        return utility(board)

    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action), alpha, beta))
        beta = min(beta, v)
        if alpha >= beta:
            break
    return v


def minimax(board):
  
    #Returns the optimal action for the current player on the board.
    
    if terminal(board):
        return None

    current = player(board)
    best_action = None

    if current == X:
        best = -math.inf
        for action in actions(board):
            value = min_value(result(board, action), best, math.inf)
            if value > best:
                best, best_action = value, action
    else:
        best = math.inf
        for action in actions(board):
            value = max_value(result(board, action), -math.inf, best)
            if value < best:
                best, best_action = value, action

    return best_action