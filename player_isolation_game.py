
import time
from isolation import Board
import random


class OpenMoveEvalFn:
    def score(self, game, my_player=None):
        """Score the current game state
        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.

        Note:
            Basic Evaluation function. CustomEvalFn below has better evaluation function

            Args
                game (Board): The board and game state.
                my_player (Player object): This specifies which player you are.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """

        X = len(game.get_player_moves(my_player))
        Y = len(game.get_opponent_moves(my_player))
        if X == 0 and Y !=0 :
            return float("-inf")
        if X !=0 and Y==0:
            return float("inf")
        if X==0 and Y==0:
            return -10
        return X-(Y)
        

class CustomPlayer:
    
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=9, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.

        For a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Evaluation function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        print("Entered custom init")

    def move(self, game, time_left):
        """Called to determine one move by your agent

        Note:
            1. Call alphabeta or minimax depending on algorithm you want to use.
        Args:
            game (Board): The board and game state.
            time_left (function): Used to determine time left before timeout

        Returns:
            tuple: (int,int): Your best move
        """

        best_move, utility = alphabeta(self, game, time_left, depth=self.search_depth,alpha=float("-inf"), beta=float("inf"),my_turn=True)
        #best_move,utility = minimax(self,game,time_left,depth=self.search_depth,my_turn=True)
        return best_move

    def utility(self, game, my_turn):
        """You can handle special cases here (e.g. endgame)"""
        return self.eval_fn.score(game, self)


def minimax(player, game, time_left, depth, my_turn=True):
    """Implementation of the minimax algorithm.

    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you
            need from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer()).
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """

    def max_val(game,depth,time_left):
        if time_left()<20 or depth==0:
            return player.utility(game,1)
        max_best_val = float("-inf")
        moves = game.get_active_moves()
        for move in moves:
            max_game_subbranch = game.forecast_move(move)[0]
            max_score = min_val(max_game_subbranch,depth-1,time_left)
            if max_score > max_best_val:
                max_best_val = max_score
        return max_best_val

    def min_val(game,depth,time_left):
        if time_left()<20 or depth==0:
            return player.utility(game,1)
        min_best_val = float("inf")
        moves = game.get_active_moves()
        for move in moves:
            min_game_subbranch = game.forecast_move(move)[0]
            min_score = max_val(min_game_subbranch,depth-1,time_left)
            if min_score < min_best_val:
                min_best_val = min_score
        return min_best_val


    moves = game.get_active_moves()

    if my_turn:
        if not moves:
            return None,float("-inf")

        best_val = float("-inf")
        for move in moves:
            game_subbranch = game.forecast_move(move)[0]
            score          = min_val(game_subbranch,depth-1,time_left)
            if score>=best_val:
                best_val = score
                best_move = move
    else:
        if not moves:
            return None,float("inf")

        best_val = float("inf")
        for move in moves:
            game_subbranch = game.forecast_move(move)[0]
            score          = max_val(game_subbranch,depth-1,time_left)
            if score<=best_val:
                best_val = score
                best_move = move

    return best_move,best_val


####Interative deepening is a better version of alpha beta algorithm below#####
def id_ab(player,game,time_left,depth, alpha=float("-inf"), beta=float("inf"), my_turn=True):

    moves = game.get_active_moves()

    #Terminal nodes. Game reached the end as either of the player has no moves
    if not moves:
        if my_turn:
            return  None,float("-inf")
        else:
            return  None,float("inf")

    #sort moves according to their utility values
    #for next_move in moves:
    #    moves_dic = {}
    #    next_move_val = player.utility(game.forecast_move(next_move)[0],1)
    #    moves_dic.update({next_move:next_move_val})
    #sorted_moves = sorted((value,key) for (key,value) in moves_dic.items())

    #best_move = (sorted_moves[len(sorted_moves)-1])[1]
    #best_val =  (sorted_moves[len(sorted_moves)-1])[0]

    for i in range(1,depth):
        if time_left()<20:
            return best_move,best_val
        best_move,best_val = alphabeta(player,game,time_left,depth=i,alpha=float("-inf"), beta=float("inf"),my_turn=my_turn)
    
    return best_move,best_val

def alphabeta(player, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), my_turn=True):
    """Implementation of the alphabeta algorithm.

    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you need
            from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer())
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        alpha (float): Alpha value for pruning
        beta (float): Beta value for pruning
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """

    def max_val(game,depth,time_left,alpha,beta):
        if time_left()<30 or depth==0:
            return player.utility(game,1)
        max_best_val = float("-inf")
        moves = game.get_active_moves()
        for move in moves:
            max_game_subbranch = game.forecast_move(move)[0]
            max_score = min_val(max_game_subbranch,depth-1,time_left,alpha,beta)
            if max_score > max_best_val:
                max_best_val = max_score
            if max_best_val >= beta:
                return max_best_val
            alpha = max(alpha,max_best_val)
        return max_best_val

    def min_val(game,depth,time_left,alpha,beta):
        if time_left()<30 or depth==0:
            return player.utility(game,1)
        min_best_val = float("inf")
        moves = game.get_active_moves()
        for move in moves:
            min_game_subbranch = game.forecast_move(move)[0]
            min_score = max_val(min_game_subbranch,depth-1,time_left,alpha,beta)
            if min_score < min_best_val:
                min_best_val = min_score
            if min_best_val <= alpha:
                return min_best_val
            beta = min(beta,min_best_val)
        return min_best_val


    moves = game.get_active_moves()


    if my_turn:
        if not moves:
            return None,float("-inf")

        best_val = float("-inf")
        for move in moves:
            game_subbranch = game.forecast_move(move)[0]
            score          = min_val(game_subbranch,depth-1,time_left,alpha,beta)
            if score>=best_val:
                best_val = score
                best_move = move
            if score >= beta:
                return best_move, best_val
            alpha = max(alpha,score)
    else:
        if not moves:
            return None,float("inf")

        best_val = float("inf")
        for move in moves:
            game_subbranch = game.forecast_move(move)[0]
            score          = max_val(game_subbranch,depth-1,time_left,alpha,beta)
            if score<=best_val:
                best_val = score
                best_move = move
            if score <= alpha:
                return best_move,best_val
            beta = min(beta,score)

    return best_move,best_val


class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, my_player=None):
        """Score the current game state.

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args:
            game (Board): The board and game state.
            my_player (Player object): This specifies which player you are.

        Returns:
            float: The current state's score, based on your own heuristic.
        """

        X = len(game.get_player_moves(my_player))
        Y = (len(game.get_opponent_moves(my_player)))
        if X == 0 and Y !=0 :
            return float("-inf")
        if X !=0 and Y==0:
            return float("inf")
        if X==0 and Y==0:
            return -5
        return X-(Y)
        