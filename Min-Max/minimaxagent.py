import copy
import math
import random

from exceptions import AgentException


class MinimaxAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        score_move_list = self.minmax(connect4, 1, True, 7, -math.inf, math.inf)
        return score_move_list[1]

    def minmax(self, connect4, depth, is_maximizing, max_depth, alpha, beta):
        if self.my_token == 'o':
            opponent_token = 'x'
        else:
            opponent_token = 'o'

        if connect4.check_game_over():

            if connect4.wins == self.my_token:
                return [10000, None]
            elif connect4.wins == opponent_token:
                return [-10000, None]
            else:
                return [0, None]
        elif depth >= max_depth:
            score = 0
            score += 3 * connect4.center_column().count(self.my_token)
            fours = connect4.iter_fours()
            for four in fours:
                my = four.count(self.my_token)
                op = four.count(opponent_token)
                if my == 3:
                    score += 5
                if my == 2:
                    score += 2
                if op == 3:
                    score += 4
            return [score, None]

        if is_maximizing:
            best_score = -math.inf
            best_move = None
            moves = connect4.possible_drops()
            for move in moves:
                board_copy = copy.deepcopy(connect4)
                board_copy.drop_token(move)
                score_move_list = self.minmax(board_copy, depth + 1, False, max_depth, alpha, beta)
                if score_move_list[0] > best_score:
                    best_score = score_move_list[0]
                    best_move = move
                alpha = max(best_score, alpha)
                if alpha >= beta:
                    break

            return [best_score, best_move]

        else:
            best_score = math.inf
            best_move = None
            moves = connect4.possible_drops()
            for move in moves:
                board_copy = copy.deepcopy(connect4)
                board_copy.drop_token(move)
                score_move_list = self.minmax(board_copy, depth + 1, True, max_depth, alpha, beta)
                if score_move_list[0] < best_score:
                    best_score = score_move_list[0]
                    best_move = move

                beta = min(beta, best_score)
                if alpha >= beta:
                    break
            return [best_score, best_move]
