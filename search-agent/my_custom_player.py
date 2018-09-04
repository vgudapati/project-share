
from sample_players import DataPlayer
import random


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        
        #import random
        #self.queue.put(random.choice(state.actions()))
        
        depth = 1
        while 1:
            #self.queue.put(random.choice(state.actions()))
            #self.queue.put(self.alpha_beta_search(state, depth))
            self.queue.put(self.principal_variation_search(state, depth))
            depth += 1

    def alpha_beta_search(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        actions = state.actions()
        if actions:
            best_move = actions[0]
        else:
            best_move = None
        maximizingPlayer = True
        for a in actions:
            new_state = state.result(a)
            v = self._alpha_beta_min_max(
                new_state, depth-1, alpha, beta, maximizingPlayer)
            if v > alpha:
                alpha = v
                best_move = a
        return best_move

    def _alpha_beta_min_max(self, state, depth, alpha, beta, maximizingPlayer):
        #print("Maximizing Player: ", maximizingPlayer)
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        if maximizingPlayer:
            v = -float('inf')
            for action in state.actions():
                new_state = state.result(action)
                v = max(v, self._alpha_beta_min_max(
                    new_state, depth-1, alpha, beta, False))
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v
        else:
            v = float('inf')
            for action in state.actions():
                new_state = state.result(action)
                v = min(v, self._alpha_beta_min_max(
                    new_state, depth-1, alpha, beta, True))
                beta = min(beta, v)
                if alpha >= beta:
                    break
            return v

    def principal_variation_search(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        actions = state.actions()
        if actions:
            best_move = actions[0]
        else:
            best_move = None
        maximizingPlayer = True
        v = -float('inf')
        for i, action in enumerate(actions):
            new_state = state.result(action)
            if i == 0:
                v = max(v, self._pvs_min_max(
                    new_state, depth-1, alpha, beta, maximizingPlayer))
            else:
                v = max(v, self._pvs_min_max(
                    new_state, depth-1, alpha, alpha+1, maximizingPlayer))
                if v > alpha:
                    v = max(v, self._pvs_min_max(
                        new_state, depth-1, alpha, beta, maximizingPlayer))
            if v > alpha:
                alpha = v
                best_move = action
        return best_move

    def _pvs_min_max(self, state, depth, alpha, beta, maximizingPlayer):
        #print("Maximizing Player: ", maximizingPlayer)
        
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        if maximizingPlayer:
            v = -float('inf')
            for i, action in enumerate(state.actions()):
                new_state = state.result(action)
                if i == 0:
                    v = max(v, self._pvs_min_max(
                        new_state, depth-1, alpha, beta, False))
                else:
                    v = max(v, self._pvs_min_max(
                        new_state, depth-1, alpha, alpha+1, False))
                    if v > alpha:
                        v = max(v, self._pvs_min_max(
                            new_state, depth-1, alpha, beta, False))
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v
        else:
            v = float('inf')
            for i, action in enumerate(state.actions()):
                new_state = state.result(action)
                if i == 0:
                    v = min(v, self._pvs_min_max(
                        new_state, depth-1, alpha, beta, True))
                else:
                    v = min(v, self._pvs_min_max(
                        new_state, depth-1, beta-1, beta, True))
                    if v < beta:
                        v = min(v, self._pvs_min_max(
                            new_state, depth-1, alpha, beta, True))
                beta = min(beta, v)
                if alpha >= beta:
                    break
            return v

    def score(self, state):
        self_location = state.locs[self.player_id]
        opponent_location = state.locs[1 - self.player_id]
        self_liberties = state.liberties(self_location)
        opponent_liberties = state.liberties(opponent_location)
        return len(self_liberties) - len(opponent_liberties)

