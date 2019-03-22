import numpy as np
from random import shuffle

from agent_code.my_agent.arena import *
from settings import s
from settings import e


class RLFeatureExtraction:
    def __init__(self, game_state, coin_limit=2, crate_limit=6):
        """
        Extract relevant properties from the environment for feature
        extraction.
        """
        # The actions set here determine the order of the columns in the returned
        # feature matrix. (Take as an argument?)
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
        # Set the amount of features / weights
        self.dim = 12

        # Collect commonly used data from the environment.
        self.arena = game_state['arena']
        self.bombs = game_state['bombs']
        self.coins = game_state['coins']
        self.explosions = game_state['explosions']
        self.others = game_state['others']
        self.x, self.y, self.name, self.bombs_left = game_state['self']

        # Some methods only require the coordinates of bombs and
        # opposing agents.
        self.bombs_xy = [(x,y) for (x,y,t) in self.bombs]
        self.others_xy = [(x,y) for (x,y,n,b) in self.others]

        # Map actions to coordinates. Placing a bomb or waiting does
        # not move the agent. Other actions assume that the origin
        # (0,0) is located at the top left of the arena.
        self.agent = (self.x, self.y)
        self.directions = {
            'UP'   : (self.x, (self.y)-1),
            'DOWN' : (self.x, (self.y)+1),
            'LEFT' : ((self.x)-1, self.y),
            'RIGHT': ((self.x)+1, self.y),
            'BOMB' : self.agent,
            'WAIT' : self.agent
        }

        # Check the arena (np.array) for free tiles, and include the comparison
        # result as a boolean np.array.
        self.free_space = self.arena == 0
        # Do not include agents as obstacles, as they are likely to
        # move in the next round.
        # for xb, yb, _ in self.bombs:
        #     self.free_space[xb, yb] = False

        # Observation: look_for_targets requires that any targets are considered
        # free space. (feature 10)
        # for xc, yc in self.crates:
        #     self.free_space[xc, yc] = True

        # The blast range (discounted for walls) of a bomb is only available if the
        # bomb has already exploded. We thus compute the range manually.
        self.danger_zone = []
        if len(self.bombs) != 0:
            for xb, yb, _ in self.bombs:
                self.danger_zone += get_blast_coords(self.arena, xb, yb)

        # Define a "safe zone" of all free tiles in the arena, which are not part of
        # the above danger zone.
        self.safe_zone = [(x, y) for x in range(1, 16) for y in range(1, 16)
                          if (self.arena[x, y] == 0)
                          and (x, y) not in self.danger_zone]

        # Compute a list of all crates in the arena.
        self.crates = [(x,y) for x in range(1,16) for y in range(1,16)
                       if (self.arena[x,y] == 1)]

        # Compute dead ends, i.e. tiles with only a single neighboring, free
        # tile. Only crates and walls are taken into account; opposing agents and
        # bombs are ignored: moving the agent towards these targets may impose a
        # negative reward, and should be left to other features.
        self.dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16)
                          if (self.arena[x, y] == 0)
                          and ([self.arena[x+1, y],
                                self.arena[x-1, y],
                                self.arena[x, y+1],
                                self.arena[x, y-1]].count(0) == 1)]

        # bomb_map gives the maximum blast range of a bomb, if walls are not taken
        # into account. The values in this array are set to the timer for each
        # bomb. (taken from simple_agent)
        self.bomb_map = np.ones(self.arena.shape) * 5
        for xb, yb, t in self.bombs:
            for (i, j) in [(xb+h, yb) for h in range(-3, 4)] + [(xb, yb+h) for h in range(-3, 4)]:
                if (0 < i < self.bomb_map.shape[0]) and (0 < j < self.bomb_map.shape[1]):
                    self.bomb_map[i, j] = min(self.bomb_map[i, j], t)                    

        # Compute the feature matrix with columns F_i(S, A) and rows ordered by the
        # actions defined in self.actions.
        self.feature = np.vstack(
            ([1] * len(self.actions),
             self.feature1(),
             self.feature2(),
             self.feature3(),
             self.feature4(),
             self.feature5(),
             self.feature6(),
             self.feature7(),
             self.feature8(coin_limit, crate_limit),
             self.feature9(),
             self.feature10(),
             self.feature11(coin_limit, crate_limit))).T


    def state(self):
        """
        Return the feature matrix F, where every column represents an
        a feature F_i(S,A), and rows represent actions A.
        """
        return feature


    def state_action(self, action):
        """
        Return the column vector for the feature:
           F(S, A) = F_1(S,A) ... F_n(S,A)
        """
        return feature[self.actions.index(action)]


    def max_q(self, weights):
        """
        Return the maximum Q-value for all possible actions, and the corresponding
        action to this maximum. It may be used to update weights during training, or
        to implement a greedy policy. The required weights are assumed known, and
        taken as a parameter.
        """
        # Compute the dot product (w, F_i(S,A)) for every action.
        Q_lfa = np.dot(weights, self.feature)            
        Q_max = np.max(Q_lfa)

        # Multiple actions may give the same (optimal) reward. To avoid bias towards
        # a particular action, shuffle the resulting index array before return it.
        A_max = np.where(Q_lfa == Q_max)[0]
        A_max = shuffle(A_max)

        return Q_max, [self.actions[a] for a in A_max]


    def feature1(self):
        """
        Reward the agent to move in a direction towards a coin.
        """
        feature = []

        # Check if there are coins available in the game.
        if len(self.coins) == 0:
            return [0] * len(self.actions)
        best_direction = look_for_targets(self.free_space, self.agent, self.coins)

        # Check if the next move action matches the direction of the nearest coin.
        for action in self.actions:
            if action == 'BOMB' or action == 'WAIT':
                # Placing a bomb or waiting does not allow the agent to collect a coin.
                feature.append(0)
            else:
                if self.directions[action] == best_direction:
                    feature.append(1)
                else:
                    feature.append(0)

        return feature


    def feature2(self):
        """
        Penalize the action if it places the agent into a location
        where it is most likely to die.
        """
        feature = []

        # TODO: 'BOMB' and 'WAIT' are computed twice here
        for action in self.actions:
            d = self.directions[action]

            # Check if the tile reached by the next action is occupied by an
            # object. (Opposing agents may wait, thus we should check them even if
            # they can move away.) This object may be destroyed by bombs, but
            # prevents us from moving into a free tile.
            if (self.arena[d] != 0) or (d in self.others_xy) or (d in self.bombs_xy):
                d = self.agent

            # We first check if the agent moves into the blast range of a bomb which
            # will explode directly after. The second condition checks if the agent
            # moves into an ongoing explosion. In both cases, such a movement causes
            # certain death for the agent (that is, we set F_i(s, a) = 1).
            if (d in self.danger_zone) and (self.bomb_map[d] == 0) or (self.explosions[d] > 1):
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature3(self):
        """Penalize the agent for going or remaining into an area threatened
        by a bomb.

        The logic used in this feature is very similar to feature2. The
        main difference is that we consider any bomb present in the arena,
        not only those that will explode in the next step.
        """
        feature = []

        # TODO: 'BOMB' and 'WAIT' are computed twice here
        for action in self.actions:
            d = self.directions[action]

            # Check if the tile reached by the next action is occupied by an
            # object. If so, only consider the current location of the agent.
            if (self.arena[d] != 0) or (d in self.others_xy) or (d in self.bombs_xy):
                d = self.agent
            if d in self.danger_zone:
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature4(self):
        """
        Reward the agent for moving in the shortest direction outside
        the blast range of (all) bombs in the game.
        """
        feature = []

        # Check if the arena contains any bombs with a blast radius affecting the agent.
        if len(self.bombs) == 0 or (self.agent not in self.danger_zone):
            return [0] * len(self.actions)

        # Check if the agent can move into a safe area.
        if len(self.safe_zone) == 0:
            return [0] * len(self.actions)

        safety_direction = look_for_targets(self.free_space, self.agent, self.safe_zone)

        for action in self.actions:
            d = self.directions[action]

            if action == 'BOMB':
                # When the agent is retreating from one or several bombs, we do not
                # wish to expand the danger zone by dropping a bomb ourselves.
                feature.append(0)
            elif d == safety_direction:
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature5(self):
        """
        Penalize the agent taking an invalid action.
        """
        feature = []

        for action in self.actions:
            d = self.directions[action]

            if (action == 'WAIT'):
                # We should check explicitely if the agent is waiting; when dropping
                # a bomb, the agent may remain in the same tile until either the
                # bomb explodes, or the agent takes a move action (after which he
                # may longer move to the tile containing the placed bomb).
                feature.append(0)
            elif (action == 'BOMB') and (self.bombs_left == 0):
                # An agent may only place a bomb if it has any remaining.
                feature.append(1)
            elif (self.arena[d] != 0) or (d in self.others_xy) or (d in self.bombs_xy):
                # When checking other objects than walls (immutable), we make the
                # following observations regarding invalid actions. Which agent
                # moves first is decided randomly; an initially free square may thus
                # be later occupied by an agent. Crates may be destroyed by a
                # ticking bomb, but this is done only after all agents have
                # performed their respective agents.
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature6(self):
        """
        Reward the agent for collecting a coin.
        """
        feature = []

        for action in self.actions:
            d = self.directions[action]

            if action == 'BOMB' or action == 'WAIT':
                feature.append(0)
            elif d in self.coins:
                feature.append(1)
            else:
                feature.append(0)

        return feature


    def feature7(self):
        """
        Reward the agent for placing a bomb next to a crate.
        """
        feature = []

        for action in self.actions:
            if action == 'BOMB' and self.bombs_left > 0:
                CHECK_FOR_CRATE = False
                for d in self.directions.values():
                    if d == self.agent:
                        continue
                    if self.arena[d] == 1:
                        CHECK_FOR_CRATE = True
                        break

                if CHECK_FOR_CRATE:
                    feature.append(1)
                else:
                    feature.append(0)
            else:
                feature.append(0)

        return feature


    def feature8(self, coins_limit, crates_limit):
        """Hunting mode

        Reward the agent for placing a bomb next to an opponent.
        """
        feature = []

        if len(self.coins) > coins_limit or len(self.crates) > crates_limit:
            return [0] * len(self.actions)

        for action in self.actions:
            if action == 'BOMB' and self.bombs_left > 0:
                CHECK_FOR_OTHERS = False
                for d in self.directions.values():
                    if d == self.agent:
                        continue
                    if d in self.others_xy:
                        CHECK_FOR_OTHERS = True
                        break

                if CHECK_FOR_OTHERS:
                    feature.append(1)
                else:
                    feature.append(0)
            else:
                feature.append(0)

        return feature


    # TODO: Rearding moving towards a dead end might encourage the agent to walk
    # into a trap by other agents...
    def feature9(self):
        """
        Reward the agent taking a step towards a dead end (a tile with
        only a single free, neighboring tile).
        """
        feature = []
        best_direction = look_for_targets(self.free_space, self.agent, self.dead_ends)
        print("PATH TO DEAD END", look_for_targets_path(self.free_space, self.agent, self.dead_ends), sep=" ")

        # Do not reward if the agent is already in a dead-end, or if there
        # are none in the arena.
        if self.agent in self.dead_ends or best_direction is None:
            return [0] * len(self.actions)

        for action in self.actions:
            if action == 'BOMB' or action == 'WAIT':
                # Only a move action can move the agent towards a dead end.
                feature.append(0)
            else:
                d = self.directions[action]

                if d == best_direction:
                    feature.append(1)
                else:
                    feature.append(0)

        return feature


    def feature10(self):
        """
        Reward the agent for going towards a crate.
        """
        feature = []

        # Check if crates are available in the game.
        if len(self.crates) == 0:
            return [0] * len(self.actions)

        best_direction = look_for_targets(self.free_space, self.agent, self.crates)

        # If we are directly next to a create, look_for_targets will
        # return the tile where the agent is located in, rewarding an
        # (unnecessary) wait action.
        # if best_direction == self.agent:
        #     return np.zeros(6)

        for action in self.actions:
            if action == 'BOMB' or action == 'WAIT':
                # The feature rewards movement towards a crate. In
                # particular, placing a bomb or waiting is not given a
                # reward.
                feature.append(0)
            else:
                d = self.directions[action]

                # A-priori, blowing up one crate is not better than
                # blowing up another. We thus give no reward for any
                # movement if the agent is directly next to a crate.
                if d in self.crates:
                    return [0] * len(self.actions)

                if d == best_direction:
                    feature.append(1)
                else:
                    feature.append(0)

        return feature


    def feature11(self, coins_limit, crates_limit):
        """Hunting mode

        Reward moving towards opposing agents with less than a certain
        amount of coins and crates inside the game arena.
        """
        feature = []

        if len(self.coins) > coins_limit or len(self.crates) > crates_limit:
            return [0] * len(self.actions)

        best_direction = look_for_targets(self.free_space, self.agent, self.others_xy)

        for action in self.actions:
            if action == 'BOMB' or action == 'WAIT':
                # The feature rewards movement towards an agent. In particular,
                # placing a bomb or waiting is given no reward.
                feature.append(0)
            else:
                d = self.directions[action]

                # We do not get more or less points for blowing up one given agent
                # over any other. Therefore, do not reward moving to a different
                # agent if there is already one in direct vicinity.
                if d in self.others_xy:
                    return [0] * len(self.actions)

                if d == best_direction:
                    feature.append(1)
                else:
                    feature.append(0)

        return feature

