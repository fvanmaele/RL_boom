import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
import pickle
import copy

from settings import s
from settings import e

# Function modified from items.py
def get_blast_coords(arena, x, y):
    """Retrieve the blast range for a bomb.

    The maximal power of the bomb (maximum range in each direction) is
    imported directly from the game settings. The blast range is
    adjusted according to walls (immutable obstacles) in the game
    arena.

    Parameters:
    * arena:  2-dimensional array describing the game arena.
    * x, y:   Coordinates of the bomb.

    Return Value:
    * Array containing each coordinate of the bomb's blast range.
    """
    bomb_power = s.bomb_power
    blast_coords = [(x, y)]

    for i in range(1, bomb_power+1):
        if arena[x+i, y] == -1: break
        blast_coords.append((x+i, y))
    for i in range(1, bomb_power+1):
        if arena[x-i, y] == -1: break
        blast_coords.append((x-i, y))
    for i in range(1, bomb_power+1):
        if arena[x, y+i] == -1: break
        blast_coords.append((x, y+i))
    for i in range(1, bomb_power+1):
        if arena[x, y-i] == -1: break
        blast_coords.append((x, y-i))

    return blast_coords


# Function modified from simple_agent to return the path towards a
# target, instead of only the first step.
def look_for_targets_path(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until
    a target is encountered.  If no target can be reached, the path
    that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        the path towards closest target or towards tile closest to any
        target, beginning at the next step.
    """
    if len(targets) == 0:
        return []

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)

        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break

        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    if logger:
        logger.debug(f'Suitable target found at {best}')

    # Determine the path towards the best found target tile, start not included
    current = best
    path = []
    while True:
        path.insert(0, current)
        if parent_dict[current] == start:
            return path
        current = parent_dict[current]


def look_for_targets(free_space, start, targets, logger=None):
    """Returns the coordinate of first step towards closest target, or
    towards tile closest to any target.
    """
    path = look_for_targets_path(free_space, start, targets, logger=None)

    if len(path):
        return path[0]
    else:
        return None


############# FEATURES ##############

class RLFeatureExtraction:
    def __init__(self, game_state):
        """
        Extract relevant properties from the environment for feature
        extraction.
        """
        # The actions set here determine the order of the columns in
        # the feature matrix.
        # TODO: Pass this list as an argument
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

        # Set the amount of features
        self.dim = 10

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


    def __call__(self):
        """
        Compute the feature matrix F, where every column represents an
        action A, and every row a feature F_i(S, A).
        """
        # TODO: Cache results for later calls
        f0 = np.ones(6) # for bias
        print("f0 ", f0)
        f1 = self.feature1()
        print("f1 ", f1)
        f2 = self.feature2()
        print("f2 ", f2)
        f3 = self.feature3()
        print("f3 ", f3)
        f4 = self.feature4()
        print("f4 ", f4)
        f5 = self.feature5()
        print("f5 ", f5)
        f6 = self.feature6()
        print("f6 ", f6)
        f7 = self.feature7()
        print("f7 ", f7)
        # TODO: Set crate limits depending on crate density!
        f8 = self.feature8(3, 8)
        print("f8 ", f8)
        f9 = self.feature9()
        print("f9 ", f9)
        f10 = self.feature10()
        print("f10 ", f10)
        f11 = self.feature11(3, 8)
        print("f11 ", f11)
        # f12 = self.feature12()
        # print("f12 ", f12)

        return np.vstack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11)).T


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
                    if d == self.agent: continue
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
                    if d == self.agent: continue
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
