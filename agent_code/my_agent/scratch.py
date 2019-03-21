# TODO: The performance of this feature appears lacking in spite of obvious
# nature... maybe due to look_for_targets_path function?
def feature12(self):
    """
    Penalize the agent for moving directly into a dead end, if a bomb was placed
    as the previous action.
    """
    path_dead_end = look_for_targets_path(self.free_space, self.agent, self.dead_ends)
    feature = []

    # Check if there are dead ends in the arena.
    if len(path_dead_end) == 0:
        return [0] * len(self.actions)

    for action in self.actions:
        if action == 'BOMB' or action == 'WAIT':
            # Note that placing a bomb if one has been placed previously (the
            # scenario envisioned by this feature), that the agent performs an
            # invalid agent. However, this is already covered by a different
            # feature.
            feature.append(0)
        elif len(path_dead_end) == 1:
            d = self.directions[action]

            # If a bomb was placed in the previous step, it is in the same tile
            # the agent is located in.
            if d == path_dead_end[0] and self.agent in self.bombs_xy:
                feature.append(1)
            else:
                feature.append(0)
        else:
            # Do not penalize a longer path towards a dead end.
            feature.append(0)

    return feature
