# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''Module to manage and advanced game state'''
from collections import defaultdict

import numpy as np

from . import constants
from . import characters
from . import utility


class ForwardModel(object):
    """Class for helping with the [forward] modeling of the game state."""

    def run(self,
            num_times,
            board,
            agents,
            bombs,
            items,
            flames,
            is_partially_observable,
            agent_view_size,
            action_space,
            training_agent=[],
            is_communicative=False):
        """Run the forward model.

        Args:
          num_times: The number of times to run it for. This is a maximum and
            it will stop early if we reach a done.
          board: The board state to run it from.
          agents: The agents to use to run it.
          bombs: The starting bombs.
          items: The starting items.
          flames: The starting flames.
          is_partially_observable: Whether the board is partially observable or
            not. Only applies to TeamRadio.
          agent_view_size: If it's partially observable, then the size of the
            square that the agent can view.
          action_space: The actions that each agent can take.
          training_agent: The training agent to pass to done.
          is_communicative: Whether the action depends on communication
            observations as well.

        Returns:
          steps: The list of step results, which are each a dict of "obs",
            "next_obs", "reward", "action".
          board: Updated board.
          agents: Updated agents, same models though.
          bombs: Updated bombs.
          items: Updated items.
          flames: Updated flames.
          done: Whether we completed the game in these steps.
          info: The result of the game if it's completed.
        """
        steps = []
        for _ in num_times:
            obs = self.get_observations(
                board, agents, bombs, is_partially_observable, agent_view_size)
            actions = self.act(
                agents, obs, action_space, is_communicative=is_communicative)
            board, agents, bombs, items, flames = self.step(
                actions, board, agents, bombs, items, flames)
            next_obs = self.get_observations(
                board, agents, bombs, is_partially_observable, agent_view_size)
            reward = self.get_rewards(agents, game_type, step_count, max_steps)
            done = self.get_done(agents, game_type, step_count, max_steps,
                                 training_agent)
            info = self.get_info(done, rewards, game_type, agents)

            steps.append({
                "obs": obs,
                "next_obs": next_obs,
                "reward": reward,
                "actions": actions,
            })
            if done:
                # Callback to let the agents know that the game has ended.
                for agent in agents:
                    agent.episode_end(reward[agent.agent_id])
                break
        return steps, board, agents, bombs, items, flames, done, info

    @staticmethod
    def act(agents, obs, action_space, is_communicative=False):
        """Returns actions for each agent in this list.

        Args:
          agents: A list of agent objects.
          obs: A list of matching observations per agent.
          action_space: The action space for the environment using this model.
          is_communicative: Whether the action depends on communication
            observations as well.

        Returns a list of actions.
        """

        def act_ex_communication(agent):
            '''Handles agent's move without communication'''
            if agent.is_alive:
                return agent.act(obs[agent.agent_id], action_space=action_space)
            else:
                return constants.Action.Stop.value

        def act_with_communication(agent):
            '''Handles agent's move with communication'''
            if agent.is_alive:
                action = agent.act(
                    obs[agent.agent_id], action_space=action_space)
                if type(action) == int:
                    action = [action] + [0, 0]
                assert (type(action) == list)
                return action
            else:
                return [constants.Action.Stop.value, 0, 0]

        ret = []
        for agent in agents:
            if is_communicative:
                ret.append(act_with_communication(agent))
            else:
                ret.append(act_ex_communication(agent))
        return ret

    @staticmethod
    def step(actions,
             curr_board,
             curr_agents,
             curr_bombs,
             curr_items,
             curr_flames,
             max_blast_strength=10):
        board_size = len(curr_board)

        # Tick the flames. Replace any dead ones with passages. If there is an
        # item there, then reveal that item.
        flames = []
        for flame in curr_flames:
            position = flame.position
            if flame.is_dead():
                item_value = curr_items.get(position)
                if item_value:
                    del curr_items[position]
                else:
                    item_value = constants.Item.Passage.value
                curr_board[position] = item_value
            else:
                flame.tick()
                flames.append(flame)
        curr_flames = flames

        # Redraw all current flames
        # Multiple flames may share a position and the map should contain
        # a flame until all flames are dead to avoid issues with bomb
        # movements and explosions.
        for flame in curr_flames:
            curr_board[flame.position] = constants.Item.Flames.value

        # Step the living agents and moving bombs.
        # If two agents try to go to the same spot, they should bounce back to
        # their previous spots. This is complicated with one example being when
        # there are three agents all in a row. If the one in the middle tries
        # to go to the left and bounces with the one on the left, and then the
        # one on the right tried to go to the middle one's position, she should
        # also bounce. A way of doing this is to gather all the new positions
        # before taking any actions. Then, if there are disputes, correct those
        # disputes iteratively.
        # Additionally, if two agents try to switch spots by moving into each
        # Figure out desired next position for alive agents
        alive_agents = [agent for agent in curr_agents if agent.is_alive]
        desired_agent_positions = [agent.position for agent in alive_agents]

        for num_agent, agent in enumerate(alive_agents):
            position = agent.position
            # We change the curr_board here as a safeguard. We will later
            # update the agent's new position.
            curr_board[position] = constants.Item.Passage.value
            action = actions[agent.agent_id]

            if action == constants.Action.Stop.value:
                pass
            elif action == constants.Action.Bomb.value:
                position = agent.position
                if not utility.position_is_bomb(curr_bombs, position):
                    bomb = agent.maybe_lay_bomb()
                    if bomb:
                        curr_bombs.append(bomb)
            elif utility.is_valid_direction(curr_board, position, action):
                desired_agent_positions[num_agent] = agent.get_next_position(
                    action)

        # Gather desired next positions for moving bombs. Handle kicks later.
        desired_bomb_positions = [bomb.position for bomb in curr_bombs]

        for num_bomb, bomb in enumerate(curr_bombs):
            curr_board[bomb.position] = constants.Item.Passage.value
            if bomb.is_moving():
                desired_position = utility.get_next_position(
                    bomb.position, bomb.moving_direction)
                if utility.position_on_board(curr_board, desired_position) \
                   and not utility.position_is_powerup(curr_board, desired_position) \
                   and not utility.position_is_wall(curr_board, desired_position):
                    desired_bomb_positions[num_bomb] = desired_position

        # Position switches:
        # Agent <-> Agent => revert both to previous position.
        # Bomb <-> Bomb => revert both to previous position.
        # Agent <-> Bomb => revert Bomb to previous position.
        crossings = {}

        def crossing(current, desired):
            '''Checks to see if an agent is crossing paths'''
            current_x, current_y = current
            desired_x, desired_y = desired
            if current_x != desired_x:
                assert current_y == desired_y
                return ('X', min(current_x, desired_x), current_y)
            assert current_x == desired_x
            return ('Y', current_x, min(current_y, desired_y))

        for num_agent, agent in enumerate(alive_agents):
            if desired_agent_positions[num_agent] != agent.position:
                desired_position = desired_agent_positions[num_agent]
                border = crossing(agent.position, desired_position)
                if border in crossings:
                    # Crossed another agent - revert both to prior positions.
                    desired_agent_positions[num_agent] = agent.position
                    num_agent2, _ = crossings[border]
                    desired_agent_positions[num_agent2] = alive_agents[
                        num_agent2].position
                else:
                    crossings[border] = (num_agent, True)

        for num_bomb, bomb in enumerate(curr_bombs):
            if desired_bomb_positions[num_bomb] != bomb.position:
                desired_position = desired_bomb_positions[num_bomb]
                border = crossing(bomb.position, desired_position)
                if border in crossings:
                    # Crossed - revert to prior position.
                    desired_bomb_positions[num_bomb] = bomb.position
                    num, is_agent = crossings[border]
                    if not is_agent:
                        # Crossed bomb - revert that to prior position as well.
                        desired_bomb_positions[num] = curr_bombs[num].position
                else:
                    crossings[border] = (num_bomb, False)

        # Deal with multiple agents or multiple bomb collisions on desired next
        # position by resetting desired position to current position for
        # everyone involved in the collision.
        agent_occupancy = defaultdict(int)
        bomb_occupancy = defaultdict(int)
        for desired_position in desired_agent_positions:
            agent_occupancy[desired_position] += 1
        for desired_position in desired_bomb_positions:
            bomb_occupancy[desired_position] += 1

        # Resolve >=2 agents or >=2 bombs trying to occupy the same space.
        change = True
        while change:
            change = False
            for num_agent, agent in enumerate(alive_agents):
                desired_position = desired_agent_positions[num_agent]
                curr_position = agent.position
                # Either another agent is going to this position or more than
                # one bomb is going to this position. In both scenarios, revert
                # to the original position.
                if desired_position != curr_position and \
                      (agent_occupancy[desired_position] > 1 or bomb_occupancy[desired_position] > 1):
                    desired_agent_positions[num_agent] = curr_position
                    agent_occupancy[curr_position] += 1
                    change = True

            for num_bomb, bomb in enumerate(curr_bombs):
                desired_position = desired_bomb_positions[num_bomb]
                curr_position = bomb.position
                if desired_position != curr_position and \
                      (bomb_occupancy[desired_position] > 1 or agent_occupancy[desired_position] > 1):
                    desired_bomb_positions[num_bomb] = curr_position
                    bomb_occupancy[curr_position] += 1
                    change = True

        # Handle kicks.
        agent_indexed_by_kicked_bomb = {}
        kicked_bomb_indexed_by_agent = {}
        delayed_bomb_updates = []
        delayed_agent_updates = []

        # Loop through all bombs to see if they need a good kicking or cause
        # collisions with an agent.
        for num_bomb, bomb in enumerate(curr_bombs):
            desired_position = desired_bomb_positions[num_bomb]

            if agent_occupancy[desired_position] == 0:
                # There was never an agent around to kick or collide.
                continue

            agent_list = [
                (num_agent, agent) for (num_agent, agent) in enumerate(alive_agents) \
                if desired_position == desired_agent_positions[num_agent]]
            if not agent_list:
                # Agents moved from collision.
                continue

            # The agent_list should contain a single element at this point.
            assert (len(agent_list) == 1)
            num_agent, agent = agent_list[0]

            if desired_position == agent.position:
                # Agent did not move
                if desired_position != bomb.position:
                    # Bomb moved, but agent did not. The bomb should revert
                    # and stop.
                    delayed_bomb_updates.append((num_bomb, bomb.position))
                continue

            # NOTE: At this point, we have that the agent in question tried to
            # move into this position.
            if not agent.can_kick:
                # If we move the agent at this point, then we risk having two
                # agents on a square in future iterations of the loop. So we
                # push this change to the next stage instead.
                delayed_bomb_updates.append((num_bomb, bomb.position))
                delayed_agent_updates.append((num_agent, agent.position))
                continue

            # Agent moved and can kick - see if the target for the kick never had anyhing on it
            direction = constants.Action(actions[agent.agent_id])
            target_position = utility.get_next_position(desired_position,
                                                        direction)
            if utility.position_on_board(curr_board, target_position) and \
                       agent_occupancy[target_position] == 0 and \
                       bomb_occupancy[target_position] == 0 and \
                       not utility.position_is_powerup(curr_board, target_position) and \
                       not utility.position_is_wall(curr_board, target_position):
                # Ok to update bomb desired location as we won't iterate over it again here
                # but we can not update bomb_occupancy on target position and need to check it again
                # However we need to set the bomb count on the current position to zero so
                # that the agent can stay on this position.
                bomb_occupancy[desired_position] = 0
                delayed_bomb_updates.append((num_bomb, target_position))
                agent_indexed_by_kicked_bomb[num_bomb] = num_agent
                kicked_bomb_indexed_by_agent[num_agent] = num_bomb
                bomb.moving_direction = direction
                # Bombs may still collide and we then need to reverse bomb and agent ..
            else:
                delayed_bomb_updates.append((num_bomb, bomb.position))
                delayed_agent_updates.append((num_agent, agent.position))

        for (num_bomb, bomb_position) in delayed_bomb_updates:
            desired_bomb_positions[num_bomb] = bomb_position
            bomb_occupancy[bomb_position] += 1
            change = True

        for (num_agent, agent_position) in delayed_agent_updates:
            desired_agent_positions[num_agent] = agent_position
            agent_occupancy[agent_position] += 1
            change = True

        while change:
            change = False
            for num_agent, agent in enumerate(alive_agents):
                desired_position = desired_agent_positions[num_agent]
                curr_position = agent.position
                # Agents and bombs can only share a square if they are both in their
                # original position (Agent dropped bomb and has not moved)
                if desired_position != curr_position and \
                      (agent_occupancy[desired_position] > 1 or bomb_occupancy[desired_position] != 0):
                    # Late collisions resulting from failed kicks force this agent to stay at the
                    # original position. Check if this agent successfully kicked a bomb above and undo
                    # the kick.
                    if num_agent in kicked_bomb_indexed_by_agent:
                        num_bomb = kicked_bomb_indexed_by_agent[num_agent]
                        bomb = curr_bombs[num_bomb]
                        desired_bomb_positions[num_bomb] = bomb.position
                        bomb_occupancy[bomb.position] += 1
                        del agent_indexed_by_kicked_bomb[num_bomb]
                        del kicked_bomb_indexed_by_agent[num_agent]
                    desired_agent_positions[num_agent] = curr_position
                    agent_occupancy[curr_position] += 1
                    change = True

            for num_bomb, bomb in enumerate(curr_bombs):
                desired_position = desired_bomb_positions[num_bomb]
                curr_position = bomb.position

                # This bomb may be a boomerang, i.e. it was kicked back to the
                # original location it moved from. If it is blocked now, it
                # can't be kicked and the agent needs to move back to stay
                # consistent with other movements.
                if desired_position == curr_position and num_bomb not in agent_indexed_by_kicked_bomb:
                    continue

                bomb_occupancy_ = bomb_occupancy[desired_position]
                agent_occupancy_ = agent_occupancy[desired_position]
                # Agents and bombs can only share a square if they are both in their
                # original position (Agent dropped bomb and has not moved)
                if bomb_occupancy_ > 1 or agent_occupancy_ != 0:
                    desired_bomb_positions[num_bomb] = curr_position
                    bomb_occupancy[curr_position] += 1
                    num_agent = agent_indexed_by_kicked_bomb.get(num_bomb)
                    if num_agent is not None:
                        agent = alive_agents[num_agent]
                        desired_agent_positions[num_agent] = agent.position
                        agent_occupancy[agent.position] += 1
                        del kicked_bomb_indexed_by_agent[num_agent]
                        del agent_indexed_by_kicked_bomb[num_bomb]
                    change = True

        for num_bomb, bomb in enumerate(curr_bombs):
            if desired_bomb_positions[num_bomb] == bomb.position and \
               not num_bomb in agent_indexed_by_kicked_bomb:
                # Bomb was not kicked this turn and its desired position is its
                # current location. Stop it just in case it was moving before.
                bomb.stop()
            else:
                # Move bomb to the new position.
                # NOTE: We already set the moving direction up above.
                bomb.position = desired_bomb_positions[num_bomb]

        for num_agent, agent in enumerate(alive_agents):
            if desired_agent_positions[num_agent] != agent.position:
                agent.move(actions[agent.agent_id])
                if utility.position_is_powerup(curr_board, agent.position):
                    agent.pick_up(
                        constants.Item(curr_board[agent.position]),
                        max_blast_strength=max_blast_strength)

        # Explode bombs.
        exploded_map = np.zeros_like(curr_board)
        has_new_explosions = False

        for bomb in curr_bombs:
            bomb.tick()
            if bomb.exploded():
                has_new_explosions = True
            elif curr_board[bomb.position] == constants.Item.Flames.value:
                bomb.fire()
                has_new_explosions = True

        # Chain the explosions.
        while has_new_explosions:
            next_bombs = []
            has_new_explosions = False
            for bomb in curr_bombs:
                if not bomb.exploded():
                    next_bombs.append(bomb)
                    continue

                bomb.bomber.incr_ammo()
                for _, indices in bomb.explode().items():
                    for r, c in indices:
                        if not all(
                            [r >= 0, c >= 0, r < board_size, c < board_size]):
                            break
                        if curr_board[r][c] == constants.Item.Rigid.value:
                            break
                        exploded_map[r][c] = 1
                        if curr_board[r][c] == constants.Item.Wood.value:
                            break

            curr_bombs = next_bombs
            for bomb in curr_bombs:
                if bomb.in_range(exploded_map):
                    bomb.fire()
                    has_new_explosions = True

        # Update the board's bombs.
        for bomb in curr_bombs:
            curr_board[bomb.position] = constants.Item.Bomb.value

        # Update the board's flames.
        flame_positions = np.where(exploded_map == 1)
        for row, col in zip(flame_positions[0], flame_positions[1]):
            curr_flames.append(characters.Flame((row, col)))
        for flame in curr_flames:
            curr_board[flame.position] = constants.Item.Flames.value

        # Kill agents on flames. Otherwise, update position on curr_board.
        for agent in alive_agents:
            if curr_board[agent.position] == constants.Item.Flames.value:
                agent.die()
            else:
                curr_board[agent.position] = utility.agent_value(agent.agent_id)

        return curr_board, curr_agents, curr_bombs, curr_items, curr_flames

    def get_observations(self, curr_board, agents, bombs, flames,
                         is_partially_observable, agent_view_size,
                         game_type, game_env):
        """Gets the observations as an np.array of the visible squares.

        The agent gets to choose whether it wants to keep the fogged part in
        memory.
        """
        board_size = len(curr_board)

        def make_bomb_maps(position):
            ''' Makes an array of an agents bombs and the bombs attributes '''
            blast_strengths = np.zeros((board_size, board_size))
            life = np.zeros((board_size, board_size))
            moving_direction = np.zeros((board_size, board_size))

            for bomb in bombs:
                x, y = bomb.position
                if not is_partially_observable \
                   or in_view_range(position, x, y):
                    blast_strengths[(x, y)] = bomb.blast_strength
                    life[(x, y)] = bomb.life
                    if bomb.moving_direction is not None:
                        moving_direction[(x, y)] = bomb.moving_direction.value
            return blast_strengths, life, moving_direction

        def make_flame_map(position):
            ''' Makes an array of an agents flame life'''
            life = np.zeros((board_size, board_size))

            for flame in flames:
                x, y = flame.position
                if not is_partially_observable \
                   or in_view_range(position, x, y):
                    # +1 needed because flame removal check is done
                    # before flame is ticked down, i.e. flame life
                    # in environment is 2 -> 1 -> 0 -> dead
                    life[(x, y)] = flame.life + 1
            return life

        def in_view_range(position, v_row, v_col):
            '''Checks to see if a tile is in an agents viewing area'''
            row, col = position
            return all([
                row >= v_row - agent_view_size, row <= v_row + agent_view_size,
                col >= v_col - agent_view_size, col <= v_col + agent_view_size
            ])

        attrs = [
            'position', 'blast_strength', 'can_kick', 'teammate', 'ammo',
            'enemies'
        ]
        alive_agents = [
            utility.agent_value(agent.agent_id)
            for agent in agents
            if agent.is_alive
        ]

        observations = []
        for agent in agents:
            agent_obs = {'alive': alive_agents}
            board = curr_board.copy()
            if is_partially_observable:
                for row in range(board_size):
                    for col in range(board_size):
                        if not in_view_range(agent.position, row, col):
                            board[row, col] = constants.Item.Fog.value
            agent_obs['board'] = board
            bomb_blast_strengths, bomb_life, bomb_moving_direction = make_bomb_maps(agent.position)
            agent_obs['bomb_blast_strength'] = bomb_blast_strengths
            agent_obs['bomb_life'] = bomb_life
            agent_obs['bomb_moving_direction'] = bomb_moving_direction
            flame_life = make_flame_map(agent.position)
            agent_obs['flame_life'] = flame_life
            agent_obs['game_type'] = game_type.value
            agent_obs['game_env'] = game_env

            for attr in attrs:
                assert hasattr(agent, attr)
                agent_obs[attr] = getattr(agent, attr)
            observations.append(agent_obs)

        return observations

    @staticmethod
    def get_done(agents, step_count, max_steps, game_type, training_agent):
        alive = [agent for agent in agents if agent.is_alive]
        alive_ids = sorted([agent.agent_id for agent in alive])
        if step_count >= max_steps:
            return True
        elif game_type == constants.GameType.FFA or game_type == constants.GameType.OneVsOne:
            if training_agent is not None:
                for agent_id in training_agent:
                    if agent_id in alive_ids:
                        return False
                return True
            return len(alive) <= 1
        elif any([
                len(alive_ids) <= 1,
                alive_ids == [0, 2],
                alive_ids == [1, 3],
        ]):
            return True
        return False

    @staticmethod
    def get_info(done, rewards, game_type, agents):
        if game_type == constants.GameType.FFA or game_type == constants.GameType.OneVsOne:
            alive = [agent for agent in agents if agent.is_alive]
            if done:
                if len(alive) != 1:
                    # Either we have more than 1 alive (reached max steps) or
                    # we have 0 alive (last agents died at the same time).
                    return {
                        'result': constants.Result.Tie,
                    }
                else:
                    return {
                        'result': constants.Result.Win,
                        'winners': [num for num, reward in enumerate(rewards) \
                                    if reward == 1]
                    }
            else:
                return {
                    'result': constants.Result.Incomplete,
                }
        elif done:
            # We are playing a team game.
            if rewards == [-1] * 4:
                return {
                    'result': constants.Result.Tie,
                }
            else:
                return {
                    'result': constants.Result.Win,
                    'winners': [num for num, reward in enumerate(rewards) \
                                if reward == 1],
                }
        else:
            return {
                'result': constants.Result.Incomplete,
            }

    @staticmethod
    def get_rewards(agents, game_type, step_count, max_steps):

        def any_lst_equal(lst, values):
            '''Checks if list are equal'''
            return any([lst == v for v in values])

        alive_agents = [num for num, agent in enumerate(agents) \
                        if agent.is_alive]
        if game_type == constants.GameType.FFA:
            if len(alive_agents) == 1:
                # An agent won. Give them +1, others -1.
                return [2 * int(agent.is_alive) - 1 for agent in agents]
            elif step_count >= max_steps:
                # Game is over from time. Everyone gets -1.
                return [-1] * 4
            else:
                # Game running: 0 for alive, -1 for dead.
                return [int(agent.is_alive) - 1 for agent in agents]
        elif game_type == constants.GameType.OneVsOne:
            if len(alive_agents) == 1:
                # An agent won. Give them +1, the other -1.
                return [2 * int(agent.is_alive) - 1 for agent in agents]
            elif step_count >= max_steps:
                # Game is over from time. Everyone gets -1.
                return [-1] * 2
            else:
                # Game running
                return [0, 0]
        else:
            # We are playing a team game.
            if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
                # Team [0, 2] wins.
                return [1, -1, 1, -1]
            elif any_lst_equal(alive_agents, [[1, 3], [1], [3]]):
                # Team [1, 3] wins.
                return [-1, 1, -1, 1]
            elif step_count >= max_steps:
                # Game is over by max_steps. All agents tie.
                return [-1] * 4
            elif len(alive_agents) == 0:
                # Everyone's dead. All agents tie.
                return [-1] * 4
            else:
                # No team has yet won or lost.
                return [0] * 4
