# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import os, json
import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point

# Weights files are loaded / saved in the same directory as my_team.py
# (each agent class defines its own file name)
this_filepath = os.path.abspath(__file__)
FOLDER = os.path.dirname(this_filepath)

# Enemies are only visible within 5 squares, so ghost distances are capped at this
GHOST_VISION = 5

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveApproxQAgent', second='DefensiveApproxQAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    # Both agents are learning agents, so both need to know how many games are training
    return [eval(first)(first_index, num_training=num_training), eval(second)(second_index, num_training=num_training)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

##########################
# Approximate Q-learning #
##########################

class ApproxQAgent(ReflexCaptureAgent):
    """
    Base class for approximate Q-learning agents. It contains everything the
    offensive and defensive agents share: Q-values, epsilon-greedy action choice,
    the weight update, training/testing mode and saving/loading weights.

    Subclasses must define:
        - WEIGHTS_FILENAME: JSON file where their weights are saved/loaded
        - get_initial_weights(): warm start weights
        - get_features(game_state, action)
        - get_reward(prev_state, curr_state)
    """

    WEIGHTS_FILENAME = None    # Each subclass has its own file, since they use different features

    def __init__(self, index, num_training=0, verbose=True, time_for_computing=.1, lrate=0.01, discount=0.9, epsilon=0.1):
        super().__init__(index, time_for_computing)
        self.start = None                      # My spawn position, set in register_initial_state
        self.home_border = []                  # Squares on my side of the border, set in register_initial_state
        self.verbose = verbose
        self.weights_file = os.path.join(FOLDER, self.WEIGHTS_FILENAME)

        self.weights = util.Counter()          # Keys are the feature names, missing keys read as 0
        self.load_weights()                    # Load weights from file, or apply warm start
        self.lrate = float(lrate)              # Learning rate, controls how big each weight update is
        self.discount = float(discount)        # Discount, controls how much future rewards matter
        self.epsilon = float(epsilon)          # Epsilon, controls how much agent explores new paths (higher = more exploration)

        # To know whether we are in training mode or testing mode
        self.num_training = int(num_training)  # Number of training games
        self.episodes_so_far = 0               # Number of games played so far

        self.accum_train_rewards = 0.0
        self.accum_test_rewards = 0.0
        self.episode_rewards = 0.0

        # Need to remember previous action to update weights
        self.prev_action = None

        # If we are in the tournament, don't explore or train
        if not self.is_in_training():
            self.epsilon = 0.0
            self.lrate = 0.0

        # Show the hyperparameters of this run
        if self.verbose:
            print(f"{type(self).__name__} (index {self.index}): lrate={self.lrate}, epsilon={self.epsilon}, "
                  f"discount={self.discount}, num_training={self.num_training}")

    def register_initial_state(self, game_state):
        """Called once at the start of every game (15 seconds allowed)"""
        super().register_initial_state(game_state)   # Sets self.start, self.red and the maze distances

        # The column of my side right next to the border
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        border_x = width // 2 - 1 if self.red else width // 2

        # Every square of that column that isn't a wall: stepping onto one of
        # these means I'm back home (and deposit any food I'm carrying)
        self.home_border = [(border_x, y) for y in range(height) if not game_state.has_wall(border_x, y)]

    ####################
    # Weights save/load #
    ####################

    def get_initial_weights(self):
        """Warm start weights, used when there is no weights file yet"""
        util.raise_not_defined()

    def save_weights(self):
        """Save current weights to a JSON file.
        Creates file if it doesn't exist, otherwise overwrites it."""

        with open(self.weights_file, "w") as f:
            json.dump(dict(self.weights), f,
                      indent=2)             # To make it human-readable

    def load_weights(self):
        """Load weights from a JSON file, or warm start if there is no file"""

        if os.path.exists(self.weights_file):
            if self.verbose:
                print(f"Loading weights from {self.weights_file}")
            with open(self.weights_file, "r") as f:
                weights_dict = json.load(f)
                self.weights = util.Counter(weights_dict)

        else:
            if self.verbose:
                print(f"No weights found at {self.weights_file}, using warm start")
            # Warm start: initialize weights to reasonable values instead of 0
            self.weights = util.Counter(self.get_initial_weights())

    def is_in_training(self):
        return self.episodes_so_far < self.num_training

    ###############################
    # Q-values and action choice #
    ###############################

    def get_features(self, game_state, action):
        """Returns a Counter of features for (state, action). Defined by each subclass."""
        util.raise_not_defined()

    def get_reward(self, prev_state, curr_state):
        """Returns the reward for going from prev_state to curr_state. Defined by each subclass."""
        util.raise_not_defined()

    def get_q_value(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        return features * self.weights

    def compute_action_from_q_values(self, game_state):
        """
          Compute the best action to take in a state, i.e. the action
          with the highest Q-value. If there are no legal actions,
          which is the case at the terminal state, returns None.
        """

        legal_actions = game_state.get_legal_actions(self.index)

        # No legal actions in this state
        if len(legal_actions) == 0:
            return None

        action_to_q_value = util.Counter()

        # Find Q-values of all legal actions
        for action in legal_actions:
            action_to_q_value[action] = self.get_q_value(game_state, action)

        # Break ties: Collect all actions that have the optimal value, and pick one of them at random
        best_value = max(action_to_q_value.values())
        best_actions = [a for a, v in action_to_q_value.items() if v == best_value]

        return random.choice(best_actions)

    def choose_action(self, game_state):

        # If there is a previous state, update weights
        prev_state = self.get_previous_observation()
        if prev_state:
            reward = self.get_reward(prev_state, game_state)
            self.update(prev_state, self.prev_action, game_state, reward)

            # Keep track of how much reward agent is getting
            self.episode_rewards += reward
            if self.is_in_training():
                self.accum_train_rewards += reward
            else:
                self.accum_test_rewards += reward

        # Pick Action
        legal_actions = game_state.get_legal_actions(self.index)

        if len(legal_actions) == 0:
            action = None

        # With prob. epsilon, take a random action
        elif util.flip_coin(p=self.epsilon):
            action = random.choice(legal_actions)

        # With prob. 1-epsilon, take the action with the highest q_value
        else:
            action = self.compute_action_from_q_values(game_state)

        # Remember what we did for next turn
        self.prev_action = action

        return action

    ############
    # Learning #
    ############

    def update(self, state, action, next_state, reward):
        """
           Updates weights based on transition
        """

        # Find Q-value of next state
        best_next_action = self.compute_action_from_q_values(next_state)
        val_next_state = self.get_q_value(next_state, best_next_action)

        q_value_estimate = (reward + self.discount * val_next_state)
        diff = q_value_estimate - self.get_q_value(state, action)

        for feature, feature_val in self.get_features(state, action).items():
            self.weights[feature] += self.lrate * diff * feature_val

    def final(self, state):
        "Called at the end of each game."

        # Call the super-class final method
        super().final(state)

        # Reset previous action
        self.prev_action = None

        # print weights & reward here for debugging
        name = type(self).__name__
        if self.is_in_training():
            print(f"{name} TRAIN - weights: {self.weights}")
            print(f"{name} TRAIN - Reward: {self.episode_rewards}")
            print(f"{name} TRAIN - Accumulated reward: {self.accum_train_rewards}")

            self.save_weights()
            print(f"Saved weights to {self.weights_file}")
        else:
            print(f"{name} TEST - weights: {self.weights}")
            print(f"{name} TEST - Reward: {self.episode_rewards}")
            print(f"{name} TEST - Accumulated reward: {self.accum_test_rewards}")

        self.episodes_so_far += 1
        self.episode_rewards = 0

        # If we completed all training games, stop learning
        if not self.is_in_training():
            self.epsilon = 0.0      # Stop exploring, only exploit
            self.lrate = 0.0        # Weights no longer change

    ##################
    # Shared helpers #
    ##################

    def get_next_position(self, game_state, action):
        """Returns the square `action` takes me to, computed directly from my current position.
        Don't read it from get_successor(): the simulator applies collisions, so if
        the move walks INTO an enemy that eats me, my position in the successor is
        already reset to my spawn, and the most dangerous move would look safe."""

        my_pos = game_state.get_agent_position(self.index)
        next_x, next_y = Actions.get_successor(my_pos, action)
        return (int(next_x), int(next_y))

    def is_on_my_side(self, game_state, pos):
        """True if `pos` is on my team's half of the map (where I'm a ghost).
        (Don't use game_state.is_red() for this: it has a bug with plain (x, y) positions)"""

        width = game_state.data.layout.width
        pos_is_red_side = pos[0] < width // 2
        return pos_is_red_side == self.red

    def get_dist_to_home(self, pos):
        """Maze distance from `pos` to the closest square on my side of the border"""
        return min(self.get_maze_distance(pos, border_pos) for border_pos in self.home_border)

    def detect_death(self, prev_state, curr_state):
        """Returns True if agent got eaten at `prev_state` and
         respawned in its own field at `curr_state`."""

        my_prev_pos = prev_state.get_agent_state(self.index).get_position()
        my_curr_pos = curr_state.get_agent_state(self.index).get_position()

        # Agent moved > 1 square in one turn, that must mean it died
        # and respawned in its own field
        if self.get_maze_distance(my_prev_pos, my_curr_pos) > 1:
            return True
        return False

    def enemy_eaten_next_to_me(self, opp, prev_state, curr_state):
        """Returns True if opponent `opp` was next to where I ended up and was
        then sent back to its spawn. The caller decides which opponents count
        (scared ghosts for offense, invaders for defense)."""

        my_curr_pos = curr_state.get_agent_state(self.index).get_position()
        opp_prev_pos = prev_state.get_agent_state(opp).get_position()
        opp_curr_pos = curr_state.get_agent_state(opp).get_position()

        # 1. In the previous state the enemy was visible
        if opp_prev_pos is None:
            return False

        # 2. Enemy was 1 square away or less from where I ended up:
        #    I stepped onto it, or after my move it stepped onto me
        if self.get_maze_distance(opp_prev_pos, my_curr_pos) > 1:
            return False

        # 3a. Now I can't see it: it was next to me and can only move 1 square
        #     per turn, so it couldn't walk out of view. It was teleported home
        if opp_curr_pos is None:
            return True

        # 3b. Still visible, but it moved more than 1 square in one turn:
        #     same idea as detect_death, it was sent back to its spawn
        return self.get_maze_distance(opp_prev_pos, opp_curr_pos) > 1


class OffensiveApproxQAgent(ApproxQAgent):
    """
    Approximate Q-learning agent that attacks: eats enemy food, avoids
    ghosts and brings the food back home.
    """

    WEIGHTS_FILENAME = "offensive_weights.json"

    def __init__(self, index, carry_limit=5, power_limit=5, **kwargs):
        # Carry limit: once I carry this much food, my target changes from food to home
        self.carry_limit = int(carry_limit)
        # Power limit: a scared ghost is only safe if it stays scared for MORE than
        # this many moves. Below that it's about to recover, so treat it as dangerous
        self.power_limit = int(power_limit)

        super().__init__(index, **kwargs)

    def get_initial_weights(self):
        # All distance features use the same scale (divided by layout size),
        # so these weights can be compared directly with each other
        return {
            'bias': 0.0,
            'successor_score': 1.0,        # eating food is good
            'ghosts_1_step_away': -5.0,    # a ghost next to me means I'm about to die
            'distance_to_ghost': 1.0,      # being far from ghosts is good
            'distance_to_food': -1.0,      # far from my target (food, or home) is bad
            'is_home': 2.0,                # bringing food home is good
        }

    def get_features(self, game_state, action):
        """
        Returns a dict of features for the state, all roughly between 0 and 1
        (except ghosts_1_step_away, which is 0, 1 or 2)
        """
        features = util.Counter()
        next_state = self.get_successor(game_state, action)
        next_pos = self.get_next_position(game_state, action)
        layout_size = game_state.data.layout.width * game_state.data.layout.height

        food_list = self.get_food(next_state).as_list()
        curr_food_list = self.get_food(game_state).as_list()
        carrying = game_state.get_agent_state(self.index).num_carrying

        # Go home when I carry enough food (carry limit), or when there's
        # nothing left to win by eating (the game ends at 2 pellets left)
        going_home = carrying >= self.carry_limit or len(curr_food_list) <= 2

        # Feature 0: constant bias
        features['bias'] = 1.0

        # Feature 1: number of dangerous ghosts within 1 step of where this action takes me (0, 1 or 2)
        features['ghosts_1_step_away'] = self.count_ghosts_1_step_away(game_state, next_pos)

        # Feature 2: distance to the closest dangerous ghost, scaled by layout size.
        # Capped at GHOST_VISION: further than that, or no ghost in sight, all look the same
        features['distance_to_ghost'] = self.get_dist_to_closest_ghost(game_state, next_pos) / layout_size

        # Feature 3: successor_score: this action eats a pellet.
        # Not if a ghost is next to it: that pellet isn't worth dying for, and
        # this way "food is good" and "ghosts are bad" don't fight each other
        eats_food = len(food_list) < len(curr_food_list)
        if eats_food and features['ghosts_1_step_away'] == 0:
            features['successor_score'] = 1.0

        # Feature 4: distance to my target, scaled by layout size.
        # Normally the target is the closest food. Over the carry limit it's home instead,
        # so the same "get closer to the target" weight now pulls me back to deposit
        if going_home:
            target_distance = self.get_dist_to_home(next_pos)
        elif len(food_list) > 0:
            target_distance = min([self.get_maze_distance(next_pos, food) for food in food_list])
        else:
            target_distance = 0
        features['distance_to_food'] = target_distance / layout_size

        # Feature 5: is_home: this action brings me back to my side while carrying food.
        # Being home with nothing to deposit isn't useful, so it only counts when carrying
        if carrying > 0 and self.is_on_my_side(game_state, next_pos):
            features['is_home'] = 1.0

        return features

    def get_dangerous_ghosts(self, game_state):
        """Returns the positions of visible enemy ghosts that can eat me"""

        ghost_positions = []
        for opp in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp)
            opp_pos = opp_state.get_position()

            # Can't see it (more than 5 squares away)
            if opp_pos is None:
                continue

            # An enemy Pacman is on MY side and can't eat me; only ghosts can
            if opp_state.is_pacman:
                continue

            # Scared for long enough: safe to approach (power limit)
            if opp_state.scared_timer > self.power_limit:
                continue

            ghost_positions.append(opp_pos)

        return ghost_positions

    def count_ghosts_1_step_away(self, game_state, next_pos):
        """Returns how many dangerous ghosts could reach `next_pos` within one move (0, 1 or 2)"""

        # On my own side I'm a ghost, enemy ghosts can't eat me there
        if self.is_on_my_side(game_state, next_pos):
            return 0

        # Distance <= 1: the ghost is already on that square, or it can step
        # onto it on its next move. Either way I'd get eaten there
        ghosts = self.get_dangerous_ghosts(game_state)
        return sum(1 for ghost_pos in ghosts if self.get_maze_distance(next_pos, ghost_pos) <= 1)

    def get_dist_to_closest_ghost(self, game_state, next_pos):
        """Maze distance from `next_pos` to the closest dangerous ghost, capped at GHOST_VISION"""

        ghosts = self.get_dangerous_ghosts(game_state)

        # On my side ghosts can't hurt me, and with no ghost in sight there's
        # nothing to run from: both count as "as far as I can see"
        if self.is_on_my_side(game_state, next_pos) or len(ghosts) == 0:
            return GHOST_VISION

        closest = min(self.get_maze_distance(next_pos, ghost_pos) for ghost_pos in ghosts)
        return min(closest, GHOST_VISION)

    def detect_ghost_eaten(self, prev_state, curr_state):
        """Returns True if this agent ate a scared enemy ghost between
        `prev_state` (my previous turn) and `curr_state` (my current turn)."""

        # If I died, I didn't eat anyone
        if self.detect_death(prev_state, curr_state):
            return False

        for opp in self.get_opponents(prev_state):
            opp_prev = prev_state.get_agent_state(opp)

            # Only a scared ghost can be eaten by Pacman
            # (note the OR: skip it if it's a Pacman, or if it isn't scared)
            if opp_prev.is_pacman or opp_prev.scared_timer <= 0:
                continue

            if self.enemy_eaten_next_to_me(opp, prev_state, curr_state):
                return True

        # No opponent satisfied all the conditions
        return False

    def get_reward(self, prev_state, curr_state):
        my_prev_state = prev_state.get_agent_state(self.index)
        my_curr_state = curr_state.get_agent_state(self.index)

        reward = 0

        # +1 Ate a pellet
        if my_curr_state.num_carrying > my_prev_state.num_carrying:
            reward += 1

        # +2n Deposited n pellets (depositing more gives higher reward)
        n_deposited = my_curr_state.num_returned - my_prev_state.num_returned
        reward += 2 * n_deposited

        # +2 Eat a scared ghost
        if self.detect_ghost_eaten(prev_state, curr_state):
            reward += 2

        # -2n - 5 Got eaten while carrying n pellets
        if self.detect_death(prev_state, curr_state):
            base_penalty = 5 # Punish dying, even if you don't carry pellets
            n_lost_pellets = my_prev_state.num_carrying - my_curr_state.num_carrying
            reward -= 2 * n_lost_pellets + base_penalty

        # -0.05 Every step to penalize useless moves
        reward -= 0.05

        return reward


class DefensiveApproxQAgent(ApproxQAgent):
    """
    Approximate Q-learning agent that defends: guards the middle of the border,
    chases invaders and keeps its distance from them while scared.
    """

    WEIGHTS_FILENAME = "defensive_weights.json"

    def __init__(self, index, **kwargs):
        self.middle = None    # Where I guard when no invader is visible, set in register_initial_state
        super().__init__(index, **kwargs)

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)   # Also computes self.home_border

        # The middle of the map is usually a wall or on the enemy side, so guard
        # the square of my border that's closest to the vertical center instead
        height = game_state.data.layout.height
        self.middle = min(self.home_border, key=lambda pos: abs(pos[1] - height / 2))

    def get_initial_weights(self):
        # All distance features use the same scale (divided by layout size),
        # so these weights can be compared directly with each other
        return {
            'bias': 0.0,
            'num_invaders': -1.0,          # every invader on my side is bad (eating one lowers this)
            'distance_to_middle': -1.0,    # stay near the middle of the border when nothing's happening
            'distance_to_invader': -1.0,   # chase the closest invader
            'scared_distance': 1.0,        # while scared, keep away from invaders (they can eat me)
        }

    def get_features(self, game_state, action):
        """
        Returns a dict of features for the state, all roughly between 0 and 1
        (except num_invaders, which is 0, 1 or 2)
        """
        features = util.Counter()
        next_state = self.get_successor(game_state, action)
        next_pos = self.get_next_position(game_state, action)
        layout_size = game_state.data.layout.width * game_state.data.layout.height
        am_scared = game_state.get_agent_state(self.index).scared_timer > 0

        # Feature 0: constant bias
        features['bias'] = 1.0

        # Feature 1: number of visible invaders after this action.
        # Uses the successor, so an action that eats an invader lowers this number
        features['num_invaders'] = len(self.get_invader_positions(next_state))

        # Distances are measured from next_pos to where invaders are NOW
        # (if I'm scared and walk into one, the successor already moved me to my spawn)
        invaders = self.get_invader_positions(game_state)

        if len(invaders) == 0:
            # Feature 2: no invader in sight, patrol: distance to the middle of my border
            features['distance_to_middle'] = self.get_maze_distance(next_pos, self.middle) / layout_size

        else:
            closest = min(self.get_maze_distance(next_pos, invader_pos) for invader_pos in invaders)

            if am_scared:
                # Feature 4: scared_distance: I'm scared, the invader can eat me, keep away
                features['scared_distance'] = closest / layout_size
            else:
                # Feature 3: distance_to_invader: chase the closest invader
                features['distance_to_invader'] = closest / layout_size

        return features

    def get_invader_positions(self, game_state):
        """Returns the positions of visible invaders (enemy Pacmen on my side)"""

        positions = []
        for opp in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp)
            opp_pos = opp_state.get_position()

            # Visible and a Pacman (i.e. on my side of the map)
            if opp_pos is not None and opp_state.is_pacman:
                positions.append(opp_pos)

        return positions

    def get_reward(self, prev_state, curr_state):
        reward = 0
        i_died = self.detect_death(prev_state, curr_state)

        # +5 for each invader I ate
        if not i_died:
            for opp in self.get_opponents(prev_state):
                if prev_state.get_agent_state(opp).is_pacman and \
                        self.enemy_eaten_next_to_me(opp, prev_state, curr_state):
                    reward += 5

        # -1 for each of our pellets the enemies ate.
        # max(0, ...) because when an invader dies, the food it carried is dropped
        # back onto the board, and that shouldn't count as a reward
        prev_food = len(self.get_food_you_are_defending(prev_state).as_list())
        curr_food = len(self.get_food_you_are_defending(curr_state).as_list())
        reward -= max(0, prev_food - curr_food)

        # -5 Got eaten (only possible while I'm scared)
        if i_died:
            reward -= 5

        # No per-step penalty here: the defender isn't trying to finish anything
        # quickly, and a constant reward every step would only shift the bias weight

        return reward
