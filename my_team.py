# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='ApproxQAgent', second='DefensiveReflexAgent', num_training=0):
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
    return [eval(first)(first_index, num_training=num_training), eval(second)(second_index)]


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

class ApproxQAgent(ReflexCaptureAgent):

    def __init__(self, index, num_training, time_for_computing=.1, lrate=0.01, discount=0.9, epsilon=0.1):
        super().__init__(index, time_for_computing) 
        self.start = None                           # Should I keep this line? When is it used? My guess: yes, bc we need it in register_initial_state
        
        self.weights = util.Counter()          # All weights initialized to 0 (what are the keys of this dict?)
        self.lrate = float(lrate)              # Learning rate, controls how big each weight update is
        self.discount = float(discount)        # Discount, controls how much future rewards matter
        self.epsilon = float(epsilon)          # Epsilon, controls how much agent explores new paths (higher = more exploration)

        # To know whether we are in training mode or testing mode
        self.num_training = int(num_training)  # Number of training games
        self.episodes_so_far = 0                  # Number of games played so far

        self.accum_train_rewards = 0.0
        self.accum_test_rewards = 0.0
        self.episode_rewards = 0.0

        # Need to remember previous action to update weights
        self.prev_action = None

        # Show the hyperparameters of this run
        print(f"ApproxQAgent (index {self.index}): lrate={self.lrate}, epsilon={self.epsilon}, "
              f"discount={self.discount}, num_training={self.num_training}")
        
        # Warm start: initialize weights to reasonable values instead of 0
        self.weights['bias'] = 0
        self.weights['eats_food'] = 1           # eating food is good
        self.weights['distance_to_food'] = -1   # food far away is bad
        self.weights['ghosts_1_step_away'] = -1 # dangerous ghosts close is bad

    def is_in_training(self):
        return self.episodes_so_far < self.num_training

    def get_q_value(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.weights
        return features * weights
    
    def get_features(self, game_state, action):
        """
        Returns a dict of features for the state
        For now, 3 features with values from 0 to 1
        """
        features = util.Counter()
        next_state = self.get_successor(game_state, action)
        food_list = self.get_food(next_state).as_list()
        curr_food_list = self.get_food(game_state).as_list()
        layout_size = game_state.data.layout.width * game_state.data.layout.height

        # Feature 0: constnt bias
        features['bias'] = 1.0
        
        # Feature 1: eats_food
        # Performing action in current state makes Pacman eat a pellet
        curr_food_pellets = len(curr_food_list)
        next_food_pellets = len(food_list)
        features['eats_food'] = 1.0 if next_food_pellets < curr_food_pellets else 0.0


        # the way to increase this feature is to eat more food
        features['next_state_score'] = -len(food_list) / layout_size

        # Feature: number of dangerous ghosts within 1 step of where this action takes me (0, 1 or 2)
        features['ghosts_1_step_away'] = self.count_ghosts_1_step_away(game_state, action)

        # Feature 2: distance to the nearest food, scaled by layout size
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = next_state.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance / layout_size

        return features
           
    
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

        # TODO: complete part C
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

    def get_dist_to_closest_ghost(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if len(ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in ghosts]
            return min(dists)
        
        # No ghosts nearby
        return float("inf")

    def count_ghosts_1_step_away(self, game_state, action):
        """Returns how many visible, non-scared enemy ghosts could reach the
        square that `action` takes me to within one move (0, 1 or 2)."""

        # Compute the square this action leads to directly from my current
        # position. Don't read it from get_successor(): the simulator applies
        # collisions, so if the move walks INTO a ghost my position in the
        # successor is already reset to my spawn, and the most dangerous
        # move of all would look perfectly safe
        my_pos = game_state.get_agent_position(self.index)
        next_x, next_y = Actions.get_successor(my_pos, action)
        next_pos = (int(next_x), int(next_y))

        count = 0
        for opp in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp)
            opp_pos = opp_state.get_position()

            # Can't see it (more than 5 squares away): we can't count it.
            # That's fine, a ghost that far can't reach me in one move anyway
            if opp_pos is None:
                continue

            # An enemy Pacman is on MY side and can't eat me; only ghosts can
            if opp_state.is_pacman:
                continue

            # A scared ghost isn't dangerous (I can eat it instead)
            if opp_state.scared_timer > 0:
                continue

            # Distance <= 1: the ghost is already on that square, or it can step
            # onto it on its next move. Either way I'd get eaten there.
            # (Simplification: this also counts a ghost right across the border
            # when next_pos is still on my side, where it can't actually eat me)
            if self.get_maze_distance(next_pos, opp_pos) <= 1:
                count += 1

        return count

    def detect_ghost_eaten(self, prev_state, curr_state):
        """Returns True if this agent ate a scared enemy ghost between
        `prev_state` (my previous turn) and `curr_state` (my current turn)."""

        # If I died, I didn't eat anyone
        if self.detect_death(prev_state, curr_state):
            return False

        my_curr_pos = curr_state.get_agent_state(self.index).get_position()

        # Check each opponent separately
        for opp in self.get_opponents(prev_state):
            opp_prev = prev_state.get_agent_state(opp)
            opp_curr = curr_state.get_agent_state(opp)
            opp_prev_pos = opp_prev.get_position()

            # 1. In the previous state the enemy was visible
            if opp_prev_pos is None:
                continue

            # 2. The enemy was a scared ghost
            if opp_prev.is_pacman and opp_prev.scared_timer <= 0:
                continue

            # 3. Enemy was 1 square away or less
            if self.get_maze_distance(opp_prev_pos, my_curr_pos) > 1:
                continue

            opp_curr_pos = opp_curr.get_position()

            # 4a. Now i can't see it (it must have teleported to its home)
            if opp_curr_pos is None:
                return True

            # 4b. Still visible, but it moved more than 1 square in one turn:
            #     same idea as detect_death, it was sent back to its spawn
            if self.get_maze_distance(opp_prev_pos, opp_curr_pos) > 1:
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
        ate_ghost = self.detect_ghost_eaten(prev_state, curr_state)
        if ate_ghost:
            reward += 2

        # -2n - 5 Got eaten while carrying n pellets
        i_died = self.detect_death(prev_state, curr_state)
        if i_died:
            base_penalty = 5 # Punish dying, even if you don't carry pellets
            n_lost_pellets = my_prev_state.num_carrying - my_curr_state.num_carrying
            reward -= 2 * n_lost_pellets + base_penalty

        # -0.05 Every step to penalize useless moves
        reward -= 0.05

        return reward
    
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

        # TODO: I don't understand why we have to do one last update
        # This is called only when the game ends right?
        # At this point we training has finished

        # Call the super-class final method
        super().final(state)

        # Reset previous action
        self.prev_action = None

        # print weights & reward here for debugging
        if self.is_in_training():
            print(f"TRAIN - weights: {self.weights}")
            print(f"TRAIN- Reward: {self.episode_rewards}")
            print(f"TRAIN- Accumulated reward: {self.accum_train_rewards}")
        else:
            print(f"TEST - weights: {self.weights}")
            print(f"TEST- Reward: {self.episode_rewards}")
            print(f"TEST- Accumulated reward: {self.accum_test_rewards}")

        self.episodes_so_far += 1
        self.episode_rewards = 0
        
        # If we completed all training games, stop learning
        if not self.is_in_training():
            self.epsilon = 0.0      # Stop exploring, only exploit
            self.lrate = 0.0        # Weights no longer change

