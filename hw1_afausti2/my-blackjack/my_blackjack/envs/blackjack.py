import gym
from gym import spaces
from gym.utils import seeding

""" 
This environment is an adaptation of the Simple blackjack environment included in the standard OpenAi Gym repository.
Their environment corresponds to the version of the blackjack problem in described in section 5.1 of Sutton and 
Barto.
SEE: https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
     http://incompleteideas.net/book/the-book-2nd.html
Changes have been made to better reflect the modern game and allow for scaling to a larger action space and the 
introduction of additional players. Eventually.

In the one-on-one game:
The agent is the player and is dealt both cards face up
The dealer is dealt one card face up and one face down
The observation space is therefore a 3-tuple of: the sum of the player's hand, the dealer's showing card, and whether
the player holds an ace.
The player may choose to hit or stay.
The action space is therefore a discrete space of 2.
Once the player chooses to stay, the hand will be concluded by the dealer playing themselves according to standard
rules, i.e. hit below or at 16, stay at 17, etc.
Reward is then distributed and the hand is over.
Rewards are:
-5 for a loss
1 for a draw
5 for a win
Currently, the deck is infinite with replacement. 
"""
# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


class MyBlackjackEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -5
            else:
                done = False
                reward = 0
        else:  # stay: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = get_reward(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 5:
                reward = 7.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()


def get_reward(player, dealer):
    if player > dealer:
        return 5
    elif player < dealer:
        return -5
    else:
        return 1


def draw_card(np_random):
    return int(np_random.choice(DECK))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]
