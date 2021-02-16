import random
import math

def categorical_draw(probs):
  z = random.random()
  cum_prob = 0.0
  for i in range(len(probs)):
    prob = probs[i]
    cum_prob += prob
    if cum_prob > z:
      return i

  return len(probs) - 1

class MyExp3():
  def __init__(self, l_rate, weights, probs):
    self.weights = weights
    self.l_rate = l_rate
    self.probs = probs
    return
  
  def initialize(self, n_arms):
    self.weights = [0.0 for i in range(n_arms)]
    self.probs = [1/n_arms for i in range(n_arms)]
    return
  
  def select_arm(self):
    return categorical_draw(self.probs)
  
  def update(self, chosen_arm, reward):
    n_arms = len(self.weights)
    for arm in range(n_arms):
        if arm == chosen_arm:
            self.weights[arm] = self.weights[arm] + 1 - ((1-reward)/self.probs[arm])
        else:
            self.weights[arm] = self.weights[arm] + 1
            
    new_weights = [self.l_rate*weight for weight in self.weights]
    partial_probs = [math.exp(weight) for weight in new_weights]
    total_prob = sum(partial_probs)
    self.probs = [prob/total_prob for prob in partial_probs]
