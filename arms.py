import math
import random


class NormalArm:
    # Initializer for the NormalArm class with mean and standard deviation for the normal distribution.
    def __init__(self, mu: float, sigma: float):
        self.mu = mu  # Mean reward for the arm.
        self.__sigma = sigma  # Standard deviation of the reward for the arm.

    # Simulate drawing a reward for this arm based on its normal distribution.
    def draw(self):
        """Returns the achieved reward of the arm at this round."""
        return random.gauss(self.mu, self.__sigma)  # Draw and return a sample from the normal distribution.


class StrategicArm(NormalArm):
    # Class variables for the minimum and maximum values among all costs/bids.
    # These values are used to normalize costs/bids across all StrategicArm instances.
    c_min, c_max = 1., 0.1

    # Initializer for the StrategicArm class, which models an arm with strategic behavior.
    def __init__(self):
        # Randomly determine the expected reward for the arm.
        r = random.uniform(0.1, 1)  # Expected reward between 0.1 and 1.
        # Randomly determine the variance of the reward, ensuring it is a fraction of the reward.
        sigma = random.uniform(0, min(r / 3, (1 - r) / 3))  # Variance is constrained relative to the reward.
        # Initialize the base NormalArm class with the determined reward and variance.
        super().__init__(r, sigma)

        # Randomly determine the cost for using this arm.
        self.c = random.uniform(0.1, 1)  # Cost is between 0.1 and 1.
        # Set the bid to the cost, assuming truthfulness as per Theorem 2 from the paper.
        self.b = self.c  # Bid is equal to the cost.

        # Update the class variables c_min and c_max based on the cost of this instance.
        # This ensures that c_min and c_max reflect the minimum and maximum costs across all instances.
        StrategicArm.c_min = min(StrategicArm.c_min, self.c)
        StrategicArm.c_max = max(StrategicArm.c_max, self.c)
