import sys
import mapel

from abcvoting.preferences import Profile
from abcvoting import abcrules
import time
import math

if __name__ == "__main__":


    v = [0.2, 0.3, 0.5]

    v = [math.log(1/p) for p in v]

    print(v)