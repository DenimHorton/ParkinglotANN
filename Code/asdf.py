import numpy as np


p = [[5, 3, 1],[-1, -1, -1], [-1, 100, 0]]
print(np.array(p))
print(np.max(p))

print(p)

q = [3, 2]
h = [2, 3]

print((q[0], q[1]) in p)
print(h in p)