import time
import numpy as np

start_time = time.time()

# generate matrix
a = np.random.rand(10000,10000)
b = np.random.rand(10000,10000)

# dot
result = np.dot(a, b)

end_time = time.time()

print(f"time: {end_time-start_time}")