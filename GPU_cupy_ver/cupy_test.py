import time
import cupy as cp

start_time = time.time()

# generate matrix
a = cp.random.rand(10000,10000)
b = cp.random.rand(10000,10000)

# dot
result = cp.dot(a, b)

end_time = time.time()

print(f"time: {end_time-start_time}")