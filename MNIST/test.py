import ray
from ray.util.actor_pool import ActorPool
import numpy as np
import pandas as pd
# ray.init()

# @ray.remote
# class Test():
#     def test(self, a):
#         return a+1

# dd = [10,11,12,13]
# pool = ActorPool([Test.remote() for i in range(4)])
# print(list(pool.map(lambda a,v: a.test.remote(v), dd)))
# ray.shutdown()

# list = [[1,2,3], [4,5,6]]
# ap = [7,8,9]
# list.append(ap)
# tb = np.array(list)
# df = pd.DataFrame(tb, columns=['11', '22', '33'])
# print(df)

import pickle

with open('Store_datatype_1.pickle', 'rb') as files:
    data = pickle.load(files)
print(data)