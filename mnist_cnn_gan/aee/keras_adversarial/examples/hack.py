import pandas as pd
import numpy as np

xtrain = np.ones(60000)

mask = np.ones(len(xtrain), dtype=bool)

mask[[2,3,4,5]] = False

xtrain = np.delete(xtrain,np.s_[9999:59999:],0)

print(xtrain)
print(xtrain.shape)
