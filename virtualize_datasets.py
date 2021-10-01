import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv("./data/train.csv")

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.