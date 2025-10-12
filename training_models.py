import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("./datasets/arhv_prepro_5.0.csv")
pd.set_option('display.max_columns', None)

print(df.describe())


