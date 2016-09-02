import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randint(2, size=(15, 14)))

cols = df.columns.tolist()

df_counts = df.apply(pd.value_counts).iloc[1]
max_value = df_counts.idxmax()



cols_more_pars = df.apply(pd.value_counts).iloc[1].idxmax()
print (df.apply(pd.value_counts).iloc[1])

cols.insert(0, cols.pop(cols_more_pars))
