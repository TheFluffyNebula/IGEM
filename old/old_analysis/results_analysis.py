import os
import pandas as pd
import matplotlib.pyplot as plt
import re

PATH = "eval_results/"
results = {file: pd.read_csv(PATH + file) for file in os.listdir(PATH)}
col_names = {file : cols.keys().tolist() for file, cols in results.items()}
#print(col_names)
pattern = r".*Projection.*"
filtered_results = {}
for file, text_list in col_names.items():
    for text in text_list:
        match_object = re.search(pattern, text)
        if match_object:
            s = match_object.group()
            filtered_results[file] = s

igem_pattern = r"gem.*sgem.csv"
gem_pattern = r"gem.*[^sgem].csv"


methods = ["I-GEM cifar", "GEM mnist", "GEM cifar", "I-GEM mnist"]
time = []
print(results)
# for file, df in results.items():
#     key = filtered_results[file]
#     # 1) show exactly what we're working with
#     #print("Available columns:", [repr(c) for c in df.columns])

#     # 2) find the real key
#     matches = [c for c in df.columns if c == key]
#     if not matches:
#         continue

#     real_key = matches[0]
#     # 3) index with the real key
#     print(file, df[real_key])
    

# projection overhead plot
# plt.bar(methods, time, color='red')
# plt.xlabel('Method')
# plt.ylabel('Time')
# plt.title('Projection Overhead')
# plt.show()
