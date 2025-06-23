import pandas as pd
import os
import matplotlib.pyplot as plt
PATH = 'eval_results/'

metrics = ["Projection", "Top1_Acc"]

files = os.listdir(PATH)
metric_data = {}
for file in files:
    for metric in metrics:
        df = pd.read_csv(PATH + file)
        matches = [k for k in df.keys() if metric in k]
        if matches:
            metric_data[file+"||"+metric] = [df[match].mean() for match in matches]


for d1 in metric_data.items():
    for d2 in metric_data.items():
        if d1 is d2:
            continue
        name1, means1 = d1
        name2, means2 = d2
        
        print(f"Comparing {name1} and {name2}")
        for i, m in enumerate(zip(means1, means2)):
            m1, m2 = m
            print(f"Exp {i}")
            print(f" {m1} : {m2}")
    
    
    
        

