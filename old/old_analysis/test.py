import re
import pandas as pd
import os
PATH = "eval_results/"
results = {file_name : pd.read_csv(PATH + file_name) for file_name in os.listdir(PATH)}
col_names = {file : cols.keys().tolist() for file, cols in results.items()}

# 1) Pre-compile your regex for efficiency
pattern_string = r'Projection'
pattern = re.compile(pattern_string)
# 2) For each file, grab the *first* column that matches, or skip if none do
filtered_results = {
    file: next((col for col in cols if pattern.search(col)), None)
    for file, cols in col_names.items()
}
# 3) Drop any files that didn’t actually have a match
filtered_results = {f: col for f, col in filtered_results.items() if col}

pattern = re.compile(r'^(.*)(_gem|_rate)(.csv)$')

pairs = []
for fname in results:
    m = pattern.match(fname)
    if not m:
        continue
    base = m.group(1) + m.group(2)
    if base in results:
        pairs.append((base, fname))

# 2) for each pair, pull out the two columns and show them side-by-side
for base, sgem in pairs:
    col_base = filtered_results[base]
    col_sgem = filtered_results[sgem]

    df_base = results[base]
    df_sgem = results[sgem]

    # rename for clarity
    series_base = df_base[col_base].rename(pattern_string)
    series_sgem = df_sgem[col_sgem].rename(pattern_string)

    # align on index, then concat
    cmp_df = pd.concat([series_base, series_sgem], axis=1)
    print(f"\n--- Comparing “{base}” vs “{sgem}” ---")
    print(cmp_df)   