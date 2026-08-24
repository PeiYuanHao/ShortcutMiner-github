import pandas as pd
p=r'E:\小龙虾\ham10000_novel_shortcut_screen_results\image_features.csv'
df=pd.read_csv(p)
print(df.columns.tolist())
print(df.head().to_string())
