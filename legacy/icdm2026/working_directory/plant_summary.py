import pandas as pd
p=r'E:\小龙虾\plant_disease_qualitative\features.csv'
df=pd.read_csv(p)
print('shape counts by class')
print(df.groupby(['cls','width','height']).size().sort_values(ascending=False).head(30).to_string())
print('\naspect stats')
print(df.groupby('cls')['aspect'].describe().round(3).to_string())
print('\nsplit class means selected')
cols=['lap_var','edge_density','tex_entropy','mean_b','brown_ratio','fg_area','border_r','border_b','border_std','aspect']
print(df.groupby(['split','cls'])[cols].mean().round(3).to_string())
print('\nextreme aspect counts')
print(df.assign(aspect_bin=pd.qcut(df.aspect.rank(method='first'),10)).groupby(['aspect_bin','cls']).size().unstack(fill_value=0).to_string())
