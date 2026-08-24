import pandas as pd
p=r'E:\小龙虾\ham10000_novel_shortcut_screen_results\image_features.csv'
df=pd.read_csv(p)
groups={
 'high_saturation': df.nlargest(501,'mean_S'),
 'high_brightness': df.nlargest(501,'mean_V'),
 'low_brightness': df.nsmallest(501,'mean_V'),
 'high_dark_corner': df.nlargest(501,'dark_corner_frac'),
 'high_dark_border': df.nlargest(501,'dark_border_frac'),
 'high_border_center_l2': df.nlargest(501,'border_center_l2'),
 'high_file_size': df.nlargest(501,'file_size'),
 'low_file_size': df.nsmallest(501,'file_size'),
}
for name,sub in groups.items():
    print('\n==',name,'==')
    print('dx:', sub.dx.value_counts(normalize=True).round(3).to_dict())
    print('part:', sub.part.value_counts(normalize=True).round(3).to_dict())
    print('file_index min/median/max:', int(sub.file_index.min()), int(sub.file_index.median()), int(sub.file_index.max()))
    print('file_size mean/median:', round(sub.file_size.mean(),1), int(sub.file_size.median()))
    print('bpp mean/median:', round(sub.bytes_per_pixel.mean(),3), round(sub.bytes_per_pixel.median(),3))
