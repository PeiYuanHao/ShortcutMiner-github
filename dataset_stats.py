import os, glob
root = r'E:\小龙虾\waterbirds'
classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
counts=[]
for d in classes:
    n=0
    for ext in ('*.jpg','*.jpeg','*.png'):
        n += len(glob.glob(os.path.join(root,d,ext)))
    counts.append(n)
print('class_count', len(classes))
print('image_count', sum(counts))
print('min_per_class', min(counts))
print('max_per_class', max(counts))
print('mean_per_class', round(sum(counts)/len(counts),2))
print('median_like_sorted_mid', sorted(counts)[len(counts)//2])
print('smallest10', sorted(counts)[:10])
print('largest10', sorted(counts)[-10:])
