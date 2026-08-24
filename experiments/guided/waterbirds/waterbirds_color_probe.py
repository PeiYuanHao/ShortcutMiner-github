import os, glob, random, math
from collections import defaultdict
from PIL import Image
import numpy as np

root = r'E:\小龙虾\waterbirds'
random.seed(0)
class_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
paths_by_class = {}
for d in class_dirs:
    paths=[]
    for ext in ('*.jpg','*.jpeg','*.png'):
        paths += glob.glob(os.path.join(root,d,ext))
    paths_by_class[d]=sorted(paths)

# sample up to 20 images per class
sample_paths=[]
for d, paths in paths_by_class.items():
    k=min(20, len(paths))
    sample_paths += [(d,p) for p in random.sample(paths, k)]


def border_mask(h,w,frac=0.15):
    bh=max(1,int(h*frac)); bw=max(1,int(w*frac))
    m=np.zeros((h,w), dtype=bool)
    m[:bh,:]=True; m[-bh:,:]=True; m[:,:bw]=True; m[:,-bw:]=True
    return m


def color_buckets(arr):
    # arr RGB uint8 Nx3
    arr=arr.astype(np.float32)/255.0
    r,g,b=arr[:,0],arr[:,1],arr[:,2]
    mx=np.max(arr,axis=1); mn=np.min(arr,axis=1); diff=mx-mn
    blue=((b>g+0.05)&(b>r+0.05)&(b>0.28))
    green=((g>r+0.05)&(g>b+0.02)&(g>0.22))
    brown=((r>g)&(g>b)&(r>0.20)&(b<0.45))
    gray=((diff<0.10)&(mx>0.25)&(mx<0.85))
    bright=((mx>0.85)&(diff<0.12))
    return {
        'blue': blue.mean(),
        'green': green.mean(),
        'brown': brown.mean(),
        'gray': gray.mean(),
        'bright': bright.mean(),
    }

res=[]
for cls,p in sample_paths:
    try:
        img=Image.open(p).convert('RGB')
        arr=np.array(img)
        h,w,_=arr.shape
        m=border_mask(h,w,0.18)
        border=arr[m]
        whole=arr.reshape(-1,3)
        c=color_buckets(border)
        cw=color_buckets(whole)
        res.append((cls, os.path.basename(p), h,w,c,cw))
    except Exception:
        pass

agg=defaultdict(lambda: defaultdict(list))
for cls,fn,h,w,c,cw in res:
    agg[cls]['n'].append(1)
    agg[cls]['h'].append(h)
    agg[cls]['w'].append(w)
    for k,v in c.items(): agg[cls]['border_'+k].append(v)
    for k,v in cw.items(): agg[cls]['whole_'+k].append(v)

rows=[]
for cls,d in agg.items():
    row={'class':cls,'n':len(d['n'])}
    for k,v in d.items():
        if k in ('n',): continue
        row[k]=float(np.mean(v))
    rows.append(row)

rows=sorted(rows,key=lambda x:x['class'])
print('sampled_classes', len(rows))
print('sampled_images', len(res))

def top(key, n=15):
    return sorted(rows, key=lambda x:x.get(key,0), reverse=True)[:n]

for key in ['border_blue','border_green','border_brown','whole_blue','whole_green']:
    print('\nTOP', key)
    for r in top(key,10):
        print(r['class'], round(r.get(key,0),3), 'n=',r['n'])

# summary across classes
for key in ['border_blue','border_green','border_brown','whole_blue','whole_green']:
    vals=[r.get(key,0) for r in rows]
    print('SUMMARY', key, 'mean=', round(float(np.mean(vals)),3), 'std=', round(float(np.std(vals)),3), 'max=', round(float(np.max(vals)),3))
