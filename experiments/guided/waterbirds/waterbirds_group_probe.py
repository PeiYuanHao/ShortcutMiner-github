import os, glob, random
from PIL import Image
import numpy as np

root = r'E:\小龙虾\waterbirds'
random.seed(0)
class_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
water_kw = ['Albatross','Auklet','Cormorant','Frigatebird','Fulmar','Gadwall','Grebe','Guillemot','Gull','Jaeger','Kingfisher','Kittiwake','Loon','Mallard','Merganser','Pelican','Puffin','Tern','Waterthrush','Pelagic']
water_classes=[]
land_classes=[]
for d in class_dirs:
    name=d.split('.',1)[1]
    if any(k.lower() in name.lower() for k in water_kw):
        water_classes.append(d)
    else:
        land_classes.append(d)

def border_blue_score(path):
    img=Image.open(path).convert('RGB')
    arr=np.array(img)
    h,w,_=arr.shape
    bh=max(1,int(h*0.18)); bw=max(1,int(w*0.18))
    m=np.zeros((h,w), dtype=bool)
    m[:bh,:]=True; m[-bh:,:]=True; m[:,:bw]=True; m[:,-bw:]=True
    border=arr[m].astype(np.float32)/255.0
    r,g,b=border[:,0],border[:,1],border[:,2]
    blue=((b>g+0.05)&(b>r+0.05)&(b>0.28)).mean()
    green=((g>r+0.05)&(g>b+0.02)&(g>0.22)).mean()
    return blue, green

def sample_group(group):
    vals=[]
    for d in group:
        paths=[]
        for ext in ('*.jpg','*.jpeg','*.png'):
            paths += glob.glob(os.path.join(root,d,ext))
        for p in random.sample(paths, min(20,len(paths))):
            try:
                vals.append(border_blue_score(p))
            except Exception:
                pass
    return np.array(vals)

w=sample_group(water_classes)
l=sample_group(land_classes)
print('water_classes', len(water_classes))
print('land_classes', len(land_classes))
print('water_samples', len(w), 'land_samples', len(l))
print('water_border_blue_mean', round(float(w[:,0].mean()),3))
print('land_border_blue_mean', round(float(l[:,0].mean()),3))
print('water_border_green_mean', round(float(w[:,1].mean()),3))
print('land_border_green_mean', round(float(l[:,1].mean()),3))
print('water_minus_land_blue', round(float(w[:,0].mean()-l[:,0].mean()),3))
print('land_minus_water_green', round(float(l[:,1].mean()-w[:,1].mean()),3))
print('example_water_classes', water_classes[:20])
