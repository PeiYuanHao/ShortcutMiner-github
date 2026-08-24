from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np, pandas as pd, math, os
root=Path(r'E:\小龙虾\Plant Disease Recognition Dataset')
out=Path(r'E:\小龙虾\plant_disease_qualitative')
out.mkdir(exist_ok=True)
rows=[]
for splitdir in ['Train/Train','Validation/Validation','Test/Test']:
    for cls in ['Healthy','Powdery','Rust']:
        d=root/splitdir/cls
        for p in d.glob('*'):
            if p.suffix.lower() not in ['.jpg','.jpeg','.png','.bmp','webp']: continue
            try:
                im=Image.open(p).convert('RGB')
                arr=np.asarray(im).astype(np.float32)
                h,w=arr.shape[:2]
                gray=arr.mean(axis=2)
                # laplacian var simple
                lap = -4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
                lap_var=float(lap.var())
                # gradients edge density
                gx=np.zeros_like(gray); gy=np.zeros_like(gray)
                gx[:,1:-1]=gray[:,2:]-gray[:,:-2]; gy[1:-1,:]=gray[2:,:]-gray[:-2,:]
                grad=np.sqrt(gx*gx+gy*gy)
                edge_density=float((grad>35).mean())
                # simple LBP entropy on resized-ish gray sample
                g=gray
                center=g[1:-1,1:-1]
                code=np.zeros(center.shape,dtype=np.uint8)
                neigh=[g[:-2,:-2],g[:-2,1:-1],g[:-2,2:],g[1:-1,2:],g[2:,2:],g[2:,1:-1],g[2:,:-2],g[1:-1,:-2]]
                for i,n in enumerate(neigh): code |= ((n>=center).astype(np.uint8)<<i)
                hist=np.bincount(code.ravel(), minlength=256).astype(float); hist/=hist.sum()
                lbp_entropy=float(-(hist[hist>0]*np.log2(hist[hist>0])).sum())
                R,G,B=arr[:,:,0],arr[:,:,1],arr[:,:,2]
                mean_r,mean_g,mean_b=map(float,[R.mean(),G.mean(),B.mean()])
                # HSV-ish saturation/value
                maxc=arr.max(axis=2); minc=arr.min(axis=2)
                sat=np.where(maxc>0,(maxc-minc)/maxc,0)
                # brown/rust ratio: R > G > B-ish and warm/dark orange-brown pixels
                brown=((R>G*1.05)&(G>B*1.05)&(R>80)&(G>40)&(B<150)&(sat>0.18)).mean()
                # foreground: saturated or non-bright non-white pixels, approximate leaf occupying image
                fg=((sat>0.12)&(maxc<250)).mean()
                # border 5% stats
                bw=max(1,int(w*0.05)); bh=max(1,int(h*0.05))
                border=np.concatenate([arr[:bh,:,:].reshape(-1,3),arr[-bh:,:,:].reshape(-1,3),arr[:, :bw,:].reshape(-1,3),arr[:, -bw:,:].reshape(-1,3)],axis=0)
                border_r=float(border[:,0].mean()); border_b=float(border[:,2].mean()); border_std=float(border.std())
                # center-border color diff
                ch0,ch1=int(h*0.25),int(h*0.75); cw0,cw1=int(w*0.25),int(w*0.75)
                center_arr=arr[ch0:ch1,cw0:cw1,:].reshape(-1,3).mean(axis=0)
                border_mean=border.mean(axis=0)
                border_center_l2=float(np.linalg.norm(center_arr-border_mean))
                rows.append(dict(path=str(p), split=splitdir.split('/')[0], cls=cls, filename=p.name, file_size=p.stat().st_size, width=w,height=h,aspect=w/h,lap_var=lap_var,edge_density=edge_density,lbp_entropy=lbp_entropy,mean_r=mean_r,mean_g=mean_g,mean_b=mean_b,brown_ratio=float(brown),fg_area=float(fg),border_r=border_r,border_b=border_b,border_std=border_std,border_center_l2=border_center_l2))
            except Exception as e:
                print('ERR',p,e)
df=pd.DataFrame(rows)
df.to_csv(out/'features.csv', index=False)
print('n',len(df)); print(df.groupby(['split','cls']).size());
print('\nmeans by cls')
cols=['lap_var','edge_density','lbp_entropy','mean_b','brown_ratio','fg_area','border_r','border_b','border_std','aspect','file_size']
print(df.groupby('cls')[cols].mean().round(4).to_string())
# sample sheets
def thumb(path, size=(170,130)):
    im=Image.open(path).convert('RGB'); im.thumbnail(size)
    canv=Image.new('RGB',size,'white'); canv.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2)); return canv
def make_sheet(name, sub, label_cols):
    sub=sub.head(30)
    cols=5; tw,th=170,130; lh=50
    rowsn=(len(sub)+cols-1)//cols
    sheet=Image.new('RGB',(cols*tw, rowsn*(th+lh)),(245,245,245)); draw=ImageDraw.Draw(sheet)
    for i,(_,r) in enumerate(sub.iterrows()):
        x=(i%cols)*tw; y=(i//cols)*(th+lh)
        sheet.paste(thumb(r.path),(x,y))
        lab=f"{r.cls} {r.split}\n" + ' '.join([f"{c[:3]}={getattr(r,c):.2f}" for c in label_cols if isinstance(getattr(r,c),float)])
        draw.text((x+3,y+th+2),lab,fill='black')
    sheet.save(out/f'{name}.jpg', quality=92)
# class samples
for cls in ['Healthy','Powdery','Rust']:
    make_sheet(f'random_{cls}', df[df.cls==cls].sample(min(30,(df.cls==cls).sum()), random_state=7), ['lap_var','edge_density','mean_b','brown_ratio','fg_area'])
make_sheet('low_lbp_entropy', df.sort_values('lbp_entropy'), ['lbp_entropy','lap_var','edge_density','mean_b'])
make_sheet('low_sharpness', df.sort_values('lap_var'), ['lap_var','edge_density','lbp_entropy'])
make_sheet('high_blue', df.sort_values('mean_b', ascending=False), ['mean_b','lbp_entropy','lap_var'])
make_sheet('high_brown_ratio', df.sort_values('brown_ratio', ascending=False), ['brown_ratio','mean_b','fg_area'])
make_sheet('low_fg_area', df.sort_values('fg_area'), ['fg_area','aspect','border_std'])
make_sheet('high_fg_area', df.sort_values('fg_area', ascending=False), ['fg_area','aspect','border_std'])
make_sheet('extreme_aspect_low', df.sort_values('aspect'), ['aspect','fg_area','border_std'])
make_sheet('extreme_aspect_high', df.sort_values('aspect', ascending=False), ['aspect','fg_area','border_std'])
make_sheet('high_border_std', df.sort_values('border_std', ascending=False), ['border_std','border_r','border_b','fg_area'])
print('sheets saved', out)
