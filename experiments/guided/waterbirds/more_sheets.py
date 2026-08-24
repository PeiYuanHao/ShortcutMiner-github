from pathlib import Path
from PIL import Image, ImageDraw
root = Path(r'E:\小龙虾\waterbirds')
out = Path(r'E:\小龙虾\analysis')
out.mkdir(exist_ok=True)
clusters = {
    'gulls': ['059.California_Gull','060.Glaucous_winged_Gull','061.Heermann_Gull','062.Herring_Gull','064.Ring_billed_Gull','066.Western_Gull'],
    'terns': ['141.Artic_Tern','142.Black_Tern','143.Caspian_Tern','144.Common_Tern','145.Elegant_Tern','146.Forsters_Tern'],
    'warblers': ['158.Bay_breasted_Warbler','160.Black_throated_Blue_Warbler','169.Magnolia_Warbler','171.Myrtle_Warbler','175.Pine_Warbler','182.Yellow_Warbler'],
    'woodpeckers': ['187.American_Three_toed_Woodpecker','188.Pileated_Woodpecker','189.Red_bellied_Woodpecker','190.Red_cockaded_Woodpecker','191.Red_headed_Woodpecker','192.Downy_Woodpecker']
}

def load_thumb(p, size=(220,180)):
    im = Image.open(p).convert('RGB')
    im.thumbnail(size)
    canvas = Image.new('RGB', size, 'white')
    x = (size[0]-im.width)//2
    y = (size[1]-im.height)//2
    canvas.paste(im, (x,y))
    return canvas

for cname, class_list in clusters.items():
    thumbs=[]; labels=[]
    for cls in class_list:
        files = sorted((root/cls).glob('*.jpg'))[:3]
        for f in files:
            thumbs.append(load_thumb(f))
            labels.append(cls.split('.',1)[1][:16])
    cols=3
    tw,th=thumbs[0].size; lh=24
    rows=(len(thumbs)+cols-1)//cols
    sheet=Image.new('RGB',(cols*tw,rows*(th+lh)),(245,245,245))
    draw=ImageDraw.Draw(sheet)
    for i,(im,lab) in enumerate(zip(thumbs,labels)):
        r,c=divmod(i,cols); x,y=c*tw,r*(th+lh)
        sheet.paste(im,(x,y)); draw.text((x+5,y+th+4),lab,fill='black')
    path=out/f'{cname}_sheet2.jpg'; sheet.save(path, quality=90); print(path)
