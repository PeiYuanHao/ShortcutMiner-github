from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
root = Path(r'E:\小龙虾\waterbirds')
out = Path(r'E:\小龙虾\analysis')
out.mkdir(exist_ok=True)
clusters = {
    'seabirds': ['001.Black_footed_Albatross','023.Brandt_Cormorant','059.California_Gull','100.Brown_Pelican','141.Artic_Tern','106.Horned_Puffin'],
    'forest_perchers': ['028.Brown_Creeper','094.White_breasted_Nuthatch','187.American_Three_toed_Woodpecker','188.Pileated_Woodpecker','193.Bewick_Wren','175.Pine_Warbler'],
    'wetland_water': ['046.Gadwall','050.Eared_Grebe','079.Belted_Kingfisher','089.Hooded_Merganser','126.Nelson_Sharp_tailed_Sparrow','197.Marsh_Wren'],
    'small_colorful': ['067.Anna_Hummingbird','068.Ruby_throated_Hummingbird','069.Rufous_Hummingbird','047.American_Goldfinch','054.Blue_Grosbeak','182.Yellow_Warbler']
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
    thumbs=[]
    labels=[]
    for cls in class_list:
        files = sorted((root/cls).glob('*.jpg'))[:4]
        for f in files:
            thumbs.append(load_thumb(f))
            labels.append(cls.split('.',1)[1][:18])
    cols=4
    tw,th = thumbs[0].size
    label_h=24
    rows=(len(thumbs)+cols-1)//cols
    sheet = Image.new('RGB',(cols*tw,rows*(th+label_h)),(245,245,245))
    draw=ImageDraw.Draw(sheet)
    for i,(im,lab) in enumerate(zip(thumbs,labels)):
        r,c=divmod(i,cols)
        x,y=c*tw,r*(th+label_h)
        sheet.paste(im,(x,y))
        draw.text((x+5,y+th+4),lab,fill='black')
    sheet.save(out/f'{cname}_sheet.jpg', quality=90)
    print('saved', out/f'{cname}_sheet.jpg')
