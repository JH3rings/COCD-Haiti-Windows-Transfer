"""Vector architecture diagram of the final F_SSP model; no model changes.

Feature prisms show spatial resolution, not measured activations. Input
thumbnails are taken from the first case of the existing qualitative figure.
All arrows, feature blocks, matrices, and labels remain editable in the SVG.
"""
from pathlib import Path
import base64
import html
import math
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
W, H = 1200, 556
INK = '#303035'
BLUE = ('#DAE2F5', '#E7E6E6', '#B1AFD4')
LILAC = ('#B1AFD4', '#F7EAE9', '#ECC1C0')
PEACH = ('#F8CDAC', '#FAE4D5', '#ECC1C0')
GREEN = ('#E1EFD9', '#E1EFD9', '#E7E6E6')
ROSE = '#F5B7BE'
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<rect width="1200" height="556" fill="white"/>']


def rect(x, y, w, h, fill='white', stroke='#E7E6E6', sw=1, rx=0, dash=False):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'+(' stroke-dasharray="6 4"' if dash else '')+'/>')


def txt(x, y, value, size=17, bold=False, anchor='middle', color=INK):
    # Separate text objects keep mathematical subscripts stable in MuPDF.
    if 'L_CEW' in value:
        a, b = value.split('L_CEW')
        parts = [(a+'L', size, 0), ('CEW', size*.68, 3), (b, size, 0)]
        widths = [fitz.get_text_length(t, fontname='tibo' if bold else 'tiro', fontsize=z) for t,z,_ in parts]
        widths[1] += 3
        left = x - (sum(widths)/2 if anchor == 'middle' else 0)
        for (t,z,dy),width in zip(parts,widths):
            txt(left,y+dy,t,z,bold,'start',color); left += width
        return
    svg.append(f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}" text-anchor="{anchor}" fill="{color}">{html.escape(value)}</text>')


def poly(points, fill, stroke=INK, sw=.85):
    svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def route(points, color=INK, sw=1.35, dash=False, head=True):
    svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"'+(' stroke-dasharray="5 3"' if dash else '')+'/>')
    if head:
        (x,y),(u,v)=points[-2:]; angle=math.atan2(v-y,u-x)
        poly([(u,v),(u-5*math.cos(angle)+2.4*math.sin(angle),v-5*math.sin(angle)-2.4*math.cos(angle)),(u-5*math.cos(angle)-2.4*math.sin(angle),v-5*math.sin(angle)+2.4*math.cos(angle))],color,color,.5)


def prism(x, cy, w, h, palette=BLUE, d=7):
    y=cy-h/2;front,top,side=palette
    poly([(x,y),(x+d,y-d),(x+w+d,y-d),(x+w,y)],top,side)
    poly([(x+w,y),(x+w+d,y-d),(x+w+d,y+h-d),(x+w,y+h)],side,side)
    rect(x,y,w,h,front,side,.9)
    # An inner edge indicates a stack of channels without glossy shadows.
    route([(x+3,y+2),(x+3,y+h-2)],top,.7,head=False)
    return x+w+d


def label_box(x,y,w,h,lines,fill,stroke=INK,size=16):
    rect(x,y,w,h,fill,stroke,1,4)
    for i,line in enumerate(lines):txt(x+w/2,y+h/2+(i-(len(lines)-1)/2)*(size+1)+size*.33,line,size)


def circle(x,y,r,label,color=INK):
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="1.25"/>')
    txt(x,y+5,label,18,color=color)


doc=fitz.open(ROOT/'figures_dlr/qualitative.pdf')
images=sorted(doc[0].get_image_info(xrefs=True), key=lambda q:(round(q['bbox'][1]),q['bbox'][0]))


def thumbnail(index,x,y,size=30):
    data=doc.extract_image(images[index]['xref'])
    uri='data:image/'+data['ext']+';base64,'+base64.b64encode(data['image']).decode()
    svg.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" xlink:href="{uri}"/>')
    rect(x,y,size,size,'none','#303035',.7)


def input_stack(y, teacher):
    # Both rows use the same SAR/terrain input; optical is teacher-only.
    if teacher:
        thumbnail(0,21,y,28);thumbnail(1,51,y,28)
        txt(50,y+41,'Optical',14)
        thumbnail(2,21,y+48,28);thumbnail(3,51,y+48,28)
        txt(50,y+90,'S1 VV shown',12)
    else:
        thumbnail(2,21,y+19,28);thumbnail(3,51,y+19,28)
        txt(50,y+62,'S1 VV shown',12)
    txt(34,y-6 if teacher else y+12,'pre',12)
    txt(64,y-6 if teacher else y+12,'post',12)


def tokens(x,y,palette):
    prism(x,y+15,36,32,palette,5)
    for i in range(1,4):
        route([(x+i*9,y-1),(x+i*9,y+31)],palette[2],.55,head=False)
        route([(x,y-1+i*8),(x+36,y-1+i*8)],palette[2],.55,head=False)


# (a) Two matched U-Nets with explicit resolution changes and skip connections.
txt(15,21,'(a) Optical-guided training and SAR–terrain deployment',21,True,'start')
rect(10,31,1180,145,'#F7EAE9','#ECC1C0',.8,4,False)
rect(10,183,1180,153,'#E7E6E6','#B1AFD4',.8,4)
txt(98,48,'Frozen teacher',17,True,'start',INK)
txt(285,48,'S2 pre/post (12) + S1 VV/VH pre/post (4) + DEM/slope (2) = 18ch',12,anchor='start')
txt(96,201,'Trainable student',17,True,'start',INK)
txt(294,201,'S1 VV/VH pre/post (4) + DEM/slope (2) = 6ch; optical-free inference',12,anchor='start')
input_stack(67,True); input_stack(220,False)
xs=[139+83*i for i in range(9)]
hs=[44,34,25,18,13,18,25,34,44]
ws=[23,27,31,35,40,35,31,27,23]
offset=[0,10,20,30,40,30,20,10,0]
res=[128,64,32,16,8,16,32,64,128]

for teacher,base,palette in [(True,92,LILAC),(False,244,BLUE)]:
    centers=[base+v for v in offset]
    # Skip features are concatenated before each decoder's double convolution.
    for j in range(4):
        x=xs[j]+ws[j]/2;end=xs[8-j]+ws[8-j]/2;line_y=base-40+j*7
        route([(x,centers[j]-hs[j]/2-7),(x,line_y),(end,line_y),(end,centers[8-j]-hs[8-j]/2-7)],'#E7E6E6' if teacher else '#B1AFD4',.95)
    if teacher:
        route([(82,81),(95,81),(95,129),(82,129)],head=False)
        route([(95,base),(139,base)])
    else:
        route([(83,base+5),(111,base+5),(111,base),(139,base)])
    for i,(x,cy,w,h) in enumerate(zip(xs,centers,ws,hs)):
        prism(x,cy,w,h,palette)
        txt(x+w/2+3,cy+h/2+16,str(res[i])+'²/'+str([64,128,256,512,1024,512,256,128,64][i])+'ch',11.5)
        if i<8:
            start=x+w+7;dest=xs[i+1];nextcy=centers[i+1]
            if not teacher and i in (6,7):
                ex=start+5;prism(ex,cy,26,23,PEACH,4);txt(ex+13,cy+4,'EPM',11,True)
                route([(start,cy),(ex,cy)],sw=1.1)
                route([(ex+30,cy),(dest-8,cy),(dest-8,nextcy),(dest,nextcy)])
            else:
                route([(start,cy),(dest-10,cy),(dest-10,nextcy),(dest,nextcy)])
    last=xs[-1]+ws[-1]+7
    if not teacher:
        ex=last+7;prism(ex,base,29,27,PEACH,5);txt(ex+14,base+4,'EPM',12,True)
        route([(last,base),(ex,base)],sw=1.1);last=ex+34
    route([(last,base),(900,base)])
    label_box(900,base-18,65,36,['1×1 head'],'#E7E6E6','#B1AFD4',14)
    route([(965,base),(987,base)])
    # First DLR qualitative case: teacher and final F_SSP student predictions.
    # These are error-overlay maps (TP green, FP red, FN blue), NOT raw logits.
    thumbnail(9 if teacher else 8,988,base-19,36)
    txt(1006,base+34,'Prediction',12)
    if teacher:
        txt(935,55,'Teacher seg. output (not a KD target)',12)
    # Relation features are tapped after the stage-2 refinement in the student.
    tapx=xs[6]+ws[6]+(7 if teacher else 42)
    tapy=centers[6]
    lane=base+63
    route([(tapx,tapy),(tapx,lane),(1047,lane),(1047,base+18)],ROSE,1.3,True)
    tokens(1030,base-17,LILAC if teacher else BLUE)
    txt(1050,base-28,'4×4 pooled',12)
    route([(1072,base),(1100,base),(1100,166 if teacher else 197),(1117,166 if teacher else 197)],ROSE,1.3,True)

txt(425,165,'Encoder: 2×(3×3 Conv) + 2×2 MaxPool',12)
txt(777,165,'Decoder: 2×2 UpConv + skip concat + 2 Conv',12)
txt(414,325,'32 / 64 / 128-pixel EPMs; scale-specific projections, shared K=16',12)
label_box(1117,151,64,61,['L_CEW','(c)'], '#F5B7BE',INK,16)
txt(1108,143,'Training only',13,color=ROSE)
# Shared bank feeds all three EPMs; it is a parameter bank, not teacher features.
label_box(675,314,238,18,['Shared prototype bank  K = 16'], '#FAE4D5','#ECC1C0',13)
for x,cy in [(695,264),(772,254),(855,244)]:
    route([(x,314),(x,cy+16)],'#ECC1C0',1.05)
# Segmentation supervision is separate from the relation loss.
route([(1003,270),(1003,293),(1057,293)],'#303035',1.1)
label_box(1057,279,124,29,['BCE + Dice'],'#E1EFD9','#E7E6E6',15)
thumbnail(4,1081,308,25)  # Actual binary ground truth in first DLR test case.
txt(1112,327,'GT (same case)',11,anchor='start')
# GT thumbnail directly abuts segmentation supervision; no decorative connector.

# (b) Expanded prototype module: cosine assignments, projected values, residual.
rect(10,353,648,193,'#FAE4D5','#F8CDAC',1,6)
txt(23,376,'(b) Shared-prototype refinement (EPM)',20,True,'start')
prism(31,443,25,43,BLUE,7);txt(47,486,'f',18)
route([(63,443),(92,443)])
label_box(92,420,82,46,['Query 1×1','Wq(l)'],'#DAE2F5',INK,14)
route([(174,443),(210,443)])
txt(194,414,'cos / τ; τ=0.1',12)
# Shared prototypes (K rows schematically represented by a small tiled bank).
for row in range(2):
    for col in range(8):rect(218+col*9,390+row*7,8,6,PEACH[0],PEACH[2],.4)
txt(304,387,'Shared bank P: 16×D, normalized',13,anchor='start')
route([(249,406),(249,420)],PEACH[2],1.1)
for row in range(4):
    for col in range(4):
        rect(214+col*11,423+row*10,10,9,['#FAE4D5','#F8CDAC','#FDE694','#ECC1C0'][(row+col)%4],PEACH[2],.35)
txt(237,482,'Softmax assignments',14)
route([(290,398),(353,398),(353,419)],PEACH[2],1.1)
label_box(309,419,88,31,['Value 1×1'],'#FAE4D5',INK,15)
circle(373,474,11,'×',PEACH[2]);route([(258,443),(280,443),(280,474),(362,474)],PEACH[2],1.1)
route([(353,450),(353,461),(373,461),(373,463)],PEACH[2],1.1)
route([(384,474),(417,474)])
label_box(417,451,82,46,['Output 1×1','zero-init'],'#FAE4D5',INK,14)
txt(458,442,'Zero initialized',14)
route([(499,474),(528,474)])
circle(539,474,11,'+')
route([(45,465),(45,509),(539,509),(539,485)],BLUE[2],1.2)
route([(550,474),(582,474)])
prism(582,474,25,43,BLUE,7);txt(598,516,'f′',18)
txt(24,536,'a=softmax(cos(Wq f,P)/τ); f′=f+Wo∑(a·Wv P); stage-specific 1×1 paths',12,anchor='start')

# (c) Exact computation of the cross-event relation loss, with schematic values.
rect(674,353,516,193,'#F7EAE9','#ECC1C0',1,6)
txt(688,376,'(c) Cross-event relation distillation',20,True,'start')
for row,y,palette,name in [(0,396,LILAC,'Teacher'),(1,465,BLUE,'Student')]:
    # Gray diagonal event blocks are excluded; off-diagonal entries are valid.
    for i in range(6):
        for j in range(6):
            fill='#E7E6E6' if i//3==j//3 else [palette[0],palette[1],palette[2]][(i+2*j+row)%3]
            rect(690+j*7,y+i*7,6.8,6.8,fill,'white',.25)
    txt(711,y+58,name,14)
    route([(738,y+21),(822,y+21)],ROSE,1.1)
    txt(780,y+9,'Cross-event sort',12)
    # Independently sorted similarities (one anchor), not paired partner values.
    vals=[6,10,13,18,22,27,30,34] if row==0 else [7,9,15,17,24,26,29,35]
    for j,h in enumerate(vals):rect(830+j*9,y+40-h,6,h,palette[0],palette[2],.5)
    route([(907,y+22),(952,y+22),(952,450),(970,450)],ROSE,1.2)
label_box(970,421,204,58,['Squared 1-D Wasserstein','L_CEW'], '#F5B7BE',ROSE,16)
txt(1072,500,'Sorted similarities',13)
txt(1072,518,'(per anchor, squared)',13)
txt(687,536,'4×4 pooled regions, valid≥75%; gray = same-event pairs excluded.',12,anchor='start')

txt(34,342,'Actual DLR test patch CN0001: S2/S1 views and GT/prediction overlays; TP green, FP red, FN blue.',11,anchor='start')
svg.append('</svg>')
path=ROOT/'source/method.svg';path.write_text('\n'.join(svg))
document=fitz.open(path);pdf=fitz.open('pdf',document.convert_to_pdf())
pdf.save(ROOT/'figures_dlr/method.pdf')
pdf[0].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(ROOT/'figures_dlr/method_preview.png')
print('Generated editable U-Net architecture:',path)
