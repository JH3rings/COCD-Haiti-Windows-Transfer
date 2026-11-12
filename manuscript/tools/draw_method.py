from pathlib import Path
import math,html,pymupdf as fitz
P=Path(__file__).resolve().parents[1]
o=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="390" viewBox="0 0 900 390"><rect width="900" height="390" fill="white"/>']
blue=('#DAE2F5','#5E7899');green=('#E1EFD9','#5E7D3E');gold=('#FAE4D5','#A0823E');rose=('#ECC1C0','#B36876')
def text(x,y,s,size=13,bold=False,anchor='middle'):
 if 'L_CEW' in s:
  before,after=s.split('L_CEW')
  parts=[(before+'L',size,0),('CEW',size*.68,3),(after,size,0)]
  widths=[fitz.get_text_length(t,fontname='tibo' if bold else 'tiro',fontsize=z) for t,z,_ in parts]
  widths[1]+=3
  left=x-(sum(widths)/2 if anchor=='middle' else 0)
  for (t,z,dy),w in zip(parts,widths):
   text(left,y+dy,t,z,bold,'start');left+=w
  return
 s=html.escape(s)
 o.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Times New Roman, Times, serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="#303030">{s}</text>')
def box(x,y,w,h,lines,col=blue):
 o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{col[0]}" stroke="{col[1]}" stroke-width="1.2"/>')
 for i,s in enumerate(lines):text(x+w/2,y+h/2+(i-(len(lines)-1)/2)*15+4,s)
def arrow(x,y,u,v,dash=False):
 o.append(f'<line x1="{x}" y1="{y}" x2="{u}" y2="{v}" stroke="#505050" stroke-width="1.2" '+('stroke-dasharray="4 3"' if dash else '')+'/>')
 a=math.atan2(v-y,u-x);pts=[(u,v),(u-6*math.cos(a)+3*math.sin(a),v-6*math.sin(a)-3*math.cos(a)),(u-6*math.cos(a)-3*math.sin(a),v-6*math.sin(a)+3*math.cos(a))]
 o.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in pts)+'" fill="#505050"/>')
text(14,19,'(a) Optical guidance during training; SAR and terrain at deployment',15,True,'start')
box(14,34,142,43,['SAR + terrain + optical','18 channels'],green);box(183,34,170,43,['Frozen U-Net teacher','Stage 2 features'],green);arrow(156,55,183,55)
box(403,34,190,43,['Cross-event relation loss','L_CEW (training only)'],rose);arrow(353,55,403,55)
text(640,51,'Teacher targets: stop-gradient',12,False,'start');text(640,70,'Optical branch removed at inference',12,False,'start')
box(14,108,120,44,['SAR + terrain','6 channels']);box(153,108,100,44,['U-Net encoder']);arrow(134,130,153,130)
for x,n,res in [(278,1,16),(416,2,32),(554,3,64),(692,4,128)]:
 box(x,108,115,44,[f'Decoder {n}',f'{res} × {res}'+(' + EPM' if n>1 else '')],gold if n>1 else blue)
 arrow(x-25,130,x,130)
box(830,108,58,44,['Mask']);arrow(807,130,830,130);arrow(474,108,474,77,True)
box(435,176,356,28,['One shared prototype bank: K = 16, dimension = 64'],gold)
for x in (474,612,750):arrow(x,176,x,152)
text(15,177,'U-Net skip connections retained',12,False,'start');text(15,197,'Training: BCE + Dice + 30 L_CEW',13,False,'start')
o.append('<line x1="14" y1="218" x2="888" y2="218" stroke="#cccccc"/>')
text(14,241,'(b) Prototype refinement (EPM)',15,True,'start')
box(14,260,62,35,['Feature f']);box(96,260,140,35,['Query projection','Cosine + softmax']);box(256,260,147,35,['Weighted prototypes','Value projection'],gold)
arrow(76,278,96,278);arrow(236,278,256,278)
box(256,323,147,35,['Output projection','Zero initialized'],gold);arrow(329,295,329,323)
box(96,323,140,35,['Residual addition','Refined feature f′']);arrow(256,340,236,340);arrow(45,295,45,340);arrow(45,340,96,340)
text(14,381,'Stage-specific projections; shared normalized prototypes',12,False,'start')
o.append('<line x1="425" y1="233" x2="425" y2="380" stroke="#cccccc"/>')
text(443,241,'(c) Cross-event distribution matching',15,True,'start')
box(443,260,195,40,['Teacher regional similarities','Other-event partners only'],green);box(674,260,214,40,['Student regional similarities','Same anchors and partner set'],blue)
box(443,315,195,30,['Shift and sort independently'],green);box(674,315,214,30,['Shift and sort independently'],blue)
arrow(540,300,540,315);arrow(781,300,781,315)
text(664,366,'L_CEW = mean squared difference of sorted values',13,True)
text(664,383,'Partner identities may differ at each rank; no top-k selection',12)
o.append('</svg>');svg=P/'source/method.svg';svg.write_text(''.join(o))
d=fitz.open(str(svg));pdf=fitz.open('pdf',d.convert_to_pdf());pdf.save(str(P/'figures_dlr/method.pdf'));pdf[0].get_pixmap(dpi=120).save(str(P/'figures_dlr/method_preview.png'))
