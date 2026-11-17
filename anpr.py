from __future__ import annotations
import functools
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITS = '0123456789'
STATE_CODES = {
    'AN','AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ','HP','HR','JH','JK',
    'KA','KL','LA','LD','MH','ML','MN','MP','MZ','NL','OD','OR','PB','PY','RJ','SK',
    'TN','TR','TS','UK','UP','WB'
}
AMBIGUITIES = {'0':'O','O':'0','1':'I','I':'1','2':'Z','Z':'2',
               '5':'S','S':'5','6':'G','G':'6','8':'B','B':'8'}

@dataclass
class PlateResult:
    box: tuple[int,int,int,int]
    text: str
    confidence: float
    detection_score: float
    status: str
    crop: Optional[np.ndarray] = None

    def info(self):
        d = asdict(self)
        d.pop('crop')
        return d


def imread(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite(path, image):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or '.jpg'
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ValueError('Cannot encode image: '+str(path))
    buf.tofile(str(p))


# Image Processing: Local contrast enhancement (CLAHE)
def clahe(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)


# Image Processing: Scale glyph into fixed canvas maintaining aspect ratio
def _standardize(bw, width=26, height=42):
    ys, xs = np.where(bw > 0)
    if len(xs) < 2:
        return np.zeros((height,width), np.uint8)
    glyph = bw[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h,w = glyph.shape
    scale = min((width-4)/max(w,1),(height-4)/max(h,1))
    nw,nh = max(1,int(round(w*scale))),max(1,int(round(h*scale)))
    glyph = cv2.resize(glyph,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    glyph = ((glyph>120)*255).astype(np.uint8)
    canvas = np.zeros((height,width), np.uint8)
    x0,y0=(width-nw)//2,(height-nh)//2
    canvas[y0:y0+nh,x0:x0+nw]=glyph
    return canvas


# Image Processing: Topological enclosed contour analysis (internal holes)
def _glyph_holes(bw):
    bw=(bw>0).astype(np.uint8)*255
    if not np.any(bw): return 0
    contours,hierarchy=cv2.findContours(bw,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return 0
    area=bw.shape[0]*bw.shape[1]
    return sum(1 for i,c in enumerate(contours)
               if hierarchy[0][i][3]>=0 and cv2.contourArea(c)>=max(3,area*.013))


def _font_paths():
    candidates=[
        'C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/arialbd.ttf',
        'C:/Windows/Fonts/ARIALN.TTF','C:/Windows/Fonts/ARIALNB.TTF',
        'C:/Windows/Fonts/bahnschrift.ttf','C:/Windows/Fonts/verdana.ttf',
        'C:/Windows/Fonts/verdanab.ttf','C:/Windows/Fonts/tahoma.ttf',
        'C:/Windows/Fonts/tahomabd.ttf','C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/calibrib.ttf','C:/Windows/Fonts/segoeui.ttf',
        'C:/Windows/Fonts/segoeuib.ttf','C:/Windows/Fonts/impact.ttf',
        'C:/Windows/Fonts/consola.ttf','C:/Windows/Fonts/consolab.ttf',
        'C:/Windows/Fonts/trebuc.ttf','C:/Windows/Fonts/trebucbd.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSansNarrow-Regular.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSansNarrow-Bold.ttf',
        '/usr/share/fonts/truetype/lato/Lato-Regular.ttf',
        '/usr/share/fonts/truetype/lato/Lato-Heavy.ttf',
        '/usr/share/fonts/opentype/inter/Inter-Regular.otf',
        '/usr/share/fonts/opentype/inter/Inter-Bold.otf',
        '/usr/share/fonts/opentype/cabin/Cabin-Regular.otf',
        '/usr/share/fonts/opentype/cabin/Cabin-Bold.otf',
        '/usr/share/fonts/opentype/freefont/FreeSans.otf',
        '/usr/share/fonts/opentype/freefont/FreeSansBold.otf',
        '/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf',
        '/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf',
    ]
    custom=Path(__file__).resolve().parent/'custom_fonts'
    if custom.is_dir():
        candidates += [str(f) for f in custom.iterdir()
                       if f.is_file() and f.suffix.lower() in ('.ttf','.otf')]
    return list(dict.fromkeys(Path(x) for x in candidates if Path(x).is_file()))


def _width_var(bw,factor):
    if factor==1.0:return bw
    h,w=bw.shape
    nw=max(3,int(round(w*factor)))
    return cv2.resize(bw,(nw,h),interpolation=cv2.INTER_LINEAR)


# Image Processing: Synthesize character templates across fonts, weights and widths
@functools.lru_cache(maxsize=1)
def _templates():
    templates=[];labels=[]
    for font in (cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_DUPLEX,
                 cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_TRIPLEX):
        for thick in (2,3,5,7):
            for ch in CHARS:
                canvas=np.zeros((90,70),np.uint8)
                cv2.putText(canvas,ch,(7,70),font,2.0,255,thick,cv2.LINE_AA)
                for width in (.84,1.0,1.16):
                    templates.append(_standardize(_width_var(canvas,width)))
                    labels.append(ch)
    try:
        from PIL import Image, ImageDraw, ImageFont
        for path in _font_paths():
            try: font=ImageFont.truetype(str(path),74)
            except OSError:continue
            for stroke in (0,1,3):
                for ch in CHARS:
                    im=Image.new('L',(110,110),0)
                    draw=ImageDraw.Draw(im)
                    draw.text((9,4),ch,font=font,fill=255,
                              stroke_width=stroke,stroke_fill=255)
                    for width in (.82,1.0,1.18):
                        templates.append(_standardize(_width_var(np.array(im),width)))
                        labels.append(ch)
    except ImportError:pass
    mat=np.asarray(templates,np.float32).reshape(len(templates),-1)/255.
    mat-=np.mean(mat,axis=1,keepdims=True)
    mat/=np.maximum(np.linalg.norm(mat,axis=1,keepdims=True),1e-8)
    return mat,np.array(labels)


_COUNTERS={
    '0':(1,), '8':(2,), '6':(1,), '9':(1,), 'O':(1,), 'Q':(1,),
    'A':(1,), 'D':(1,), 'P':(1,), 'R':(1,), 'B':(2,), '4':(0,1)
}


# Image Processing: Normalized cross-correlation template matching
def _read_glyph(bw, allowed=None):
    if np.count_nonzero(bw)<8:return ('',0,{})
    mat,labels=_templates()
    norm=_standardize(bw).astype(np.float32).reshape(-1)/255.
    norm-=norm.mean()
    norm/=max(np.linalg.norm(norm),1e-8)
    scores=mat@norm
    best={ch:float(np.max(scores[labels==ch])) for ch in CHARS}
    holes=_glyph_holes(bw)
    for ch in CHARS:
        options=_COUNTERS.get(ch,(0,))
        delta=min(abs(holes-target) for target in options)
        best[ch]-=0.115*min(delta,2)
    pool=allowed or CHARS
    ranked=sorted(pool,key=lambda c:best[c],reverse=True)
    return ranked[0],best[ranked[0]],best


# Image Processing: Generate multiple Otsu and adaptive threshold binarizations
def _thresholds(roi):
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) if roi.ndim==3 else roi
    h,w=gray.shape
    sf=max(1.0, min(3.5,100/max(h,1)))
    gray=cv2.resize(gray,(min(1100,round(w*sf)),round(h*sf)),interpolation=cv2.INTER_CUBIC)
    h,w=gray.shape
    gray=cv2.GaussianBlur(gray,(3,3),0)
    eq=clahe(gray)
    variants=[]
    for base in (eq,gray):
        for inverse in (True,False):
            flag=cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
            _,ot=cv2.threshold(base,0,255,flag|cv2.THRESH_OTSU)
            variants.append(ot)
        block=max(15, (min(h,w)//3)|1)
        block=min(block,75)
        for inverse in (True,False):
            mode=cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
            variants.append(cv2.adaptiveThreshold(base,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,mode,block,11))
        break
    return variants


# Image Processing: Connected component character segmentation
def _characters(binary):
    h,w=binary.shape
    b=binary.copy()
    border=max(1,int(round(h*.025)))
    b[:border]=0;b[-border:]=0
    b[:,:max(1,int(w*.008))]=0;b[:,-max(1,int(w*.008)):]=0
    num, labels, stats, centers=cv2.connectedComponentsWithStats(b,8)
    cc=[]
    for j in range(1,num):
        x,y,cw,ch,a=map(int,stats[j])
        if a<max(8,h*w*.0001):continue
        if ch<h*.23 or ch>h*.89 or cw<w*.007 or cw>w*.20:continue
        if cw/ch>1.22 or cw/ch<.08:continue
        if a/(cw*ch)<.12:continue
        cc.append((x,y,cw,ch,a,j))
    if not cc:return []
    med=np.median([c[3] for c in cc])
    cc=[c for c in cc if c[3]>=max(h*.22,med*.50)]
    cc.sort(key=lambda c:c[0])
    return [(c, (labels[c[1]:c[1]+c[3],c[0]:c[0]+c[2]]==c[5]).astype(np.uint8)*255) for c in cc]


# Image Processing: Spatial baseline ordering (left-to-right / row-split)
def _order_chars(parts):
    if not parts:return []
    ycenters=np.array([c[1]+c[3]/2 for c,_ in parts])
    hs=np.array([c[3] for c,_ in parts]); medh=float(np.median(hs))
    if len(parts)>=7 and np.percentile(ycenters,85)-np.percentile(ycenters,15)>medh*.53:
        split=(min(ycenters)+max(ycenters))/2
        top=[p for p in parts if p[0][1]+p[0][3]/2<=split]
        bottom=[p for p in parts if p not in top]
        if len(top)>=2 and len(bottom)>=2:
            return sorted(top,key=lambda p:p[0][0])+sorted(bottom,key=lambda p:p[0][0])
    if len(parts)>=8:
        mids=np.array([p[0][1]+p[0][3]/2 for p in parts]); med=np.median(mids)
        inliers=[p for p in parts if abs(p[0][1]+p[0][3]/2-med)<medh*.40]
        if len(inliers)>=6:parts=inliers
    return sorted(parts,key=lambda p:p[0][0])


def _plate_patterns(n):
    patterns=[]
    for nd in (4,3,2):
        nl=n-4-nd
        if 0<=nl<=4:
            patterns.append('LL'+'DD'+'L'*nl+'D'*nd)
    return patterns


# Image Processing: Decode segmented character sequence using syntactic pattern constraints
def _decode(parts):
    n=len(parts)
    if n<5 or n>14:return '',0.0,False
    glyphs=[_read_glyph(bw) for _,bw in parts]
    patterns=_plate_patterns(n)
    if not patterns:
        text=''.join(g[0] for g in glyphs)
        return text,float(np.mean([g[1] for g in glyphs])),False
    options=[]
    for pat in patterns:
        out=[];scores=[]
        for i,(_,_,scmap) in enumerate(glyphs):
            allowed=LETTERS if pat[i]=='L' else DIGITS
            ch=max(allowed,key=lambda c:scmap[c]);out.append(ch);scores.append(scmap[ch])
        text=''.join(out)
        mean=float(np.mean(scores))
        bonus=0.06 if text[:2] in STATE_CODES else -0.03
        bonus+=.02 if pat.endswith('DDDD') else 0
        options.append((mean+bonus,text,mean,text[:2] in STATE_CODES))
    options.sort(reverse=True)
    _,txt,raw,valid=options[0]
    return txt,raw,valid


# Image Processing: Segment and recognize characters on cropped license plate ROI
def recognize_plate(crop):
    if crop is None or crop.size==0:return '',0.,{}
    candidates=[]
    for variant_idx,binary in enumerate(_thresholds(crop)):
        for clean_border in (False,):
            parts=_order_chars(_characters(binary))
            if not parts:continue
            text,score,valid=_decode(parts)
            if not text:continue
            n=len(parts)
            shapes=np.array([c[:4] for c,_ in parts],dtype=np.float32)
            heights=shapes[:,3]
            hconsistency=max(0,1-float(np.std(heights)/max(np.mean(heights),1)))
            plausible=(7<=n<=11)
            quality=(score*.75+.18*hconsistency+(.10 if valid else 0)+(.06 if plausible else -.15))
            candidates.append((quality,text,score,n,variant_idx,valid,binary))
    if not candidates:return '',0.,{}
    candidates.sort(key=lambda r:r[0],reverse=True)
    q,txt,score,n,idx,valid,binary=candidates[0]
    chosen_parts=_order_chars(_characters(binary))
    diagnostics=[]
    for i,(bounds,glyph) in enumerate(chosen_parts):
        if i<2 or (4<=i<n-4): allowed=LETTERS
        elif i>=n-4 or i in (2,3):allowed=DIGITS
        else:allowed=CHARS
        _,_,all_scores=_read_glyph(glyph,allowed)
        ranked=sorted(allowed,key=lambda c:all_scores[c],reverse=True)[:3]
        holes=_glyph_holes(glyph)
        explanation=''
        if holes==1:explanation='1 enclosed hole'
        elif holes>=2:explanation='2 enclosed holes'
        diagnostics.append({
            'position':i+1,'predicted':txt[i] if i<len(txt) else '',
            'enclosed_holes':holes,
            'alternatives':[{'char':c,'similarity':round(all_scores[c],3)} for c in ranked],
            'shape_note':explanation
        })
    conf=float(np.clip((q-.22)*1.22,0,1))
    return txt,conf,{'binary':binary,'char_count':n,'glyph_similarity':round(float(score),3),
                     'threshold_variant':idx,'format_valid':valid,'characters':diagnostics}


# Image Processing: Candidate region evaluation (Sobel vertical gradients, Canny density, aspect ratio)
def _score_box(gray,box):
    x,y,w,h=box
    patch=gray[y:y+h,x:x+w]
    if patch.size<90:return -1
    sm=cv2.resize(patch,(190,62))
    gx=cv2.Sobel(sm,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(sm,cv2.CV_32F,0,1,ksize=3)
    ev=cv2.Canny(sm,70,150)
    d=float(np.mean(ev>0))
    vertical=float(np.mean(np.abs(gx))/(np.mean(np.abs(gy))+1))
    density=max(0,1-abs(d-.20)/.23)
    vertical=min(vertical,2.5)/2.5
    ratio=w/h
    ratio_score=max(0,1-abs(np.log(max(ratio,.1)/3.6))/1.2)
    return .48*density+.27*vertical+.25*ratio_score


# Image Processing: Multi-scale morphological blackhat/tophat filtering & edge contour detection
def _candidate_boxes(image):
    H0,W0=image.shape[:2]
    sf=min(1.0,1150/max(W0,1))
    im=cv2.resize(image,None,fx=sf,fy=sf,interpolation=cv2.INTER_AREA) if sf<1 else image
    H,W=im.shape[:2]
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray=clahe(gray)
    sm=cv2.bilateralFilter(gray,5,45,45)
    proposals=[]
    def add_contours(mask,source):
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            if h<9 or w<26 or w/h<1.5 or w/h>9.5:continue
            ar=w*h/(W*H)
            if ar<.00010 or ar>.14:continue
            if w>W*.92 or h>H*.43:continue
            pad_x=int(w*.06)+1;pad_y=int(h*.16)+1
            xx=max(0,x-pad_x);yy=max(0,y-pad_y)
            ww=min(W-xx,w+2*pad_x);hh=min(H-yy,h+2*pad_y)
            if ww<20 or hh<8:continue
            proposals.append((xx,yy,ww,hh,source))
    for kx in (13,25,39):
        rect=cv2.getStructuringElement(cv2.MORPH_RECT,(kx,5))
        black=cv2.morphologyEx(sm,cv2.MORPH_BLACKHAT,rect)
        white=cv2.morphologyEx(sm,cv2.MORPH_TOPHAT,rect)
        for feature in (black,white):
            grad=cv2.Sobel(feature,cv2.CV_8U,1,0,ksize=3)
            grad=cv2.GaussianBlur(grad,(3,3),0)
            _,binary=cv2.threshold(grad,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT,(max(9,kx-4),3)))
            binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
            add_contours(binary,'morphology')
    canny=cv2.Canny(sm,80,185)
    for close_x in (7,15,27):
        edges=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT,(close_x,3)))
        add_contours(edges,'edge')
    for binary in (cv2.threshold(sm,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1],
                   cv2.threshold(sm,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]):
        add_contours(binary,'tone')
    options=[]
    for x,y,w,h,source in proposals:
        score=_score_box(gray,(x,y,w,h))
        if score<.19:continue
        options.append((score,(x,y,w,h)))
    options.sort(key=lambda q:q[0],reverse=True)
    selected=[]
    for score,box in options:
        if len(selected)>=45:break
        if any(_iou(box,b)>0.58 or _containment(box,b)>.88 for _,b in selected):continue
        selected.append((score,box))
    return [(score,tuple(int(round(c/sf)) for c in b)) for score,b in selected]


def _iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[0]+a[2],b[0]+b[2]);y2=min(a[1]+a[3],b[1]+b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(1,a[2]*a[3]+b[2]*b[3]-inter)


def _containment(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[0]+a[2],b[0]+b[2]);y2=min(a[1]+a[3],b[1]+b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(1,min(a[2]*a[3],b[2]*b[3]))


# Image Processing: Locate and decode license plates
def detect_plates(image, max_plates=3, max_candidates=20):
    if image is None or image.size==0:return []
    H,W=image.shape[:2]
    candidates=_candidate_boxes(image)
    decoded=[]
    for ds,(x,y,w,h) in candidates[:max_candidates]:
        x,y=max(0,x),max(0,y)
        w=min(w,W-x);h=min(h,H-y)
        if w<20 or h<8:continue
        crop=image[y:y+h,x:x+w]
        txt,conf,meta=recognize_plate(crop)
        plausible=len(txt) in range(7,12) and txt[:2] in STATE_CODES
        rank=ds*.47+conf*.43+(.10 if plausible else 0)
        decoded.append((rank,PlateResult((x,y,w,h),txt,round(conf,3),round(ds,3),
                                          'read' if conf>=.60 and plausible else 'uncertain',crop)))
    decoded.sort(key=lambda q:q[0],reverse=True)
    results=[]
    for _,result in decoded:
        if any(_iou(result.box,r.box)>.28 or _containment(result.box,r.box)>.70 for r in results):continue
        if len(results)>0 and _<decoded[0][0]*.72:continue
        results.append(result)
        if len(results)>=max_plates:break
    return results


# Image Processing: Draw bounding boxes and text labels on image
def annotate(image,results,show_confidence=True):
    out=image.copy()
    thick=max(2,round(max(image.shape[:2])/350))
    for i,r in enumerate(results):
        x,y,w,h=r.box
        color=(48,201,116) if r.status=='read' else (24,170,255)
        cv2.rectangle(out,(x,y),(x+w,y+h),color,thick)
        label=r.text if r.text else 'NOT READABLE'
        if show_confidence:label+=f' ({r.confidence:.2f})'
        font_size=max(.48,min(1.,image.shape[1]/900))
        (tw,th),baseline=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,font_size,2)
        top=max(th+baseline+4,y-6)
        cv2.rectangle(out,(x,max(0,top-th-baseline-6)),(min(out.shape[1]-1,x+tw+10),top+1),color,-1)
        cv2.putText(out,label,(x+4,top-baseline-3),cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,(18,24,30),2,cv2.LINE_AA)
    return out
