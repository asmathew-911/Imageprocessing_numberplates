"""PlateVision v5 local website. Run from the extracted project folder: python app.py.

There are two EXPLICITLY SEPARATE modes:
 * classical: detects and attempts to read plates from image pixels, with no XML aid.
 * reference: displays XML ground-truth, and is NOT an OCR result.
"""
from __future__ import annotations
import base64
import threading
import uuid
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from anpr import imread, detect_plates, annotate, recognize_plate
from batch import run_batch, ground_truth
from reference_mode import reference_entries, annotate_xml, run_reference_audit
from dataset_catalog import lookup_image_bytes, lookup_image_file, load_catalog
from prepare_dataset import extract, image_files

BASE=Path(__file__).resolve().parent
DATA=BASE/'data'
RESULTS=BASE/'results'
TEMP=BASE/'uploads'
for folder in (DATA,RESULTS,TEMP):folder.mkdir(parents=True,exist_ok=True)
app=FastAPI(title='PlateVision: classical no-ML number plate recognition')
app.mount('/static',StaticFiles(directory=str(BASE/'static')),name='static')
_manifest=image_files(DATA)
_catalog=load_catalog()
_lock=threading.Lock()
_state={'running':False,'done':0,'total':0,'current':'','summary':None,'error':None,'mode':'classical'}


def _image_data(image,max_width=1150):
    if image is None or image.size==0:return ''
    h,w=image.shape[:2]
    if w>max_width:
        image=cv2.resize(image,(max_width,round(h*max_width/w)),interpolation=cv2.INTER_AREA)
    ok,buf=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,82])
    return 'data:image/jpeg;base64,'+base64.b64encode(buf).decode() if ok else ''


def _analysis(image,reference_path=None,mode='classical'):
    if image is None:raise HTTPException(400,'Unreadable image. Use JPG, JPEG or PNG.')
    if mode=='reference':
        if reference_path is None:
            raise HTTPException(400,'XML reference mode only works on named dataset samples, not arbitrary uploads.')
        gt=ground_truth(reference_path) or []
        entries=reference_entries(reference_path,image)
        for e in entries:
            x,y,w,h=e['box']
            e['crop_image']=_image_data(image[max(0,y):max(0,y+h),max(0,x):max(0,x+w)],430)
        return {
            'mode':'reference','predictions':[],
            'reference_entries':entries,
            'annotated_image':_image_data(annotate_xml(image,gt)),
            'reference_for_evaluation_only':None,
            'annotation_notice':'This text and bounding box come directly from dataset XML. No automatic plate detection or OCR was performed.'
        }
    predictions=detect_plates(image,max_plates=3)
    return {
        'mode':'classical',
        'predictions':[{**res.info(),'crop_image':_image_data(res.crop,430),
                        'character_diagnostics':recognize_plate(res.crop)[2].get('characters',[])}
                       for res in predictions],
        'reference_entries':[],
        'annotated_image':_image_data(annotate(image,predictions)),
        'reference_for_evaluation_only':None,
        'annotation_notice':'Image-only inference completed BEFORE XML labels were read. XML appears only as a separate evaluation reference.'
    }


def _catalog_analysis(image, item):
    # This mode is exact XML ANNOTATION RETRIEVAL, not prediction.
    if image is None:
        raise HTTPException(400,'Unreadable JPG/JPEG/PNG file.')
    truth=[(a['text'],tuple(a['box'])) for a in item['annotations']]
    entries=[]
    for annotation in item['annotations']:
        x,y,w,h=annotation['box']
        entries.append({
            'text':annotation['text'],'box':annotation['box'],
            'valid_box':True,'review_needed':bool(annotation['review_reason']),
            'review_reason':annotation['review_reason'],
            'source':'Exact SHA256 image match -> supplied XML (NOT OCR)',
            'crop_image':_image_data(image[y:y+h,x:x+w],430)
        })
    return {
        'mode':'reference','recognition_source':'EXACT_DATASET_XML_LOOKUP_NOT_OCR',
        'predictions':[],'reference_entries':entries,
        'annotated_image':_image_data(annotate_xml(image,truth)),
        'reference_for_evaluation_only':None,
        'annotation_notice':'Image bytes matched the supplied dataset. Text and bounding box were retrieved from its XML annotation, NOT independently detected or read from pixels. Labels marked for review may be incorrect or non-registration text.',
    }


@app.get('/',response_class=HTMLResponse)
def home():return (BASE/'templates'/'index.html').read_text(encoding='utf-8')

@app.get('/api/health')
def health():return {'ok':True,'dataset_images':len(_manifest),'catalog_images':_catalog.get('metadata',{}).get('total_source_images',0),'engine':'Classical OpenCV plus separate SHA256-to-XML dataset lookup — NO ML'}

@app.get('/api/catalog/summary')
def catalog_summary():
    m=_catalog.get('metadata',{})
    return {'reference_source':'Original XML, NOT OCR','source_images':m.get('total_source_images',0),
            'unique_byte_patterns':m.get('unique_sha256',0),
            'identical_image_pairs':len(m.get('duplicate_identical_files',[])),
            'manual_review_flags':26,'label_match_is_not_ocr':True}

@app.get('/api/catalog/audit')
def catalog_audit():
    path=RESULTS/'ALL_442_DATASET_CHECK.csv'
    if not path.exists():raise HTTPException(404,'Built-in audit CSV missing from project ZIP.')
    return FileResponse(path,filename='ALL_442_DATASET_CHECK.csv',media_type='text/csv')

@app.get('/api/gallery')
def gallery(offset:int=0,limit:int=24):
    offset=max(0,offset);limit=max(1,min(48,limit))
    return {'total':len(_manifest),'images':[
        {'id':i,'name':_manifest[i].name}
        for i in range(offset,min(len(_manifest),offset+limit))]}

@app.get('/api/photo/{index}')
def photo(index:int):
    if index<0 or index>=len(_manifest):raise HTTPException(404,'Image not found')
    return FileResponse(_manifest[index])

@app.post('/api/analyze/{index}')
def analyze(index:int,mode:Literal['classical','reference']='classical'):
    if index<0 or index>=len(_manifest):raise HTTPException(404,'Image not found')
    p=_manifest[index]
    img=imread(p)
    if mode=='reference':
        item=lookup_image_file(p,_catalog)
        if item is not None:return _catalog_analysis(img,item)
        return _analysis(img,p,'reference')
    result=_analysis(img,p,'classical')
    
    return result

@app.post('/api/upload')
async def upload(file:UploadFile=File(...),mode:Literal['classical','reference']='classical'):
    if Path(file.filename or '').suffix.lower() not in {'.jpg','.jpeg','.png'}:
        raise HTTPException(415,'Choose JPG, JPEG, or PNG (no videos).')
    contents=await file.read()
    if len(contents)>24*1024*1024:raise HTTPException(413,'Image exceeds 24 MB')
    if not contents:raise HTTPException(400,'Empty image file')
    img=cv2.imdecode(np.frombuffer(contents,np.uint8),cv2.IMREAD_COLOR)
    if img is None:raise HTTPException(400,'Could not decode the image; use a genuine JPG, JPEG or PNG.')
    if mode=='reference':
        item=lookup_image_bytes(contents,file.filename or '',_catalog)
        if item is not None:return _catalog_analysis(img,item)
        raise HTTPException(404,'No matching reference annotation was found for this image. Choose Automatic image processing to analyze the pixels.')
    result=_analysis(img,mode='classical')
    
    return result

@app.post('/api/import-zip')
async def import_zip(file:UploadFile=File(...)):
    global _manifest
    if not (file.filename or '').lower().endswith('.zip'):
        raise HTTPException(415,'Select a ZIP archive.')
    with _lock:
        if _state['running']:raise HTTPException(409,'Finish current batch before importing.')
    path=TEMP/(uuid.uuid4().hex+'.zip')
    try:
        size=0
        with path.open('wb') as stream:
            while part:=await file.read(2*1024*1024):
                size+=len(part)
                if size>500*1024*1024:raise HTTPException(413,'ZIP exceeds 500 MB')
                stream.write(part)
        info=extract(path,DATA)
        _manifest=image_files(DATA)
        # Small, deterministic audit of the paired annotations; no OCR is claimed.
        from verify_all_442 import verify_all
        check=verify_all(DATA,RESULTS/'IMPORTED_DATASET_CHECK.csv')
        return {**info,'total_available':len(_manifest),'annotation_check':check}
    except HTTPException:raise
    except Exception as exc:raise HTTPException(400,f'Import failed: {exc}') from exc
    finally:path.unlink(missing_ok=True)


def _batch_worker(mode):
    try:
        if mode=='reference':
            def progress(i,n,name,*c):
                with _lock:_state.update(done=i,total=n,current=name,annotated=c[0],matched=c[1],exact=None)
            results=run_reference_audit(DATA,RESULTS,progress=progress,save_annotated=True)
        else:
            def progress(i,n,name,ann,loc,exact):
                with _lock:_state.update(done=i,total=n,current=name,annotated=ann,matched=loc,exact=exact)
            results=run_batch(DATA,RESULTS,progress=progress,save_annotated=True)
        with _lock:_state.update(running=False,summary=results)
    except Exception as exc:
        with _lock:_state.update(running=False,error=str(exc))

@app.post('/api/batch/start')
def batch_start(mode:Literal['classical','reference']='classical'):
    if not _manifest:raise HTTPException(400,'Import your dataset ZIP first.')
    with _lock:
        if _state['running']:raise HTTPException(409,'A batch is already running.')
        _state.update(running=True,done=0,total=len(_manifest),current='',summary=None,
                      error=None,matched=0,annotated=0,exact=0,mode=mode)
    threading.Thread(target=_batch_worker,args=(mode,),daemon=True).start()
    return {'started':True,'total':len(_manifest),'mode':mode}

@app.get('/api/batch/status')
def batch_status():
    with _lock:return dict(_state)

@app.get('/api/batch/download')
def batch_download(mode:Literal['classical','reference']='classical'):
    file=RESULTS/('reference_audit.csv' if mode=='reference' else 'predictions.csv')
    if not file.exists():raise HTTPException(404,'Run that batch mode before downloading.')
    return FileResponse(file,filename=file.name,media_type='text/csv')

if __name__=='__main__':
    print('\nOpen this website: http://127.0.0.1:8000\n')
    uvicorn.run(app,host='127.0.0.1',port=8000)
