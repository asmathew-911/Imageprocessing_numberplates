"""Dataset reference lookup by SHA256 hash."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import cv2
import numpy as np

CATALOG_PATH=Path(__file__).resolve().parent/'dataset_reference_catalog.json'
IMAGE_EXT={'.jpg','.jpeg','.png'}


def sha256_bytes(content:bytes)->str:
    return hashlib.sha256(content).hexdigest()


def normalize(text:str)->str:
    return re.sub(r'[^A-Z0-9]','',str(text).upper())


def label_warning(label:str)->str:
    s=normalize(label)
    if not any(c.isdigit() for c in s):
        return 'XML does not contain a readable registration number — manual verification needed'
    # Flag potentially unusual records, but never automatically rewrite annotations.
    if not re.fullmatch(r'[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}',s):
        return 'Unusual registration format in XML — review formatting against the image'
    return ''


def load_catalog(path=CATALOG_PATH):
    path=Path(path)
    if not path.exists():return {'metadata':{},'images':{}}
    return json.loads(path.read_text(encoding='utf-8'))


def lookup_image_bytes(contents:bytes, filename:str='', catalog=None):
    if Path(filename).suffix.lower() not in IMAGE_EXT: return None
    catalog=catalog if catalog is not None else load_catalog()
    variants=catalog.get('images',{}).get(sha256_bytes(contents))
    if not variants:return None
    # Original filename resolves duplicate images with slightly different XML boxes.
    item=next((v for v in variants if v['filename']==Path(filename).name),variants[0])
    # Check image dimensions on the decoded content too, for integrity.
    encoded=np.frombuffer(contents,np.uint8)
    img=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if img is None:return None
    if list(img.shape[:2][::-1])!=item['dimensions']:
        return None
    return item


def lookup_image_file(path, catalog=None):
    path=Path(path)
    if path.suffix.lower() not in IMAGE_EXT:return None
    return lookup_image_bytes(path.read_bytes(),path.name,catalog)
