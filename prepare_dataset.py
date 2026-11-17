"""Extract only still image files (.png, .jpg, .jpeg) and paired XML from a ZIP.

Explicitly exclude all paths inside video_images and similarly named video folders.
The XML is evaluation data, NEVER an input to the detector/recognizer.
"""
from __future__ import annotations
import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath

IMAGE_EXT={'.jpg','.jpeg','.png'}
SKIPPED_VIDEO_DIRS={'video_images','video','videos','video_frames','video-frames'}


def image_files(folder):
    folder=Path(folder)
    return sorted(p for p in folder.rglob('*') if p.is_file()
                  and p.suffix.lower() in IMAGE_EXT
                  and not any(x.lower() in SKIPPED_VIDEO_DIRS for x in p.parts))


def extract(zip_path, target='data', max_uncompressed_mb=1300):
    zip_path=Path(zip_path)
    target=Path(target)
    target.mkdir(parents=True, exist_ok=True)
    n_img=n_xml=n_skip=0
    with zipfile.ZipFile(zip_path) as archive:
        entries=archive.infolist()
        # Limit archive size to reduce accidental zip-bomb risk.
        if sum(e.file_size for e in entries)>max_uncompressed_mb*1024*1024:
            raise ValueError('ZIP expands beyond configured size limit')
        has_google=any(e.filename.startswith('google_images/') for e in entries)
        for item in entries:
            if item.is_dir():continue
            name=item.filename.replace('\\','/')
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or not path.parts:continue
            if any(part.lower() in SKIPPED_VIDEO_DIRS for part in path.parts):
                n_skip+=1; continue
            if has_google and path.parts[0]!='google_images':continue
            ext=path.suffix.lower()
            if ext not in IMAGE_EXT|{'.xml'}:continue
            # Never permit Windows drive letters etc.
            if any(':' in part for part in path.parts):continue
            out=target.joinpath(*path.parts)
            out.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(item) as fi,open(out,'wb') as fo:
                shutil.copyfileobj(fi,fo)
            if ext=='.xml':n_xml+=1
            else:n_img+=1
    return {'images':n_img,'annotations':n_xml,'video_images_excluded':n_skip,
            'destination':str(target.resolve())}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip',required=True,help='Path to your original dataset ZIP')
    ap.add_argument('--output',default='data',help='Folder to extract into')
    args=ap.parse_args()
    print(extract(args.zip,args.output))
