"""Run the frozen Stage653 pipeline on raw pairs of grayscale TIFF images."""
import argparse,hashlib,json,subprocess,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--images',type=Path,required=True,help='Folder containing subject/region/{invivo,exvivo}.tif')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
 a=p.parse_args();a.images=a.images.resolve();a.output=a.output.resolve()
 a.output.mkdir(parents=True,exist_ok=True)
 os.environ.setdefault("MPLCONFIGDIR",str(a.output/".matplotlib"))
 import torch,numpy as np,tifffile
 device=a.device
 if device=='auto':device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
 pairs=sorted(a.images.glob('*/*/invivo.tif'))
 if not pairs:raise ValueError('No paired image folders found')
 for path in pairs:
  for part in [path.parent.name,path.parent.parent.name]:
   if '__' in part:raise ValueError('Subject and region names must not contain double underscores')
  for m in ['invivo','exvivo']:
   f=path.with_name(m+'.tif');im=tifffile.imread(f)
   if im.ndim!=2 or min(im.shape)<64 or not np.isfinite(im).all():raise ValueError(f'Expected finite grayscale 2D TIFF with both dimensions >=64: {f}')
 manifest=json.loads((ROOT/'weights/manifest.json').read_text())
 for name,info in manifest.items():
  f=ROOT/'weights'/name
  if not f.is_file() or f.stat().st_size!=info['bytes'] or digest(f)!=info['sha256']:raise ValueError(f'Missing or modified frozen checkpoint: {f}')
 print(f'Validated {len(pairs)} image pairs and {len(manifest)} frozen checkpoints; device={device}',flush=True)
 a.output.mkdir(parents=True,exist_ok=True)
 signature={'weights_manifest':digest(ROOT/'weights/manifest.json'),'source':{str(f.relative_to(ROOT)):digest(f) for folder in ['src','raw_engine','vendor'] for f in sorted((ROOT/folder).rglob('*.py'))}}
 lock=a.output/'runtime_signature.json'
 if lock.exists() and json.loads(lock.read_text())!=signature:raise ValueError('Code or weights changed; use a new output folder')
 lock.write_text(json.dumps(signature,indent=2)+'\n')
 start=time.monotonic()
 subprocess.run([sys.executable,str(ROOT/'raw_engine/predict.py'),'--images',str(a.images),'--weights',str(ROOT/'weights'),'--output',str(a.output),'--device',device],check=True)
 sys.path.insert(0,str(ROOT/'src'))
 from cellmatch.data import read_rows,parse_instances
 rows=read_rows(a.output/'submission.csv')
 assert set(rows)=={f'{f.parent.parent.name}__{f.parent.name}' for f in pairs}
 for sid,row in rows.items():
  subject,region=sid.split('__')
  for m in ['invivo','exvivo']:
   shape=tifffile.imread(a.images/subject/region/(m+'.tif')).shape
   masks=parse_instances(row,m,shape);occupied=np.zeros(np.prod(shape),bool)
   for mask in masks.values():
    for lo,hi in zip(mask.starts,mask.ends):
     if occupied[lo:hi].any():raise ValueError('Overlapping instance masks')
     occupied[lo:hi]=True
 report={'rows':len(rows),'instances':{m:sum(len(r[m+'_instances']) for r in rows.values()) for m in ['invivo','exvivo']},'pairs':sum(len(r['match_pairs']) for r in rows.values()),'csv_sha256':digest(a.output/'submission.csv'),'elapsed_seconds':time.monotonic()-start,'device':device,'torch':torch.__version__,'structural_validation':'passed'}
 (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));print('Saved:',a.output/'submission.csv')
if __name__=='__main__':main()
