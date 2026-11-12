"""Compare an inference result with an independently supplied reference CSV."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('generated',type=Path);p.add_argument('reference',type=Path);a=p.parse_args()
def sha(f):
 with f.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
r={'generated_sha256':sha(a.generated),'reference_sha256':sha(a.reference)};r['byte_identical']=r['generated_sha256']==r['reference_sha256'];print(json.dumps(r,indent=2))
raise SystemExit(0 if r['byte_identical'] else 1)
