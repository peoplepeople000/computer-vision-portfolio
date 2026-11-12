"""Image-based affine registration, independent of segmentation masks."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from .data import read_rows, read_image, image_path, parse_instances, split_id, write_rows
from .matching import centroids
from .segmentation import normalize


def sift_affine(iv_image, ex_image):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(2026)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.01, edgeThreshold=12)
    images = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for image in (iv_image,ex_image):
        scaled = (normalize(image)*255).astype(np.uint8)
        images.append(clahe.apply(scaled))
    ka, da = sift.detectAndCompute(images[0],None)
    kb, db = sift.detectAndCompute(images[1],None)
    diagnostic={"method":"sift_affine","iv_features":len(ka),"ex_features":len(kb)}
    if da is None or db is None or min(len(da),len(db))<6:
        return None,dict(diagnostic,status="insufficient_features")
    # RootSIFT improves descriptor normalization without training on this dataset.
    da=np.sqrt(da/(da.sum(axis=1,keepdims=True)+1e-8)).astype(np.float32)
    db=np.sqrt(db/(db.sum(axis=1,keepdims=True)+1e-8)).astype(np.float32)
    neighbors=cv2.BFMatcher(cv2.NORM_L2).knnMatch(da,db,k=2)
    good=[pair[0] for pair in neighbors if len(pair)==2 and pair[0].distance<0.8*pair[1].distance]
    diagnostic['candidate_matches']=len(good)
    if len(good)<6:
        return None,dict(diagnostic,status="insufficient_descriptor_matches")
    source=np.float32([ka[m.queryIdx].pt for m in good])
    target=np.float32([kb[m.trainIdx].pt for m in good])
    matrix,inliers=cv2.estimateAffine2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=4,
                                      maxIters=10000,confidence=0.999,refineIters=20)
    if matrix is None:
        return None,dict(diagnostic,status="ransac_failed")
    inliers=inliers.ravel().astype(bool)
    singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
    support=int(inliers.sum())
    diagnostic.update(inliers=support, singular_values=singular.tolist(),matrix=matrix.tolist())
    if support<6 or singular.min()<0.4 or singular.max()>2.0 or np.linalg.det(matrix[:,:2])<=0:
        return None,dict(diagnostic,status="rejected_transform")
    errors=np.linalg.norm(source[inliers]@matrix[:,:2].T+matrix[:,2]-target[inliers],axis=1)
    return matrix,dict(diagnostic,status="registered",median_residual=float(np.median(errors)))


def assign_pairs(iv_masks,ex_masks,matrix,radius=6.0,ambiguity_ratio=0.8):
    if matrix is None:
        return []
    ia,a=centroids(iv_masks)
    ib,b=centroids(ex_masks)
    if not len(a) or not len(b):
        return []
    costs=cdist(a@matrix[:,:2].T+matrix[:,2],b)
    augmented=np.full((len(a),len(b)+len(a)),radius)
    augmented[:,:len(b)]=costs
    rows,columns=linear_sum_assignment(augmented)
    pairs=[]
    for i,j in zip(rows,columns):
        if j>=len(b) or costs[i,j]>=radius:
            continue
        alternatives=np.r_[np.delete(costs[i],j),np.delete(costs[:,j],i)]
        if len(alternatives) and costs[i,j]>ambiguity_ratio*alternatives.min():
            continue
        pairs.append([ia[i],ib[j]])
    return pairs


def run(args):
    rows=read_rows(args.predictions)
    output,diagnostics=[],[]
    root=Path(args.data)
    for n,(sid,row) in enumerate(rows.items(),1):
        if args.subject and split_id(sid)[0]!=args.subject:
            continue
        images={m:read_image(image_path(root,args.split,sid,m)) for m in ('invivo','exvivo')}
        matrix,diagnostic=sift_affine(images['invivo'],images['exvivo'])
        masks={m:parse_instances(row,m,images[m].shape) for m in images}
        row['match_pairs']=assign_pairs(masks['invivo'],masks['exvivo'],matrix,args.radius)
        output.append(row)
        diagnostic.update(sample_id=sid,pairs=len(row['match_pairs']))
        diagnostics.append(diagnostic)
        print(f"[{n}/{len(rows)}] {sid}: {diagnostic['status']}, {len(row['match_pairs'])} pairs",flush=True)
    if not output:
        raise ValueError('No selected samples')
    write_rows(output,args.output)
    manifest=vars(args).copy()
    manifest.update(regions=diagnostics,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(str(args.output)+'.registration.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--split',choices=['training','hidden_test'],default='training')
    parser.add_argument('--subject')
    parser.add_argument('--radius',type=float,default=6.0)
    run(parser.parse_args())


if __name__=='__main__':
    main()
