from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RADII = torch.arange(1.,14.,2.,device=DEVICE)
ANGLES = torch.arange(32,device=DEVICE)*(2*np.pi/32)
POLAR_OFFSETS = torch.stack((RADII[:,None]*ANGLES.cos(),RADII[:,None]*ANGLES.sin()),dim=-1)
CONTRAST_ANGLES = torch.arange(0.,360.,10.,device=DEVICE)
CONTRAST_SCALES = torch.tensor([0.65,0.8,1.0,1.2,1.5],device=DEVICE)
yy,xx = torch.meshgrid(torch.arange(-14.,15.,device=DEVICE),torch.arange(-14.,15.,device=DEVICE),indexing='ij')
mask = xx.square()+yy.square()<=196
CONTRAST_OFFSETS = torch.stack((xx[mask],yy[mask]),dim=1)


def detail_image(image):
    return ndi.gaussian_filter(image, 0.8, mode='reflect') - ndi.gaussian_filter(image, 4.0, mode='reflect')

@torch.inference_mode()
def sample_offsets(image_tensor, centers, offsets):
    h, w = image_tensor.shape[-2:]
    grid = centers[:, None, :] + offsets.reshape(1, -1, 2)
    grid[..., 0] = 2 * grid[..., 0] / (w - 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / (h - 1) - 1
    return F.grid_sample(image_tensor, grid[None], mode='bilinear', padding_mode='reflection', align_corners=True)[0, 0]

@torch.inference_mode()
def polar_features(samples):
    polar = samples.reshape(-1, len(RADII), len(ANGLES))
    polar = polar - polar.mean(dim=(1, 2), keepdim=True)
    spectrum = torch.fft.rfft(polar, dim=-1) / len(ANGLES)
    feature = torch.cat((spectrum[..., :1].real, spectrum[..., 1:7].abs()), dim=-1).flatten(1)
    return F.normalize(feature, dim=1, eps=1e-6)

@torch.inference_mode()
def build_search_index(image, stride=4, scales=(0.8, 1.0, 1.2), batch_size=4096):
    tensor = torch.as_tensor(detail_image(image), device=DEVICE)[None, None]
    ys, xs = torch.meshgrid(torch.arange(0, image.shape[0], stride, device=DEVICE), torch.arange(0, image.shape[1], stride, device=DEVICE), indexing='ij')
    centers = torch.stack((xs.flatten(), ys.flatten()), dim=1).float()
    features = []
    for scale in scales:
        chunks = []
        for start in range(0, len(centers), batch_size):
            samples = sample_offsets(tensor, centers[start:start+batch_size], POLAR_OFFSETS * scale)
            chunks.append(polar_features(samples))
        features.append(torch.cat(chunks))
    return {'detail': tensor, 'centers': centers, 'features': features, 'scales': scales, 'stride': stride}

@torch.inference_mode()
def coarse_candidates(index, patch, top_k=256):
    query = torch.as_tensor(detail_image(patch), device=DEVICE)[None, None]
    center = torch.tensor([[(patch.shape[1]-1)/2, (patch.shape[0]-1)/2]], device=DEVICE)
    feature = polar_features(sample_offsets(query, center, POLAR_OFFSETS))
    scores = torch.stack([(f @ feature.T).flatten() for f in index['features']]).max(dim=0).values
    best = scores.topk(min(top_k, len(scores)))
    return index['centers'][best.indices], best.values

@torch.inference_mode()
def normalized_polar(samples):
    p = samples.reshape(-1, len(RADII), len(ANGLES))
    p = p - p.mean(dim=(1, 2), keepdim=True)
    return p / p.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)

@torch.inference_mode()
def add_phase_index(index, batch_size=4096):
    index['spectra'] = []
    for scale in index['scales']:
        chunks = []
        for start in range(0, len(index['centers']), batch_size):
            p = sample_offsets(index['detail'], index['centers'][start:start+batch_size], POLAR_OFFSETS * scale)
            chunks.append(torch.fft.rfft(normalized_polar(p), dim=-1))
        index['spectra'].append(torch.cat(chunks))

@torch.inference_mode()
def phase_candidates(index, patch, top_k=4096, batch_size=8192):
    q = torch.as_tensor(detail_image(patch), device=DEVICE)[None, None]
    center = torch.tensor([[(patch.shape[1]-1)/2, (patch.shape[0]-1)/2]], device=DEVICE)
    qs = torch.fft.rfft(normalized_polar(sample_offsets(q, center, POLAR_OFFSETS)), dim=-1)
    scores = torch.full((len(index['centers']),), -1., device=DEVICE)
    for spectra in index['spectra']:
        for start in range(0, len(scores), batch_size):
            cross = (spectra[start:start+batch_size] * qs.conj()).sum(dim=1)
            corr = torch.fft.irfft(cross, n=len(ANGLES), dim=-1).max(dim=1).values
            scores[start:start+len(corr)] = torch.maximum(scores[start:start+len(corr)], corr)
    best = scores.topk(min(top_k, len(scores)))
    return index['centers'][best.indices], best.values

def local_contrast(image):
    residual = ndi.gaussian_filter(image, 0.5, mode='reflect') - ndi.gaussian_filter(image, 2.0, mode='reflect')
    energy = ndi.gaussian_filter(residual * residual, 3.0, mode='reflect')
    return residual / np.sqrt(energy + 4.0)

@torch.inference_mode()
def contrast_scores(index, patch, centers, angles, scales, batch_size=2048):
    q = torch.as_tensor(local_contrast(patch), device=DEVICE)[None, None]
    qc = torch.tensor([[(patch.shape[1]-1)/2, (patch.shape[0]-1)/2]], device=DEVICE)
    qv = sample_offsets(q, qc, CONTRAST_OFFSETS)[0]
    qv = F.normalize(qv-qv.mean(), dim=0, eps=1e-6)
    h, w = index['detail'].shape[-2:]
    results = []
    for start in range(0, len(centers), batch_size):
        a = torch.deg2rad(angles[start:start+batch_size])[:, None]
        s = scales[start:start+batch_size, None]
        dx, dy = CONTRAST_OFFSETS[:, 0][None], CONTRAST_OFFSETS[:, 1][None]
        offset = torch.stack((s*(a.cos()*dx-a.sin()*dy), s*(a.sin()*dx+a.cos()*dy)), dim=-1)
        grid = centers[start:start+batch_size, None, :] + offset
        grid[..., 0] = grid[..., 0]*2/(w-1)-1
        grid[..., 1] = grid[..., 1]*2/(h-1)-1
        v = F.grid_sample(index['detail'], grid[None], align_corners=True, padding_mode='reflection')[0, 0]
        v = F.normalize(v-v.mean(dim=1, keepdim=True), dim=1, eps=1e-6)
        results.append(v @ qv)
    return torch.cat(results)

@torch.inference_mode()
def build_contrast_index(image, stride=2, scales=(0.8,1.0,1.2), batch_size=4096):
    tensor = torch.as_tensor(local_contrast(image), device=DEVICE)[None,None]
    ys, xs = torch.meshgrid(torch.arange(16,image.shape[0]-16,stride,device=DEVICE), torch.arange(16,image.shape[1]-16,stride,device=DEVICE), indexing='ij')
    centers = torch.stack((xs.flatten(),ys.flatten()),dim=1).float()
    features = []
    for scale in scales:
        chunks = []
        for start in range(0,len(centers),batch_size):
            chunks.append(polar_features(sample_offsets(tensor,centers[start:start+batch_size],POLAR_OFFSETS*scale)))
        features.append(torch.cat(chunks))
    return dict(detail=tensor,centers=centers,features=features,scales=scales,stride=stride)

@torch.inference_mode()
def contrast_candidates(index,patch,top_k=4096):
    q = torch.as_tensor(local_contrast(patch),device=DEVICE)[None,None]
    center = torch.tensor([[(patch.shape[1]-1)/2,(patch.shape[0]-1)/2]],device=DEVICE)
    f = polar_features(sample_offsets(q,center,POLAR_OFFSETS))
    scores = torch.stack([(v@f.T).flatten() for v in index['features']]).max(dim=0).values
    best = scores.topk(min(top_k,len(scores)))
    return index['centers'][best.indices],best.values

@torch.inference_mode()
def dense_contrast_match(index, patch, xy):
    c = torch.as_tensor(xy, device=DEVICE).float()
    aa, ss = torch.meshgrid(CONTRAST_ANGLES, CONTRAST_SCALES, indexing='ij')
    aa, ss = aa.flatten(), ss.flatten()
    scores = contrast_scores(index, patch, c.repeat_interleave(len(aa), 0), aa.repeat(len(c)), ss.repeat(len(c)))
    per_center = scores.reshape(len(c), -1).max(dim=1)
    seeds = per_center.values.topk(min(128, len(c))).indices
    dx, dy, da, ds = torch.meshgrid(torch.arange(-2., 2.1, 0.5, device=DEVICE), torch.arange(-2., 2.1, 0.5, device=DEVICE), torch.arange(-5., 6., 2., device=DEVICE), torch.tensor([-0.1,-0.05,0.,0.05,0.1], device=DEVICE), indexing='ij')
    shift = torch.stack((dx.flatten(),dy.flatten()),dim=1)
    lc = (c[seeds,None]+shift[None]).reshape(-1,2)
    la = (aa[per_center.indices[seeds],None]+da.flatten()[None]).flatten()
    ls = (ss[per_center.indices[seeds],None]+ds.flatten()[None]).flatten()
    fine = contrast_scores(index, patch, lc, la, ls)
    best = fine.argmax()
    far = torch.linalg.vector_norm(lc-lc[best], dim=1) >= 32
    return {'x': lc[best,0].item(), 'y': lc[best,1].item(), 'score': fine[best].item(), 'alternative_score': fine[far].max().item() if far.any() else float('nan'), 'angle': la[best].item(), 'scale': ls[best].item()}

@torch.inference_mode()
def diverse_contrast_match(index, patch, xy):
    c = torch.as_tensor(xy, device=DEVICE).float()
    aa, ss = torch.meshgrid(CONTRAST_ANGLES, CONTRAST_SCALES, indexing='ij')
    aa, ss = aa.flatten(), ss.flatten()
    scores = contrast_scores(index, patch, c.repeat_interleave(len(aa), 0), aa.repeat(len(c)), ss.repeat(len(c)))
    per_center = scores.reshape(len(c), -1).max(dim=1)
    old_seeds = per_center.values.topk(min(128, len(c))).indices.cpu().numpy()
    order = per_center.values.argsort(descending=True).cpu().numpy()
    positions = c.cpu().numpy()
    diverse = []
    for candidate in order:
        if not diverse or np.linalg.norm(positions[diverse]-positions[candidate],axis=1).min() >= 4.0:
            diverse.append(int(candidate))
            if len(diverse) >= 128:
                break
    seeds = torch.as_tensor(np.unique(np.r_[old_seeds,diverse]),device=DEVICE,dtype=torch.long)
    dx, dy, da, ds = torch.meshgrid(torch.arange(-2., 2.1, 0.5, device=DEVICE), torch.arange(-2., 2.1, 0.5, device=DEVICE), torch.arange(-5., 6., 2., device=DEVICE), torch.tensor([-0.1,-0.05,0.,0.05,0.1], device=DEVICE), indexing='ij')
    shift = torch.stack((dx.flatten(),dy.flatten()),dim=1)
    lc = (c[seeds,None]+shift[None]).reshape(-1,2)
    la = (aa[per_center.indices[seeds],None]+da.flatten()[None]).flatten()
    ls = (ss[per_center.indices[seeds],None]+ds.flatten()[None]).flatten()
    fine = contrast_scores(index, patch, lc, la, ls)
    best = fine.argmax()
    far = torch.linalg.vector_norm(lc-lc[best], dim=1) >= 32
    return {'x': lc[best,0].item(), 'y': lc[best,1].item(), 'score': fine[best].item(), 'alternative_score': fine[far].max().item() if far.any() else float('nan'), 'angle': la[best].item(), 'scale': ls[best].item()}

def extract_pattern_nodes(path):
    rgba = np.array(Image.open(path).convert('RGBA'))
    rgb = rgba[...,:3].astype(np.float32)
    marker = (rgba[...,3] > 128) & (rgb.min(axis=2) > 180) & (np.ptp(rgb,axis=2) < 35)
    labels, count = ndi.label(marker)
    areas = np.bincount(labels.ravel())
    ids = [i for i in range(1,count+1) if areas[i] >= 3]
    centers = ndi.center_of_mass(marker, labels, ids)
    xy = np.array([(x,y) for y,x in centers],dtype=np.float32).reshape(-1,2)
    return rgba, xy

def partial_assignment(mapped, points, tolerance):
    cost = np.linalg.norm(mapped[:, None, :] - points[None, :, :], axis=-1)
    # A missed correspondence costs more than all possible inlier residuals combined.
    penalty = (min(cost.shape) + 1) * tolerance
    rows, cols = linear_sum_assignment(np.where(cost <= tolerance, cost, penalty))
    keep = cost[rows, cols] <= tolerance
    return rows[keep], cols[keep], cost[rows[keep], cols[keep]]

def unique_location_candidates(frame, minimum_distance=36.0):
    kept=[]
    for idx in frame.sort_values('score',ascending=False).index:
        xy=frame.loc[idx,['x','y']].to_numpy(dtype=float)
        if not kept or np.linalg.norm(frame.loc[kept,['x','y']].to_numpy()-xy,axis=1).min()>=minimum_distance:
            kept.append(idx)
    return frame.loc[kept].copy()

def unique_support_pattern_search(points, templates, tolerance=20.0, pool_size=256):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    observed = points[:,0] + 1j * points[:,1]
    oi, oj = np.where(~np.eye(len(points), dtype=bool))
    od = observed[oj] - observed[oi]
    tree = cKDTree(points)
    results = []
    for name, nodes in templates.items():
        if len(nodes) < 3:
            continue
        base = nodes[:,0] + 1j * nodes[:,1]
        ti, tj = np.triu_indices(len(base), 1)
        proposals = []
        for reflected in (False, True):
            z = base.conj() if reflected else base
            for i,j in zip(ti,tj):
                if abs(z[j]-z[i]) < 10:
                    continue
                aa = od/(z[j]-z[i]); bb = observed[oi]-aa*z[i]
                valid = (abs(aa)>=0.5)&(abs(aa)<=30)
                aa,bb = aa[valid],bb[valid]
                if not len(aa):
                    continue
                mapped = aa[:,None]*z+bb[:,None]
                xy = np.stack((mapped.real,mapped.imag),axis=-1)
                dist,nearest = tree.query(xy.reshape(-1,2))
                dist = dist.reshape(len(aa),-1)
                nearest=nearest.reshape(len(aa),-1)
                weights=np.maximum(0,1-(dist/tolerance)**2)
                votes=np.zeros((len(aa),len(points)))
                np.maximum.at(votes,(np.repeat(np.arange(len(aa)),len(nodes)),nearest.ravel()),weights.ravel())
                strength=votes.sum(axis=1)
                for k in np.argsort(strength)[-3:]:
                    proposals.append((float(strength[k]),xy[k]))
        proposals.sort(key=lambda p:p[0],reverse=True)
        best = None
        design = np.column_stack((nodes,np.ones(len(nodes))))
        for _, initial in proposals[:pool_size]:
            mapped = initial.copy()
            for iteration in range(4):
                rows,cols,errors = partial_assignment(mapped,points,tolerance)
                if len(rows)<3:
                    break
                matrix,_,rank,_ = np.linalg.lstsq(design[rows],points[cols],rcond=None)
                singular = np.linalg.svd(matrix[:2],compute_uv=False)
                if rank<3 or singular[-1]<0.3 or singular[0]>35 or singular[0]/singular[-1]>4:
                    break
                mapped = design@matrix
            rows,cols,errors = partial_assignment(mapped,points,tolerance)
            count = len(rows)
            mean_error = float(errors.mean()) if count else tolerance
            quality = float(np.exp(-0.5*(errors/6.0)**2).sum())
            if best is None or quality>best['quality']:
                best=dict(pattern=name,inliers=count,mean_error=mean_error,quality=quality,
                          mapped=mapped,node_indices=rows,point_indices=cols)
        if best is not None:
            results.append(best)
    return sorted(results,key=lambda r:r['quality'],reverse=True)

@torch.inference_mode()
def contrast_phase_candidates(index, patch, top_k=4096, batch_size=8192):
    q = torch.as_tensor(local_contrast(patch), device=DEVICE)[None, None]
    center = torch.tensor([[(patch.shape[1]-1)/2, (patch.shape[0]-1)/2]], device=DEVICE)
    qs = torch.fft.rfft(normalized_polar(sample_offsets(q, center, POLAR_OFFSETS)), dim=-1)
    scores = torch.full((len(index['centers']),), -1., device=DEVICE)
    for spectra in index['spectra']:
        for start in range(0, len(scores), batch_size):
            cross = (spectra[start:start+batch_size] * qs.conj()).sum(dim=1)
            corr = torch.fft.irfft(cross, n=len(ANGLES), dim=-1).max(dim=1).values
            scores[start:start+len(corr)] = torch.maximum(scores[start:start+len(corr)], corr)
    best = scores.topk(min(top_k, len(scores)))
    return index['centers'][best.indices], best.values
