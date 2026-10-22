"""Export one calibration room's first-round peak diagnostic, never full matrices."""
import gzip
import hashlib
import json
from pathlib import Path
from itertools import combinations
import numpy as np
from numba import njit
from indoor_vt.matching import _subset, _delta, _crosses, PatternMatcher
from indoor_vt.observations import native_frames, angle_to_unit, C_MPS
from indoor_vt.dataset import delay_file, unpack
from indoor_vt.reconstruction import neighborhoods

ROOT = Path(__file__).resolve().parents[1]
ROOM = 'rectangle_calibration_00'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8'))


def blob(output, value):
    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    raw_hash = hashlib.sha256(raw).hexdigest()
    path = output/'matching'/f'{raw_hash}.bin.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return dict(file=path.relative_to(output).as_posix(), bytes=path.stat().st_size,
                raw_bytes=len(raw), sha256=sha(path), raw_sha256=raw_hash)


@njit(cache=True)
def first_round(a0, b0):
    ia, ib = np.argsort(a0, kind='mergesort'), np.argsort(b0, kind='mergesort')
    a, b = a0[ia], b0[ib]
    if len(a) > len(b):
        keep = _subset(a,b,False)
        a, ia = a[keep], ia[keep]
    elif len(b) > len(a):
        keep = _subset(b,a,False)
        b, ib = b[keep], ib[keep]
    if not len(a) or not len(b):
        return ia,ib,np.empty((len(a),len(b))),np.empty((0,3))
    delta = _delta(a,b,2,.5,False,-1.,0.)
    candidates = []
    for i in range(len(a)):
        j = np.argmin(delta[i])
        if np.argmin(delta[:,j]) == i:
            candidates.append((delta[i,j],int(ia[i]),int(ib[j])))
    candidates.sort()
    out = np.empty((len(candidates),3))
    count = 0
    for cost,i,j in candidates:
        good = True
        for old in range(count):
            if _crosses(a0[i],b0[j],a0[int(out[old,0])],b0[int(out[old,1])],False):
                good = False
                break
        if good:
            out[count,0],out[count,1],out[count,2] = i,j,cost
            count += 1
    return ia,ib,delta,out[:count]


def correlation(values):
    n,pos,sx,sxx,sxy = values
    denom = (n*sxx-sx*sx)*(n*pos-pos*pos)
    return float((n*sxy-sx*pos)/np.sqrt(denom)) if denom > 0 else None


def add_moments(s, x, y):
    s += [x.size, y.sum(), x.sum(), np.square(x).sum(), x[y].sum()]


def metrics(tp, predicted, available):
    return dict(tp=int(tp), fp=int(predicted-tp), fn=int(available-tp),
                precision=float(tp/predicted) if predicted else 0.,
                recall=float(tp/available) if available else 0.,
                f1=float(2*tp/(predicted+available)) if predicted+available else 0.)


def build(root=ROOT):
    base, output = root/'data/development', root/'published/v6'
    design = read(base/'design.json')
    room = next(r for r in design['rooms'] if r['id']==ROOM)
    assert room['split']=='calibration'
    truth = read(base/'evaluation/source_truth.json')['sources'][ROOM]
    positions = np.asarray(room['survey_xyz'])
    pairs = sorted({tuple(map(int,pair)) for ids in neighborhoods(positions,'local',10) for pair in combinations(ids,2)})
    scene = dict(room=ROOM, label='Rectangle 01', polygon_xy=room['polygon_xy'],
                 survey_xyz=room['survey_xyz'], aps=room['access_points_xyz'], pairs=pairs)
    index = dict(schema='indoor-vt.peak-matching.v1', room=ROOM, room_label='Rectangle 01',
        track='direct', method='MPUrge', round=1, half_window=2, alpha=.5,
        normalized_pattern=False, neighbors=10, scene=blob(output,scene),
        source='src/indoor_vt/matching.py', source_sha256=sha(root/'src/indoor_vt/matching.py'),
        native={}, conditions=[], noise_levels=design['config']['direct_sigma_m'],
        reference_count=len(positions), pairs_per_ap=len(pairs))
    fingerprints = {}
    for path in (base/'design.json',base/'evaluation/source_truth.json',root/'src/indoor_vt/matching.py'):
        fingerprints[str(path.relative_to(root))] = sha(path)
    replayed = checks = 0
    validation = []
    production = PatternMatcher()
    for ap in ('A','B'):
        scenario = base/'scenarios'/ROOM/ap/'survey'
        acquisition = read(scenario/'input_manifest.json')
        assert acquisition['position_indices']==list(range(len(positions)))
        native_path = scenario/'native/qdOutput.json.gz'
        fingerprints[str(native_path.relative_to(root))] = sha(native_path)
        frames = list(native_frames(native_path))
        references = np.asarray([s['xyz'] for s in truth[ap]])
        native, labels = [], []
        for pos, frame in zip(positions,frames):
            xyz = pos+(frame.delay_s*C_MPS)[:,None]*angle_to_unit(frame.aoa_az_deg,frame.aoa_el_deg)
            d = np.linalg.norm(xyz[:,None,:]-references[None,:,:],axis=2)
            assert np.all(np.sum(d<.01,axis=1)==1), 'Source identity is ambiguous'
            ids = d.argmin(1)
            assert len(set(ids))==len(ids), 'Duplicate paths require explicit scoring handling'
            labels.append(ids)
            native.append(dict(source_ids=ids.tolist(), gain_db=frame.gain_db.tolist()))
        index['native'][ap] = blob(output,dict(profiles=native,sources=truth[ap]))
        for sigma in index['noise_levels']:
            directory = scenario/'observations/direct'/str(sigma)
            manifest = read(directory/'manifest.json')
            path = delay_file(directory)
            fingerprints[str(path.relative_to(root))] = sha(path)
            fingerprints[str((directory/'manifest.json').relative_to(root))] = sha(directory/'manifest.json')
            with np.load(path) as data:
                rows = unpack(data['ranges_m'],data['offsets'])
            profiles, source_rows = [], []
            for frame, row, ids, measure in zip(frames,rows,labels,manifest['measurements']):
                rng = np.random.Generator(np.random.PCG64(measure['seed']))
                clean = frame.delay_s*C_MPS
                measured = clean.copy() if sigma==0 else clean+rng.normal(0,sigma,len(clean))
                keep = np.flatnonzero(measured>=0)
                order = keep[np.argsort(measured[keep],kind='stable')]
                np.testing.assert_array_equal(measured[order],row)
                profiles.append(dict(ranges=row.tolist(),native_indices=order.tolist()))
                source_rows.append(ids[order]); replayed += 1
            counts=[]; moments=np.zeros((2,5)); hist={}; total=np.zeros(3,dtype=np.int64)
            for pair_id,(left,right) in enumerate(pairs):
                ia,ib,delta,matches = first_round(rows[left],rows[right])
                la,lb = source_rows[left],source_rows[right]
                y = (la[ia,None]==lb[None,ib]).ravel()
                x = delta.ravel()
                good = la[matches[:,0].astype(int)]==lb[matches[:,1].astype(int)]
                available = np.intersect1d(la,lb).size
                counts.append([int(good.sum()),len(matches),int(available)])
                total += counts[-1]
                add_moments(moments[0],x,y)
                add_moments(moments[1],matches[:,2],good)
                for bid in np.unique(np.floor(x/.25).astype(int)):
                    selected = (x>=bid*.25)&(x<(bid+1)*.25)
                    h=hist.setdefault(int(bid),np.zeros(3))
                    h += [selected.sum(),y[selected].sum(),x[selected].sum()]
                if pair_id%137==0:
                    np.testing.assert_array_equal(matches,production.match_array(rows[left],rows[right])[:len(matches)])
                    checks += 1
                    validation.append(dict(ap=ap,noise=sigma,left=left,right=right,
                        ia=ia.tolist(),ib=ib.tolist(),delta=delta.tolist(),matches=matches.tolist()))
            summary=dict(**metrics(*total),candidate_cells=int(moments[0,0]),
                         candidate_same_source=int(moments[0,1]),candidate_correlation=correlation(moments[0]),
                         accepted_correlation=correlation(moments[1]),pairs=len(pairs))
            pack=dict(profiles=profiles,pair_counts=counts,
                      bins=[[bid*.25,float(v[2]/v[0]),int(v[0]),int(v[1])] for bid,v in sorted(hist.items())])
            index['conditions'].append(dict(ap=ap,noise=sigma,data=blob(output,pack),summary=summary))
            print(f'{ap} sigma={sigma}: first-round F1={summary["f1"]:.5f}',flush=True)
    write(output/'matching/index.json',index)
    for name,expected in fingerprints.items():
        assert sha(root/name)==expected
    audit=root/'runs/development/publication-checkpoint/peak-matching'
    write(audit/'validation.json',validation)
    write(audit/'export-audit.json',dict(input_hashes=fingerprints,unchanged=True,
         exact_replay_rows=replayed,production_prefix_checks=checks,exporter_sha256=sha(Path(__file__))))
    manifest=read(output/'manifest.json')
    # Refresh only this feature's descriptors; preserve every existing result asset.
    files=[r for r in manifest['files'] if not r['file'].startswith('matching/')]
    files += [dict(file=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted((output/'matching').iterdir()) if p.is_file()]
    files.sort(key=lambda x:x['file'])
    manifest.update(files=files,total_bytes=sum(r['bytes'] for r in files))
    write(output/'manifest.json',manifest)
    print(f'Matching export: {sum(r["bytes"] for r in files if r["file"].startswith("matching/")):,} bytes',flush=True)


if __name__=='__main__':
    build()
