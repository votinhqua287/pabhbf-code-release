"""Check released alphabets and actual header-free bit packing for stored schemes."""
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from pabhbf.eval.evaluate import Method
from pabhbf.utils.config import ROOT,system_params
from pabhbf.data.dataset import load_split

def packed_bits(z,bits):
    levels=2**bits
    codes=np.rint((z+1)*levels/2-.5).astype(np.int64)
    restored=-1+(codes+.5)*2/levels
    if not np.all((codes>=0)&(codes<levels)) or not np.allclose(restored,z,atol=1e-6):
        raise AssertionError('release is not on the declared quantizer alphabet')
    packed=np.packbits(((codes[:,:,None] >> np.arange(bits-1,-1,-1))&1).reshape(len(z),-1),axis=1)
    assert packed.shape[1]*8==z.shape[1]*bits
    recovered=np.unpackbits(packed,axis=1).reshape(len(z),z.shape[1],bits)@(2**np.arange(bits-1,-1,-1))
    assert np.array_equal(recovered,codes)
    return packed.shape[1]*8

if __name__=='__main__':
    torch.set_num_threads(2)
    H=load_split('default_val',{'Hhat'})['Hhat'][:2,0]
    reports=[]
    with threadpool_limits(limits=2):
        patterns=[('pabhbf',t) for t in ['lam0.01','lam0.03','dz4_lam0','dz4_lam0.03','lam0']]
        patterns += [('baselines',t) for t in ['raw_csi','pca','random_projection','scalar_quant','gaussian_pca_0.2','top_m_beams','autoencoder','angle_domain','laplace_pca_10']]
        for kind,tag in patterns:
            path=sorted((ROOT/f'results/checkpoints/{kind}').glob(f'{tag}_s1_*.pt'))[-1]
            ck=torch.load(path,weights_only=False);method=Method(ck,system_params(ck['config']))
            z=method.represent(H,np.random.default_rng(8)).numpy().reshape(32,-1)
            bits=32 if tag=='raw_csi' else (1 if tag=='scalar_quant' else 4)
            if tag=='top_m_beams':
                rep=method.rep
                coords=np.stack([rep.cb.u/rep.u_max,2*rep.cb.tau/rep.tau_max-1],1).astype(np.float32)
                assert len(np.unique(coords,axis=0))==rep.cb.M
                tokens=z.reshape(len(z),rep.m,3)
                idx=np.argmin(((tokens[:,:,:2,None]-coords.T[None,None,:,:])**2).sum(2),axis=2)
                assert np.array_equal(coords[idx],tokens[:,:,:2])
                gain=np.rint((tokens[:,:,2]+1)*2**bits/2-.5).astype(int)
                stream=np.concatenate([((idx[:,:,None]>>np.arange(rep.idx_bits-1,-1,-1))&1),
                                       ((gain[:,:,None]>>np.arange(bits-1,-1,-1))&1)],axis=2).reshape(len(z),-1)
                wire=np.packbits(stream,axis=1)
                decoded=np.unpackbits(wire,axis=1).reshape(len(z),rep.m,rep.idx_bits+bits)
                assert np.array_equal(decoded[:,:,:rep.idx_bits]@(2**np.arange(rep.idx_bits-1,-1,-1)),idx)
                assert np.array_equal(decoded[:,:,rep.idx_bits:]@(2**np.arange(bits-1,-1,-1)),gain)
                actual=wire.shape[1]*8
            else:
                actual=len(z[0].tobytes())*8 if tag=='raw_csi' else packed_bits(z,bits)
            assert actual==method.payload_bits,(tag,actual,method.payload_bits)
            order='PCA, normalize, Gaussian noise, saturating quantizer' if tag.startswith('gaussian') else (
                'PCA, normalize, clip [-1,1], Laplace(2d/epsilon), saturating quantizer' if tag.startswith('laplace') else 'deterministic release')
            reports.append(dict(scheme=tag,dimension=z.shape[1],bits_per_coordinate=bits,
                quantization_order=order,actual_payload_bits=actual,archived_declared_payload_bits=ck['payload_bits'],
                checkpoint=str(path.relative_to(ROOT))))
    out=ROOT/'results/revision_final/payload_audit.json';out.write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))
