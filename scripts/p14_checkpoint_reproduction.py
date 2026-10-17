"""Exact-state/trajectory reconstruction of the EXISTING P14 cohort, not new evidence.

The scientific runner is imported without editing its source. Instrumentation records evaluation
returns and serializes/restores each donor before the frozen runner continues. A full run must
match every original curve and all P/C/D fingerprints; failure is a reproduction discrepancy,
never a new scientific result or a reason to change the registered design.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['prereg/phase14.json', 'scripts/p14_actor_repair.py',
           'src/rled/agents/ppo.py', 'src/rled/surgery.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_sources():
    out = {}
    for name in SOURCES:
        data = (ROOT / name).read_bytes()
        assert data == subprocess.check_output(['git', 'show', '6262b2b:' + name], cwd=ROOT), name
        out[name] = sha(data)
    return out


def frozen_group(seeds):
    result = json.loads((ROOT / 'results/summary/p14_single_result-analysis.json').read_text())
    for name in result['summaries']:
        data = (ROOT / name).read_bytes()
        raw = json.loads(data)
        if raw['seeds'] == seeds:
            assert sha(data) == result['source_hashes'][name]
            return name, raw
    raise ValueError('Not one of the eight fixed original seed groups')


def pack_cfg(smoke=False):
    return dict(ENV_NAME='CartPole-v1' if smoke else 'Asterix-MinAtar', HIDDEN=128,
                DEFICIT='reward_flip', DEFICIT_PARAMS={'prob': 1.},
                TOTAL_TIMESTEPS=(40 if smoke else 640)*8192)


def compose_primary(P, D, code):
    """Exactly mirror the frozen runner's pause reference/context; train-state step stays D."""
    from rled.surgery import compose_ppo
    ref = dict(D, train_state=D['train_state'].replace(
        params=P['train_state'].params, opt_state=P['train_state'].opt_state))
    ts = compose_ppo(ref, D, code)['train_state']
    return dict(D, train_state=ts)


def integrity_checks(P, C, D):
    import p14_actor_repair as scientific
    from rled.agents.ppo import ACTOR_KEYS, CRITIC_KEYS
    checks = {}
    states = {code: compose_primary(P, D, code) for code in ['DDD', 'CDD', 'CCC', 'DCC']}
    eq = scientific.equal
    checks['DDD_equals_full_untouched_D'] = eq(states['DDD'], D)
    for code, state in states.items():
        actor_source = P if code[0] == 'C' else D
        critic_source = P if code[1] == 'C' else D
        opt_source = P if code[2] == 'C' else D
        checks[code + '_actor_parameters_match_donor'] = all(eq(
            state['train_state'].params['params'][k], actor_source['train_state'].params['params'][k]) for k in ACTOR_KEYS)
        checks[code + '_critic_parameters_match_donor'] = all(eq(
            state['train_state'].params['params'][k], critic_source['train_state'].params['params'][k]) for k in CRITIC_KEYS)
        checks[code + '_Adam_state_matches_donor'] = eq(state['train_state'].opt_state, opt_source['train_state'].opt_state)
        checks[code + '_train_state_step_matches_D'] = eq(state['train_state'].step, D['train_state'].step)
        checks[code + '_all_non_train_state_fields_match_D'] = eq(
            {k:v for k,v in state.items() if k != 'train_state'},
            {k:v for k,v in D.items() if k != 'train_state'})
    assert all(checks.values()), checks
    return checks


def evaluation_specs(protocol):
    w0, w1 = protocol['window']; E = protocol['eval_every']
    specs = []
    for j, update in enumerate(range(w0, w1+1, E)):
        for arm, key in [('C',10000), ('D',10000), ('C',54321)]:
            specs.append(dict(phase='fork', arm=arm, update=update, eval_key=key, index=j))
    for cell in protocol['cells']:
        name = '/'.join(cell[k] for k in ['reference','context','opt','code'])
        for j, update in enumerate(range(w1,w1+protocol['recovery']+1,E)):
            for key in ([10000,54321] if name in protocol['second_eval_cells'] else [10000]):
                specs.append(dict(phase='recovery', cell=name, update=update, eval_key=key, index=j))
    return specs


def expected_mean(raw, spec):
    if spec['phase'] == 'fork':
        part = raw['fork_eval'] if spec['eval_key']==10000 else raw['fork_eval_alt']
        return part[spec['arm']][spec['index']]
    field = 'eval_return_mean' if spec['eval_key']==10000 else 'eval_alt_return_mean'
    return [row[spec['index']] for row in raw['cells'][spec['cell']][field]]


def full(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    import p14_actor_repair as scientific

    scientific_hashes = checked_sources()
    smoke = args.mode == 'smoke'
    seeds = args.seeds
    if smoke:
        assert seeds == [99994], 'Software-only smoke is fixed and excluded from inference'
        original_name, original = None, None
    else:
        assert len(seeds)==4
        original_name, original = frozen_group(seeds)
    proto = json.loads((ROOT/'prereg/phase14.json').read_text())
    if smoke:proto = dict(proto, window=[10,20], recovery=20)
    specs = evaluation_specs(proto)
    archive = Path(args.archive_root).resolve()/f's{seeds[0]}to{seeds[-1]}'
    archive.mkdir(parents=True,exist_ok=True)
    tag=f'repro-checkpoints-s{seeds[0]}to{seeds[-1]}-20261017'
    assert not list((ROOT/'results/summary').glob('*p14-'+tag+'-*.json')), 'Fresh reconstruction must not resume cells'
    start = time.time()
    meta = dict(schema='p14-reconstructed-runner-v1', role='same-seed exact reconstruction, not independent replication',
                created_utc=datetime.now(timezone.utc).isoformat(), smoke=smoke, seeds=seeds,
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                registered_scientific_source_sha256=scientific_hashes, config=pack_cfg(smoke),
                frozen_reference_summary=original_name,
                frozen_reference_summary_sha256=sha((ROOT/original_name).read_bytes()) if original_name else None,
                serialization='Flax state_dict/msgpack, gzip mtime0; all dynamic runner fields; static functions rebuilt from pinned code',
                reconstructed_not_originally_retained=True, status='running', checkpoints={},
                episode_data_scope='reconstructed fixed-batch episode returns; original frozen release retained only means',
                packages={name:metadata.version(name) for name in ['jax','jaxlib','flax','optax','gymnax','numpy','scipy','msgpack']})
    def save_meta():
        (archive/'bundle_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    save_meta()
    original_fp=scientific.fingerprint
    captured={}; labels=iter(['C','D','P'])
    def capture(rs):
        label=next(labels)
        fp=original_fp(rs)
        if original is not None:
            assert fp==original['state_fingerprints'][label], f'Donor {label} differs from frozen fingerprint; STOP'
        payload=serialization.to_bytes(rs)
        restored=serialization.from_bytes(rs,payload)
        restored=jax.tree.map(jnp.asarray,restored)
        assert scientific.equal(restored,rs) and original_fp(restored)==fp
        compressed=gzip.compress(payload,compresslevel=5,mtime=0)
        filename=label+'.msgpack.gz';(archive/filename).write_bytes(compressed)
        schema=[]
        for path,value in jax.tree.leaves_with_path(rs):
            value=np.asarray(value)
            schema.append(dict(path=str(path),shape=list(value.shape),dtype=str(value.dtype)))
        meta['checkpoints'][label]=dict(file=filename,sha256=sha(compressed),bytes=len(compressed),
                                       numerical_tree_fingerprint=fp,leaf_schema=schema,
                                       exact_frozen_fingerprint=(not smoke),serialization_roundtrip_equal=True)
        # Continue ALL cells from deserialized numerical donors, with unchanged static source functions.
        rs.clear();rs.update(restored);captured[label]=rs
        if len(captured)==3:
            meta['field_integrity_checks']=integrity_checks(captured['P'],captured['C'],captured['D'])
        save_meta();print(f'checkpoint {label}: exact fingerprint and serialization round trip PASS',flush=True)
        return fp
    scientific.fingerprint=capture
    episodes=[];record_specs=[]
    class JaxProxy:
        def __getattr__(self,name):return getattr(jax,name)
        @staticmethod
        def device_get(value):
            array=np.asarray(jax.device_get(value))
            i=len(episodes);assert i<len(specs)
            assert array.shape==(len(seeds),64) and np.isfinite(array).all(), array.shape
            spec=specs[i]
            if original is not None:
                assert np.array_equal(array.mean(1),np.asarray(expected_mean(original,spec))), f'Evaluation mismatch call {i}: {spec}'
            episodes.append(array.copy());record_specs.append(spec)
            return array
    scientific.jax=JaxProxy()
    oldargv=sys.argv
    sys.argv=['p14_actor_repair.py','--seeds',*map(str,seeds),'--tag',tag]+(['--smoke'] if smoke else [])
    try:scientific.main()
    finally:sys.argv=oldargv
    raw_paths=list((ROOT/'results/summary').glob('*p14-'+tag+'-*.json'));assert len(raw_paths)==1
    raw=json.loads(raw_paths[0].read_text())
    assert raw['status']=='complete' and raw['resumed_from'] is None
    assert len(episodes)==len(specs)
    if original is not None:
        for field in ['seeds','protocol','eval_updates','fork_eval','fork_eval_alt','state_fingerprints','cells','checks']:
            assert raw[field]==original[field],f'Final mismatch: {field}'
    np.savez_compressed(archive/'fixed_batch_episode_returns.npz',returns=np.stack(episodes))
    (archive/'episode_index.json').write_text(json.dumps(record_specs,indent=2)+'\n')
    (archive/'recomputed_raw_summary.json').write_text(json.dumps(raw,indent=1)+'\n')
    if original is not None:(archive/'frozen_reference_summary.json').write_text(json.dumps(original,indent=1)+'\n')
    (archive/'requirements-repro.txt').write_bytes((ROOT/'requirements-repro.txt').read_bytes())
    meta.update(status='complete',all_cells_exactly_match_frozen_reference=not smoke,
                all_continuations_used_deserialized_donors=True,eval_call_count=len(episodes),
                wallclock_s=round(time.time()-start,1),recomputed_manifest=raw['manifest'],
                episode_archive_sha256=sha((archive/'fixed_batch_episode_returns.npz').read_bytes()))
    save_meta()
    (archive/'README.txt').write_text('These numerical runner states were RECONSTRUCTED in 2026-10-17, not retained from the original run.\nP=onset390; C=clean440; D=impaired440 after removal (smoke uses10/20).\nEach fingerprint covers the complete four-seed batched numerical tree.\nStatic apply/optimizer functions require the pinned source/template.\nAll donor round trips and full-cell continuations match the frozen release exactly when smoke=false.\nUse scripts/p14_checkpoint_reproduction.py --mode replay --bundle PATH --seeds A B C D.\nThis is not a new independent training-seed cohort.\n')
    bundle=archive.parent/(archive.name+'.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_STORED) as z:
        for path in sorted(archive.iterdir()):z.write(path,path.name)
    # Keep small audit artifacts in versionable results, large bundles outside the repository.
    outdir=ROOT/'results/reproduction/p14';outdir.mkdir(parents=True,exist_ok=True)
    audit=dict(meta,bundle_file=bundle.name,bundle_bytes=bundle.stat().st_size,bundle_sha256=sha(bundle.read_bytes()),
               original_reference=original_name)
    (outdir/(archive.name+'-audit.json')).write_text(json.dumps(audit,indent=2)+'\n')
    (outdir/(archive.name+'-recomputed.json')).write_bytes(raw_paths[0].read_bytes())
    raw_paths[0].unlink()
    print('EXACT RECONSTRUCTION COMPLETE',json.dumps({k:audit[k] for k in ['seeds','smoke','bundle_bytes','bundle_sha256','status']}),flush=True)


def replay(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from rled.agents.ppo import make,make_eval
    import p14_actor_repair as scientific
    checked_sources()
    bundle=Path(args.bundle)
    with zipfile.ZipFile(bundle) as z:
        meta=json.loads(z.read('bundle_manifest.json'))
        assert not meta['smoke'] and meta['seeds']==args.seeds and meta['status']=='complete'
        cfg,init,run=make(meta['config'])
        template=jax.jit(jax.vmap(init))(jnp.stack([jax.random.PRNGKey(s) for s in args.seeds]))
        donors={}
        for label in ['P','C','D']:
            compressed=z.read(meta['checkpoints'][label]['file'])
            assert sha(compressed)==meta['checkpoints'][label]['sha256']
            state=serialization.from_bytes(template,gzip.decompress(compressed))
            state=jax.tree.map(jnp.asarray,state)
            assert scientific.fingerprint(state)==meta['checkpoints'][label]['numerical_tree_fingerprint']
            donors[label]=state
        raw=json.loads(z.read('frozen_reference_summary.json'))
        observed=np.load(io.BytesIO(z.read('fixed_batch_episode_returns.npz')),allow_pickle=False)['returns']
        specs=json.loads(z.read('episode_index.json'))
    field_checks=integrity_checks(donors['P'],donors['C'],donors['D'])
    rv=jax.jit(jax.vmap(lambda r:run(r,10)))
    evv=jax.jit(jax.vmap(make_eval(cfg,num_envs=64),in_axes=(0,None)))
    checks=[];t=time.time()
    for code in args.cells:
        r=compose_primary(donors['P'],donors['D'],code)
        keyname='pause/damaged/donor/'+code
        for j,update in enumerate(raw['eval_updates']):
            for key in [10000,54321]:
                values=np.asarray(jax.device_get(evv(r['train_state'].params,jax.random.PRNGKey(key))['ret']))
                i=next(i for i,s in enumerate(specs) if s.get('cell')==keyname and s['index']==j and s['eval_key']==key)
                assert np.array_equal(values,observed[i]), f'Independent loader replay differs {code}/{update}/{key}'
                assert np.array_equal(values.mean(1),np.asarray(expected_mean(raw,specs[i])))
            if j<len(raw['eval_updates'])-1:r,_=rv(r)
        checks.append(code);print('independent checkpoint loader/episode replay PASS',code,flush=True)
    output=ROOT/'results/reproduction/p14'/f's{args.seeds[0]}to{args.seeds[-1]}-independent-replay.json'
    output.write_text(json.dumps(dict(role='independent post-deserialization replay, not a new experimental contrast',
        seeds=args.seeds,cells=checks,evaluation_keys=[10000,54321],all21_checkpoints_episode_arrays_exact=True,
        all_field_integrity_checks=field_checks,bundle_sha256=sha(bundle.read_bytes()),wallclock_s=round(time.time()-t,1)),indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['full','smoke','replay'],default='full')
    ap.add_argument('--seeds',nargs='+',type=int,required=True)
    ap.add_argument('--archive-root',default='/home/user/build/p14-reconstruction/archives')
    ap.add_argument('--bundle');ap.add_argument('--cells',nargs='+',default=['DDD','CDD'])
    args=ap.parse_args()
    if args.mode=='replay':replay(args)
    else:full(args)


if __name__=='__main__':main()
