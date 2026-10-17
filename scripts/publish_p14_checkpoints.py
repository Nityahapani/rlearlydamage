"""Publish ONLY checked numerical-state bundles as release assets; no token persistence.

Requires GITHUB_TOKEN in this process environment. A verified group must have original donor
fingerprints, exact five primary/reference curves, and independent both-key saved-file replay.
The first group also reruns all11 cells. No duplicate seed counts or original-retention claims.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REPO='Nityahapani/rlearlydamage'
TAG='p14-reconstructed-states-20261017'
API='https://api.github.com/repos/'+REPO
EXPECTED_PRIMARY={'fork/clean/donor/CCC','pause/damaged/donor/CCC','pause/damaged/donor/DDD',
                  'pause/damaged/donor/CDD','pause/damaged/donor/DCC'}


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True)
    ap.add_argument('--worktree',default='/home/user/p14-reproduction')
    ap.add_argument('--archive-root',default='/home/user/build/p14-reconstruction/archives')
    ap.add_argument('--tag-commit',default='9ceca9611a9295b2afb0f66c54940a4ed3fa7bc9')
    args=ap.parse_args();assert args.start in range(1200,1232,4)
    token=os.environ.get('GITHUB_TOKEN');assert token,'Set GITHUB_TOKEN ephemerally, never in a file/config'
    seeds=list(range(args.start,args.start+4));name=f's{seeds[0]}to{seeds[-1]}'
    tree=Path(args.worktree);bundle=Path(args.archive_root)/(name+'.zip');digest=sha_file(bundle)
    with zipfile.ZipFile(bundle) as z:
        meta=json.loads(z.read('bundle_manifest.json'))
        raw=json.loads(z.read('recomputed_raw_summary.json'))
        original=json.loads(z.read('frozen_reference_summary.json'))
        assert meta['seeds']==seeds and not meta['smoke'] and meta['status']=='complete'
        assert raw['seeds']==original['seeds']==seeds and raw['resumed_from'] is None
        assert EXPECTED_PRIMARY<=set(raw['cells'])
        count=len(raw['cells']);assert count==(11 if args.start==1200 else 5)
        for key in raw['cells']:assert raw['cells'][key]==original['cells'][key],key
        assert raw['state_fingerprints']==original['state_fingerprints']
        assert all(meta['field_integrity_checks'].values())
        for label in ['P','C','D']:
            p=meta['checkpoints'][label];assert p['exact_frozen_fingerprint'] and p['serialization_roundtrip_equal']
            assert p['numerical_tree_fingerprint']==original['state_fingerprints'][label]
            assert hashlib.sha256(z.read(p['file'])).hexdigest()==p['sha256']
        for source,expected in meta['registered_scientific_source_sha256'].items():
            assert sha_file(ROOT/source)==expected
            assert (ROOT/source).read_bytes()==subprocess.check_output(['git','show','6262b2b:'+source],cwd=ROOT)
    replay_path=tree/'results/reproduction/p14'/(name+'-independent-replay.json')
    replay=json.loads(replay_path.read_text())
    assert replay['seeds']==seeds and replay['cells']==['DDD','CDD'] and replay['evaluation_keys']==[10000,54321]
    assert replay['all21_checkpoints_episode_arrays_exact'] and replay['bundle_sha256']==digest
    assert all(replay['all_field_integrity_checks'].values())
    headers={'Accept':'application/vnd.github+json','Authorization':'Bearer '+token,
             'X-GitHub-Api-Version':'2022-11-28','User-Agent':'rlearlydamage-reproducibility-release'}
    def request(url,method='GET',body=None,allow404=False):
        data=json.dumps(body).encode() if body is not None else None
        h=dict(headers)
        if data is not None:h['Content-Type']='application/json'
        req=urllib.request.Request(url,data=data,headers=h,method=method)
        try:
            with urllib.request.urlopen(req,timeout=60) as f:return json.load(f)
        except urllib.error.HTTPError as e:
            if allow404 and e.code==404:return None
            # Never include the authorization header/command or raw token in an exception.
            raise RuntimeError(f'GitHub release request failed HTTP{e.code} at public endpoint') from None
    release=request(API+'/releases/tags/'+TAG,allow404=True)
    if release is None:
        release=request(API+'/releases','POST',dict(tag_name=TAG,target_commitish=args.tag_commit,
            name='P14 numerical-state reconstructions: verified groups (in progress)',
            body='Same-seed reconstruction of the frozen32-seed Asterix PPO study, not new independent evidence. Only verified groups are uploaded. Index: docs/p14_reconstructed_state_release.json. Reconstructed2026-10-17, not originally retained files. Source/data inherit repository MIT terms; third-party runtime code is installed separately.',
            draft=False,prerelease=True))
    assets=request(API+f"/releases/{release['id']}/assets?per_page=100")
    matching=[a for a in assets if a['name']==bundle.name]
    if matching:
        assert len(matching)==1;asset=matching[0]
        assert asset.get('digest') in [None,'sha256:'+digest],'Existing asset has a different digest; do not overwrite'
    else:
        url=release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(bundle.name)
        command=['curl','--fail-with-body','--silent','--show-error','--request','POST',
            '--header','Authorization: Bearer '+token,'--header','Accept: application/vnd.github+json',
            '--header','Content-Type: application/zip','--header',f'Content-Length: {bundle.stat().st_size}',
            '--upload-file',str(bundle),url]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError('Release asset upload failed; credentials/command intentionally not printed')
        asset=json.loads(result.stdout)
        assert asset['size']==bundle.stat().st_size
        assert asset.get('digest') in [None,'sha256:'+digest],'Server asset digest does not match local file'
    # Check the public bytes as well, including servers that omit an asset digest field.
    public_request=urllib.request.Request(asset['browser_download_url'],headers={'User-Agent':headers['User-Agent']})
    download_hash=hashlib.sha256();download_size=0
    with urllib.request.urlopen(public_request,timeout=120) as response:
        for block in iter(lambda:response.read(1024*1024),b''):
            download_hash.update(block);download_size+=len(block)
    assert download_size==bundle.stat().st_size and download_hash.hexdigest()==digest,'Public download checksum mismatch'
    # Copy only named small provenance files, not large binaries, into the source repository.
    target=ROOT/'results/reproduction/p14';target.mkdir(parents=True,exist_ok=True)
    for suffix in ['-audit.json','-recomputed.json','-independent-replay.json']:
        shutil.copy2(tree/'results/reproduction/p14'/(name+suffix),target/(name+suffix))
    shutil.copy2(tree/'runs'/meta['recomputed_manifest'],ROOT/'runs'/meta['recomputed_manifest'])
    receipt=dict(asset_id=asset['id'],release_id=release['id'],name=asset['name'],size=asset['size'],
                 digest=asset.get('digest'),public_download_sha256=download_hash.hexdigest(),public_download_verified=True,
                 public_download_url=asset['browser_download_url'],
                 published_utc=datetime.now(timezone.utc).isoformat())
    (target/(name+'-publication.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    path=ROOT/'docs/p14_reconstructed_state_release.json'
    if path.exists():index=json.loads(path.read_text())
    else:index=dict(schema='p14-reconstructed-state-release-v1',
        role='later numerical-state reconstruction; same32 seeds, no additional empirical N',
        original_scientific_release='74c6ceb9eb75f3727137a98b42ec7caa685dc325',
        original_full_runner_states_retained=False,reconstructed_creation_date='2026-10-17',
        release_tag=TAG,release_url=release['html_url'],tag_source_commit=args.tag_commit,
        scope='P/C/D donors and all five primary/reference curves in everygroup; extra6 cells rerun only in firstgroup; independent native/restored file replay everygroup',
        serialized_material='all dynamic numerical runner fields; static functions supplied by pinned source/template',
        verified_groups=[])
    record=dict(seeds=seeds,source_commit=meta['source_commit'],recomputed_cell_count=count,
        primary_reference_curves_exact=True,all11_cells_recomputed=(count==11),
        secondary_cells_not_rerun=[k for k in original['cells'] if k not in raw['cells']],
        all_original_donor_fingerprints_exact=True,independent_replay_pass=True,
        original_frozen_summary=meta['frozen_reference_summary'],
        bundle_bytes=bundle.stat().st_size,bundle_sha256=digest,public_download_url=asset['browser_download_url'],
        checkpoint_files=meta['checkpoints'],source_sha256=meta['registered_scientific_source_sha256'],
        audit='results/reproduction/p14/'+name+'-audit.json',
        independent_replay='results/reproduction/p14/'+name+'-independent-replay.json',
        publication_receipt='results/reproduction/p14/'+name+'-publication.json')
    old=[g for g in index['verified_groups'] if g['seeds']==seeds]
    if old:assert old[0]['bundle_sha256']==digest
    index['verified_groups']=[g for g in index['verified_groups'] if g['seeds']!=seeds]+[record]
    index['verified_groups'].sort(key=lambda g:g['seeds'][0])
    allseeds=sum((g['seeds'] for g in index['verified_groups']),[])
    assert len(allseeds)==len(set(allseeds)) and set(allseeds)<=set(range(1200,1232))
    index['verified_seed_count']=len(allseeds);index['verified_group_count']=len(index['verified_groups'])
    index['status']='complete32 donor/primary/replay coverage' if len(allseeds)==32 else 'in progress; only listed groups verified'
    path.write_text(json.dumps(index,indent=2)+'\n')
    body=(f"Verified later reconstructions for {len(allseeds)}/32 original seeds ({len(index['verified_groups'])}/8 groups). "
          'No new independent cohort. P/C/D numerical fingerprints and all five primary/reference curves match the frozen study; the firstgroup reruns all11, latergroups do not claim the extra6 controls independently rerun. '
          'Every listed group passes fresh-process native/restored replay for both fixed evaluation keys at all21 checkpoints. '
          'Dynamic arrays/optimizer/environment/RNG/counters are serialized; static functions require pinned source. '
          'These files were reconstructed2026-10-17, not retained from the original run. '
          'Download/checksum/source index: https://github.com/Nityahapani/rlearlydamage/blob/main/docs/p14_reconstructed_state_release.json . '
          'Repository MIT terms apply to numerical artifacts; runtime libraries retain their own licenses.')
    request(API+f"/releases/{release['id']}",'PATCH',dict(name=f'P14 numerical-state reconstruction: {len(allseeds)}/32 verified seeds',body=body))
    print(json.dumps({'published_seeds':seeds,'public_download_url':asset['browser_download_url'],
                      'sha256':digest,'verified_seed_count':len(allseeds),'release_url':release['html_url']},indent=2))


if __name__=='__main__':main()
