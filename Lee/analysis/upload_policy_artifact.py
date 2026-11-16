"""Upload an immutable verified archive to the existing private draft release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--gh',type=Path,required=True)
p.add_argument('--archive',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
p.add_argument('--release-id',type=int,default=443333191);p.add_argument('--repo',default='UZO1226/Threemool')
a=p.parse_args();manifest=json.loads(a.index.read_text());expected=manifest['archive']
assert not manifest['source_data_included'] and not manifest['residual_banks_included'] and not manifest['pit_banks_included']
assert a.archive.name==expected['name'] and a.archive.stat().st_size==expected['bytes']
assert hashlib.sha256(a.archive.read_bytes()).hexdigest()==expected['sha256']
gh=str(a.gh.resolve());endpoint=f'repos/{a.repo}/releases/{a.release_id}'
def get():return json.loads(subprocess.check_output([gh,'api',endpoint],text=True,encoding='utf-8'))
release=get();assert release['draft']
if not any(asset['name']==a.archive.name for asset in release['assets']):
    subprocess.run([gh,'release','upload',release['tag_name'],str(a.archive.resolve()),'--repo',a.repo],check=True)
release=get();assert release['draft']
asset=next(asset for asset in release['assets'] if asset['name']==a.archive.name)
assert asset['size']==expected['bytes'] and asset['digest']=='sha256:'+expected['sha256']
manifest['remote']={'release_id':release['id'],'tag':release['tag_name'],'asset_id':asset['id'],
    'size_verified':True,'sha256_verified':True,'draft':True}
a.index.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest['remote']),flush=True)
