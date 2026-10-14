#!/usr/bin/env python3
"""Owner-side access for the five original pi0 models, without changing sources."""
import json,subprocess
from pathlib import Path
from prepare_shared_assets import share
from check_assets import audit
ROOT=Path(__file__).resolve().parents[1]
def main():
 target=Path.home()/'twostage_share_20261014'
 assert target.is_dir()
 assets=json.loads((target/'assets.json').read_text());models=json.loads((ROOT/'provenance/baseline_model_identities.json').read_text())
 paths={};proof={}
 for key,entry in models.items():
  source=Path(entry['path'])
  try:
   for name in entry['files']:audit(source/name,'yzhong7')
   destination=source
  except AssertionError:
   destination=target/key
  hashes=share(source,destination,entry['files'])
  for name in hashes:audit(destination/name,'yzhong7')
  paths[key]=str(destination);proof[key]={'path':str(destination),'files':hashes}
 assets['base_models']=paths
 (target/'assets.json').write_text(json.dumps(assets,indent=2)+'\n')
 (target/'baseline_shared_proof.json').write_text(json.dumps({'status':'verified','collaborator':'yzhong7','models':proof,'sources_unchanged':True},indent=2)+'\n')
 print(json.dumps(assets,indent=2))
if __name__=='__main__':main()
