"""Record reproducible public repository discovery, including access failures."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import requests

p=argparse.ArgumentParser();p.add_argument('--raw-out',type=Path,required=True)
p.add_argument('--summary',type=Path,required=True);a=p.parse_args()
a.raw_out.mkdir(parents=True,exist_ok=True);results=[]
for i,query in enumerate(['DARROW wastewater','Tilburg wastewater','Sobhani wastewater']):
    row={'query':query,'endpoint':'https://api.github.com/search/repositories'}
    try:
        response=requests.get(row['endpoint'],params={'q':query,'per_page':20},timeout=25,
            headers={'Accept':'application/vnd.github+json','User-Agent':'Threemool-research-source-audit'})
        (a.raw_out/f'repository_search_{i}.json').write_bytes(response.content)
        row.update({'status':response.status_code,'response_sha256':hashlib.sha256(response.content).hexdigest()})
        if response.ok:
            body=response.json();row['total_count']=body['total_count']
            row['repositories']=[{'name':item['full_name'],'url':item['html_url'],'description':item['description']} for item in body['items']]
        else:row['error']=response.text[:300]
    except requests.RequestException as error:row['error']=str(error)
    results.append(row);print(json.dumps(row),flush=True)
result={'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'queries':results,
    'limitation':'Repository metadata search is incomplete and does not establish absence of compatible public code.'}
a.summary.write_text(json.dumps(result,indent=2))
