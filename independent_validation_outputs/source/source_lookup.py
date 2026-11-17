"""At most six targeted primary-source lookups; never fetch observations.
Existing ledger prevents repeat requests; DOI landing redirects are not evidence of article contents.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError


def main():
    out=Path(__file__).resolve().parents[1];ledger=out/'evidence/source_lookup_log.json'
    records=json.loads(ledger.read_text())['records'] if ledger.exists() else []
    urls=[('barrows_doi','https://doi.org/10.1016/j.envpol.2018.02.062'),
          ('dehm_doi','https://doi.org/10.1016/j.marpolbul.2020.111239'),
          ('barrows_archive','https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:211009'),
          ('dehm_archive','https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:294676'),
          ('barrows_publisher','https://www.sciencedirect.com/science/article/pii/S0269749117349400'),
          ('dehm_publisher','https://www.sciencedirect.com/science/article/pii/S0025326X2030357X')]
    for name,url in urls:
        if name in {r['id'] for r in records}:continue
        if len(records)>=6:raise RuntimeError('Six-lookup cap reached')
        row={'id':name,'url':url,'access_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Named-study primary-source provenance only; not external-data discovery'}
        try:
            request=Request(url,headers={'User-Agent':'Marine-microplastics-reproducibility-audit/1.0'})
            with urlopen(request,timeout=25) as response:
                data=response.read(3000001);row.update(http_status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'),truncated=len(data)>3000000)
                data=data[:3000000]
        except HTTPError as e:
            data=e.read(200000);row.update(http_status=e.code,final_url=e.url,error=str(e),content_type=e.headers.get('Content-Type'))
        except Exception as e:data=b'';row.update(error=repr(e),http_status=None)
        path=out/'evidence'/f'{name}_response.txt';path.write_bytes(data)
        row.update(response_file=str(path.relative_to(out)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        records.append(row);ledger.write_text(json.dumps({'targeted_lookups':len(records),'cap':6,'records':records},indent=2))
        print(name,row.get('http_status'),row.get('error',''),len(data))


if __name__=='__main__':main()
