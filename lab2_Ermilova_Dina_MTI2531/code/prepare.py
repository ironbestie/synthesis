import csv
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'indonesian': ['mms','supertonic','omnivoice'], 'belarusian':['silero','outetts','omnivoice']}
def load(path):
    with open(path, encoding='utf-8', newline='') as f: return list(csv.DictReader(f))
def save(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(path)
def prepare(archive):
    with zipfile.ZipFile(archive) as z:
        for lang,models in MODELS.items():
            rows=[]
            for file in sorted(n for n in z.namelist() if n.startswith(lang+'/') and n.endswith('.csv')):
                cat=Path(file).stem
                data=list(csv.DictReader(io.StringIO(z.read(file).decode())))
                assert len(data)==25
                for i,r in enumerate(data,1):
                    rep=cat.startswith('03_')
                    words=[s.strip() for s in r['text'].split(',')] if rep else []
                    if rep: assert len(set(words))==1
                    base=words[0] if rep else ''
                    for k in ([1,2,4,8,16,32,64] if rep else ['']):
                        uid=f'{lang[:2]}_{cat[:2]}_{i:02}'+(f'_k{k:02}' if rep else '')
                        rows.append(dict(id=uid,category=cat,text=', '.join([base]*k) if rep else r['text'],source=r['source'],base=base,k=k))
            assert len(rows)==len(set(r['id'] for r in rows))==300
            for m in models: save(ROOT/lang/m/'texts.csv',rows)
if __name__=='__main__': prepare(sys.argv[1])
