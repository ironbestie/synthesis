"""Validate every row and audio link before creating the submission."""
import collections, csv, json, zipfile
from pathlib import Path
from prepare import ROOT,MODELS,load
def package(draft=False):
    report_name='report_draft.pdf' if draft else 'report.pdf'
    expected={report_name,'README.md','requirements.txt','summary.csv','listening_sample.csv','listening.html'}
    if not draft:expected.add('listening_notes.json')
    files=[ROOT/f for f in expected]
    assert all(p.is_file() for p in files)
    checklist=load(ROOT/'listening_sample.csv')
    assert len(checklist)==20
    if not draft:
        notes=json.loads((ROOT/'listening_notes.json').read_text())
        assert all(notes.get(r['model']+'/'+r['id'],'').strip() for r in checklist)
    files+=list((ROOT/'code').glob('*.py'))+list((ROOT/'code').glob('*.txt'))
    versions=ROOT/'belarusian/outetts/run_versions.json'
    if versions.is_file():files.append(versions)
    files.append(ROOT/'belarusian/silero/preprocessing_errors.csv')
    files+=[p for p in (ROOT/'figures').iterdir() if p.is_file()]
    summaries=load(ROOT/'summary.csv')
    assert len(summaries)==84
    for lang,models in MODELS.items():
        for m in models:
            base=ROOT/lang/m; rows=load(base/'results.csv');texts=load(base/'texts.csv')
            ids={r['id'] for r in texts}; assert len(ids)==len(rows)==300
            assert ids=={r['id'] for r in rows}=={r['id'] for r in load(base/'phonemes.csv')}
            per=collections.Counter(r['k'] for r in texts)
            assert per=={'':125,**{str(k):25 for k in [1,2,4,8,16,32,64]}}
            files += [base/f for f in ['results.csv','texts.csv','phonemes.csv','synthesis.csv']]
            for r in rows:
                assert r['status'] in ['ok','asr_error','tts_error']
                if r['status']=='ok':
                    assert float(r['cer'])>=0 and float(r['wer'])>=0 and r['audio']
                else: assert r['reason'] and not r['cer'] and not r['wer']
                if r['audio']:
                    path=base/r['audio'];assert path.exists() and path.stat().st_size>44;files.append(path)
            if m in ['omnivoice','outetts']:
                refs=load(base/'references/references.csv')
                assert all(r['reference_id'] in {ref['id'] for ref in refs} for r in rows)
                files.extend(p for p in (base/'references').iterdir() if p.is_file())
    target=ROOT/('draft' if draft else 'submission')/'Ermilova_Dina_MTI2531.zip'
    target.parent.mkdir(exist_ok=True)
    tmp=target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in sorted(set(files)):
            relpath='report.pdf' if p==ROOT/report_name else p.relative_to(ROOT).as_posix()
            z.write(p,'lab2_Ermilova_Dina_MTI2531/'+relpath)
    with zipfile.ZipFile(tmp) as z: assert z.testzip() is None
    tmp.replace(target);print(target)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true')
    package(parser.parse_args().draft)
