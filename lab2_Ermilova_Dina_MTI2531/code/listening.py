"""Build an offline listening sheet with stratified examples and local audio."""
from html import escape
from pathlib import Path
from base64 import b64encode
import json
from prepare import ROOT, MODELS, load, save
def choose(partial=False, early=None):
    panels=[]
    for lang,models in MODELS.items():
        candidates={}
        for m in models:
            folder=ROOT/lang/m
            if partial and early and m=='outetts' and not (folder/'results.csv').exists(): folder=early
            texts={r['id']:r for r in load(folder/'texts.csv')}
            if partial and not (folder/'synthesis.csv').exists():
                candidates[m]=[];continue
            results={r['id']:r for r in load(folder/'results.csv')} if (folder/'results.csv').exists() else {}
            candidates[m]=[]
            for r in load(folder/'synthesis.csv'):
                if r['status']!='tts_ok':continue
                t=texts[r['id']]
                raw=dict(r,text=t['text'],category=t['category'],k=t['k'],recognized='',cer='',wer='')
                rec=results.get(r['id'],{})
                if not partial and rec.get('status')!='ok':continue
                candidates[m].append(dict(raw,**{key:rec[key] for key in ['recognized','cer','wer'] if key in rec}))
        picked=set()
        def pick(model,kind,rows):
            row=next((r for r in rows if (model,r['id']) not in picked),None)
            if row is None: return
            picked.add((model,row['id']))
            panels.append(dict(language=lang,model=model,id=row['id'],selection=kind,category=row['category'],k=row['k'],audio=f"{lang}/{model}/"+row['audio'],text=row['text'],recognized=row['recognized'],cer=row['cer'],wer=row['wer']))
        for m in models:
            rows=candidates[m]
            pick(m,'Длинный повтор',(r for r in rows if r['k']=='64'))
            if m=='outetts':
                pick(m,'Скороговорка',(r for r in rows if r['id']=='be_01_01'))
                pick(m,'Слоговой ряд',(r for r in rows if r['id']=='be_02_01'))
                continue
            pick(m,'Обычное предложение',(r for r in rows if r['category'].startswith('06_')))
            if len(rows)==300 and all(r['cer']!='' for r in rows) or not partial:
                pick(m,'Высокая CER',sorted((r for r in rows if not r['k'] and r['cer']!=''),key=lambda r:float(r['cer']),reverse=True))
        language_complete = sum(p['language']==lang for p in panels)==9
        for m in (models if language_complete or not partial else []):
            if sum(p['language']==lang for p in panels)>=10: break
            pick(m,'Скороговорка',(r for r in candidates[m] if r['category'].startswith('01_')))
        if not partial: assert sum(p['language']==lang for p in panels)==10
    return panels
def build(partial=False,early=None):
    rows=choose(partial,early)
    save(ROOT/'listening_sample.csv',rows)
    html=['''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Прослушивание записей</title><style>
body{font:16px/1.48 system-ui;margin:2em auto;max-width:820px;padding:0 1em;color:#182f3b}section{border-top:1px solid #bdcbcf;padding:1em 0}textarea{display:block;width:96%;min-height:4em;padding:.5em}audio{max-width:100%;width:480px}small{color:#48606a}button{padding:.7em;margin:1em 0}summary{cursor:pointer}p{overflow-wrap:anywhere}</style>
<h1>Прослушивание записей</h1><p>Для каждого файла запишите, что слышно: пропуск или лишнее слово, число повторов, обрыв, пауза или ровная речь. После прослушивания можно раскрыть расшифровку ASR и сравнить. Если произношение оценить трудно, это можно указать прямо.</p>''']
    for i,r in enumerate(rows):
        e={k:escape(str(v),quote=True) for k,v in r.items()}
        key=e['model']+'/'+e['id']
        audio_data = b64encode((ROOT / r['audio']).read_bytes()).decode('ascii')
        metrics=(f"CER {float(r['cer']):.1%} · WER {float(r['wer']):.1%}" if r['cer']!='' else 'Распознавание ожидается')
        html.append(f"""<section><small>{i+1}/20 · {e['language']} · {e['selection']}</small><h2>{e['model']} / {e['id']}</h2>
<p>{e['text']}</p><audio controls preload="none" src="data:audio/wav;base64,{audio_data}"></audio><details><summary>Распознавание и метрики</summary><p>{e['recognized']}</p>{metrics}</details><label>Что слышно<textarea data-id="{key}"></textarea></label></section>""")
    html.append('''<button id="save">Скачать заметки</button><span id="count"></span><script>
const els=[...document.querySelectorAll('textarea')],key='mtusi-lab2-listening-v28';
let notes={};try{notes=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function track(){document.querySelector('#count').textContent=' Заполнено '+els.filter(e=>e.value.trim()).length+' из '+els.length}
for(const e of els){e.value=notes[e.dataset.id]||'';e.oninput=()=>{notes[e.dataset.id]=e.value;try{localStorage.setItem(key,JSON.stringify(notes))}catch(x){}track()}}track();
document.querySelector('#save').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(notes,null,2)],{type:'application/json'}));a.download='listening_notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}</script></html>''')
    (ROOT/'listening.html').write_text(''.join(html))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--early',type=Path)
    a=p.parse_args();build(a.partial,a.early)
