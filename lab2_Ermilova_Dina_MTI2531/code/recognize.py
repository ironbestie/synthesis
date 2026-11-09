import argparse, math, re, time, unicodedata
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from rapidfuzz.distance import Levenshtein
from transformers import AutoProcessor,Wav2Vec2ForCTC
from prepare import ROOT,load,save

def be_number(n, female=False):
    small='нуль адзін два тры чатыры пяць шэсць сем восем дзевяць дзесяць адзінаццаць дванаццаць трынаццаць чатырнаццаць пятнаццаць шаснаццаць сямнаццаць васямнаццаць дзевятнаццаць'.split()
    if female: small[1:3]=['адна','дзве']
    if n<20: return small[n]
    parts=[]
    for value,forms in [(10**9,('мільярд','мільярды','мільярдаў')),(10**6,('мільён','мільёны','мільёнаў')),(1000,('тысяча','тысячы','тысяч'))]:
        q,n=divmod(n,value)
        if q: parts += [be_number(q,value==1000),forms[2 if 11<=q%100<=14 else 0 if q%10==1 else 1 if 2<=q%10<=4 else 2]]
    h,n=divmod(n,100)
    if h: parts.append('сто дзвесце трыста чатырыста пяцьсот шэсцьсот семсот восемсот дзевяцьсот'.split()[h-1])
    if n>=20:
        t,n=divmod(n,10)
        parts.append('дваццаць трыццаць сорак пяцьдзясят шэсцьдзесят семдзесят восемдзесят дзевяноста'.split()[t-2])
    if n: parts.append(small[n])
    return ' '.join(parts)
def normalize(s,lang):
    from num2words import num2words
    s=unicodedata.normalize('NFC',s).casefold()
    s=re.sub(r'\d+',lambda m: be_number(int(m[0])) if lang=='belarusian' else num2words(int(m[0]),lang='id'),s)
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s).split())
def main(folder,device):
    lang=folder.parent.name
    texts={r['id']:r for r in load(folder/'texts.csv')}
    state=load(folder/'synthesis.csv')
    path=ROOT/'models/asr'
    proc=AutoProcessor.from_pretrained(path);model=Wav2Vec2ForCTC.from_pretrained(path).eval().to(device)
    torch.set_num_threads(3)
    target=folder/'results.csv'
    records=load(target) if target.exists() else []
    existing={r['id'] for r in records}
    for r in state:
        if r['id'] in existing:continue
        raw=texts[r['id']]['text'];norm=normalize(raw,lang)
        row=dict(**r,tts=folder.name,asr='omniASR_CTC_300M_v2',text=raw,recognized='',normalized_text=norm,normalized_recognized='',char_errors='',word_errors='',chars=len(norm),words=len(norm.split()),cer='',wer='',asr_seconds='')
        if r['status']=='tts_ok':
            t=time.monotonic()
            try:
                wave,sr=sf.read(folder/r['audio'],dtype='float32')
                if wave.ndim==2:wave=wave.mean(axis=1)
                g=math.gcd(sr,16000)
                wave=resample_poly(wave,16000//g,sr//g)
                hypotheses=[]
                for i in range(0,len(wave),480000):
                    clip=wave[i:i+480000]
                    if len(clip)<400: clip=np.pad(clip,(0,400-len(clip)))
                    inputs=proc(clip,sampling_rate=16000,return_tensors='pt').to(device)
                    with torch.inference_mode(): ids=model(**inputs).logits.argmax(-1)[0].cpu()
                    hypotheses.append(proc.decode(ids).replace('▁',' ').strip())
                hyp=' '.join(hypotheses);h=normalize(hyp,lang)
                ce=Levenshtein.distance(norm,h);we=Levenshtein.distance(norm.split(),h.split())
                row.update(recognized=hyp,normalized_recognized=h,char_errors=ce,word_errors=we,cer=ce/len(norm),wer=we/len(norm.split()),status='ok')
            except Exception as exc:row.update(status='asr_error',reason=repr(exc))
            row['asr_seconds']=round(time.monotonic()-t,3)
        records.append(row);save(target,records)
        if len(records)%25==0:print(folder,len(records),flush=True)
        if len(records)>=3 and all(x['status']=='asr_error' for x in records[-3:]):raise RuntimeError(records[-1]['reason'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--device',default='cpu');a=p.parse_args();main(a.folder,a.device)
