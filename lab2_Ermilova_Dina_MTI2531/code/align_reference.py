"""Viterbi alignment of the published FLEURS transcript with the assigned CTC model."""
import json
import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, Wav2Vec2ForCTC
from prepare import ROOT, load
from recognize import normalize

def align():
    folder=ROOT/'belarusian/outetts/references'
    ref=load(folder/'references.csv')[0]
    wave,rate=sf.read(folder/ref['filename'],dtype='float32')
    assert rate==16000 and wave.ndim==1
    text=normalize(ref['transcript'],'belarusian')
    model_path=ROOT/'models/asr'
    processor=AutoProcessor.from_pretrained(model_path)
    model=Wav2Vec2ForCTC.from_pretrained(model_path).eval()
    torch.set_num_threads(3)
    with torch.inference_mode():
        scores=model(**processor(wave,sampling_rate=rate,return_tensors='pt')).logits[0].log_softmax(-1).numpy()
    vocab=processor.tokenizer.get_vocab()
    labels=np.zeros(2*len(text)+1,dtype=int)
    labels[1::2]=[vocab[c] for c in text]
    values=np.full(len(labels),-np.inf)
    values[0]=0
    back=[]
    for frame in scores:
        step=np.full((3,len(labels)),-np.inf)
        step[0]=values
        step[1,1:]=values[:-1]
        eligible=(labels[2:]!=0)&(labels[2:]!=labels[:-2])
        step[2,2:]=np.where(eligible,values[:-2],-np.inf)
        back.append(step.argmax(0))
        values=step.max(0)+frame[labels]
    state=len(labels)-2+int(values[-1]>values[-2])
    assert np.isfinite(values[state])
    path=[]
    for choices in reversed(back):
        path.append(state)
        state-=choices[state]
    states=np.array(path[::-1])
    dt=len(wave)/rate/len(states)
    words=[]
    cursor=0
    for word in text.split():
        indexes=np.where((states>=2*cursor+1)&(states<=2*(cursor+len(word))-1))[0]
        words.append(dict(word=word,start=float(indexes[0]*dt),end=float((indexes[-1]+1)*dt)))
        cursor+=len(word)+1
    (folder/'alignment.json').write_text(json.dumps(dict(text=text,words=words),ensure_ascii=False,indent=2))
    print('Published transcript:',ref['transcript'])
    print('Greedy ASR:',processor.decode(scores.argmax(-1)))
if __name__=='__main__': align()
