"""Rebuild aggregate scores and figures from per-utterance results."""
from collections import Counter
from itertools import groupby
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import ROOT, MODELS, load

LANG={'indonesian':'Индонезийский','belarusian':'Белорусский'}
CATS=['Скороговорки','Слоги','Повторы','Сложные слова','Числа','Предложения']
FEATURES={'chars':'Символы','words':'Слова','phoneme_count':'Фонемы','word_repeat':'Доля повторных слов','k':'Число повторов','vowels':'Доля гласных','cons_run':'Макс. цепочка согласных','bigram_repeat':'Доля повторных пар фонем'}
VOWELS=frozenset('aeiouəɛɔɨɑɐɪʊ')
def features(row,ph):
    tokens=[x for x in ph.split() if x!='|']
    vowels=[x[0] in VOWELS for x in tokens]
    max_run=0
    bigrams=[]
    for w in ph.split('|'):
        phones=w.split()
        bigrams.extend(zip(phones,phones[1:]))
        for v,group in groupby(phones,key=lambda p:p[0] in VOWELS):
            if not v: max_run=max(max_run,sum(1 for _ in group))
    words=row['normalized_text'].split()
    return dict(phoneme_count=len(tokens), word_repeat=1-len(set(words))/len(words),
                vowels=sum(vowels)/len(tokens),consonants=1-sum(vowels)/len(tokens),
                cons_run=max_run,bigram_repeat=1-len(set(bigrams))/len(bigrams) if bigrams else 0)
def summarize(g):
    ok=g[g.status=='ok']
    return dict(planned=len(g),successful=len(ok),tts_errors=(g.status=='tts_error').sum(),
                asr_errors=(g.status=='asr_error').sum(),technical_failures=(g.status!='ok').sum(),
                micro_cer=ok.char_errors.sum()/ok.chars.sum() if len(ok) else None,
                micro_wer=ok.word_errors.sum()/ok.words.sum() if len(ok) else None,
                macro_cer=ok.cer.mean(),macro_wer=ok.wer.mean(),exact_match=(ok.char_errors==0).mean())
def run():
    figdir=ROOT/'figures'
    figdir.mkdir(exist_ok=True)
    summary=[]; correlations=[]; observations=[]; frequencies=[]
    for lang, models in MODELS.items():
        for model in models:
            folder=ROOT/lang/model
            t={r['id']:r for r in load(folder/'texts.csv')}
            p={r['id']:r['phonemes'] for r in load(folder/'phonemes.csv')}
            records=load(folder/'results.csv')
            assert len(records)==len(t)==len({r['id'] for r in records})==300
            for r in records:
                info=t[r['id']]
                r.update(language=lang,model=model,category=info['category'],k=info['k'],**features(r,p[r['id']]))
            df=pd.DataFrame(records)
            numeric=['cer','wer','char_errors','word_errors','chars','words','k']
            df[numeric]=df[numeric].apply(pd.to_numeric)
            main=df[df.k.isna() | (df.k==8)]
            assert len(main)==150
            def emit(kind,key,g):
                summary.append(dict(language=lang,model=model,scope=kind,group=key,**summarize(g)))
            emit('overall','all',main)
            for cat,g in main.groupby('category'): emit('category',cat,g)
            for k,g in df.dropna(subset='k').groupby('k'): emit('repetition',str(int(k)),g)
            counts=Counter(x for id_ in main.id for x in p[id_].split() if x!='|')
            frequencies.extend(dict(language=lang,model=model,phoneme=k,count=v,share=v/sum(counts.values())) for k,v in counts.most_common())
            for feature in FEATURES:
                selection=df.dropna(subset=['k']) if feature=='k' else main
                g=selection[selection.status=='ok']
                for metric in ['cer','wer']:
                    rho=spearmanr(g[feature],g[metric]).statistic if g[feature].nunique()>1 and g[metric].nunique()>1 else None
                    correlations.append(dict(language=lang,model=model,scope='all_repetitions' if feature=='k' else 'main_150',feature=feature,metric=metric,n=len(g),spearman=rho))
            observations.append(df)
    allrows=pd.concat(observations)
    index=['language','model','id','category','status']
    allrows[index+list(FEATURES)+['consonants','cer','wer']].to_csv(figdir/'example_features.csv',index=False)
    agg=pd.DataFrame(summary)
    agg.to_csv(ROOT/'summary.csv',index=False)
    cor=pd.DataFrame(correlations)
    cor.to_csv(figdir/'correlations.csv',index=False)
    pd.DataFrame(frequencies).to_csv(figdir/'phoneme_frequencies.csv',index=False)
    for lang,models in MODELS.items():
        sub=agg[agg.language==lang]
        fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
        for ax,metric in zip(axes,['micro_cer','micro_wer']):
            for m in models:
                d=sub[(sub.model==m)&(sub.scope=='repetition')]
                ax.plot(d.group.astype(int),d[metric]*100,'o-',label=m)
            ax.set(xscale='log',xticks=[1,2,4,8,16,32,64],xlabel='Повторов', ylabel=metric.split('_')[1].upper()+', %')
            ax.set_xticklabels([1,2,4,8,16,32,64]);ax.grid(alpha=.25);ax.legend()
        fig.suptitle(LANG[lang]);fig.savefig(figdir/(lang+'_repetitions.png'),dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,metric in zip(axes,['micro_cer','micro_wer']):
            d=sub[sub.scope=='category'].pivot(index='group',columns='model',values=metric)[models]*100
            values=d.to_numpy()
            img=ax.imshow(values,cmap='Blues',aspect='auto',vmin=0,vmax=max(100,np.nanmax(values)))
            for (y,x),v in np.ndenumerate(values):
                ax.text(x,y,f'{v:.0f}',ha='center',va='center',color='white' if img.norm(v)>.55 else 'black')
            ax.set(xticks=range(len(models)),xticklabels=models,yticks=range(6),yticklabels=CATS,title=metric.split('_')[1].upper()+', %')
        fig.suptitle(LANG[lang]);fig.savefig(figdir/(lang+'_categories.png'),dpi=150);plt.close(fig)
        # Same texts for all systems; plot distributions once per language.
        d=allrows[(allrows.language==lang)&(allrows.model==models[0])]
        d=d[d.k.isna()|(d.k==8)]
        fig,ax=plt.subplots(2,4,figsize=(12,5),layout='constrained')
        for a,f in zip(ax.flat,[f for f in FEATURES if f!='k']+['consonants']):
            a.hist(d[f],bins=12,color='#326583');a.set_title(FEATURES.get(f,'Доля согласных'),fontsize=10);a.set_ylabel('Примеров')
        fig.suptitle(LANG[lang]+' • 150 текстов');fig.savefig(figdir/(lang+'_features.png'),dpi=150);plt.close(fig)
if __name__=='__main__':run()
