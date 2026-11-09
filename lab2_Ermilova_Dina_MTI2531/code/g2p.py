import os
import espeakng_loader
os.environ['ESPEAK_DATA_PATH']=espeakng_loader.get_data_path()
from phonemizer.backend.espeak.wrapper import EspeakWrapper
EspeakWrapper.set_library(espeakng_loader.get_library_path())
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from prepare import MODELS,ROOT,load,save
import re,unicodedata
for lang,models in MODELS.items():
    backend=EspeakBackend('id' if lang=='indonesian' else 'be',with_stress=False,language_switch='remove-flags')
    rows=load(ROOT/lang/models[0]/'texts.csv')
    texts=[re.sub(r'[^\w\s\'’]',' ',r['text'].lower()) for r in rows]
    seqs=backend.phonemize(texts,separator=Separator(phone=' ',word=' | ',syllable=''),strip=True)
    assert len(seqs)==len(rows)
    result=[]
    for row,seq in zip(rows,seqs):
        tokens=[p for p in seq.split() if p!='|']
        assert tokens
        result.append(dict(id=row['id'],phonemes=seq,phoneme_count=len(tokens)))
    for m in models: save(ROOT/lang/m/'phonemes.csv',result)
    print(lang,len(result),result[0])
