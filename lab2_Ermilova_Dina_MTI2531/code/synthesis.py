import argparse, json, os, random, re, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from prepare import ROOT, MODELS, load, save

SEED=28
def stress(text):
    words=dict((w.replace('+',''),w) for w in (ROOT/'code/stress.txt').read_text().split())
    def mark(m):
        word=m[0].lower()
        if word in words: return words[word]
        if len(re.findall('[аэоуыяеёюі]',word))>1: raise ValueError('Нет ударения для '+word)
        return word
    return re.sub("[а-яёіў']+", mark, text.lower())

def engine(lang, name, folder):
    if name=='mms':
        from transformers import VitsModel,AutoTokenizer
        path=ROOT/'models/mms'
        model=VitsModel.from_pretrained(path).eval()
        tok=AutoTokenizer.from_pretrained(path)
        return lambda text:(model(**tok(text,return_tensors='pt')).waveform[0].cpu().numpy(),model.config.sampling_rate)
    if name=='supertonic':
        from supertonic import TTS
        tts=TTS(model='supertonic-3',model_dir=ROOT/'models/supertonic',auto_download=False,intra_op_num_threads=3)
        style=tts.get_voice_style('F1')
        return lambda text:(tts.model([text],style,5,1.0,'id')[0][0],tts.sample_rate)
    if name=='silero':
        importer=torch.package.PackageImporter(str(ROOT/'models/silero.pt'))
        model=importer.load_pickle('tts_models','model')
        return lambda text:(model.apply_tts(text=text,speaker='bel_larisa',sample_rate=24000).cpu().numpy(),24000)
    ref=load(folder/'references/references.csv')[0]
    path=str(folder/'references'/ref['filename'])
    if name=='omnivoice':
        from omnivoice import OmniVoice
        device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        weights = ROOT/'models/omni'
        model=OmniVoice.from_pretrained(str(weights),device_map=device,dtype=torch.float32 if device=='cpu' else torch.float16)
        prompt=model.create_voice_clone_prompt(ref_audio=path,ref_text=ref['transcript'])
        return lambda text:(model.generate(text=text,language='Indonesian' if lang=='indonesian' else 'Belarusian',voice_clone_prompt=prompt,num_step=16,speed=1.0,audio_chunk_threshold=100000)[0],model.sampling_rate)
    import outetts
    gguf = next((ROOT/'models/outegg').glob('*.gguf'),None)
    if gguf:
        config=outetts.ModelConfig(model_path=str(gguf), tokenizer_path=str(ROOT/'models/outetok'),
            backend=outetts.Backend.LLAMACPP, n_gpu_layers=99, device='cpu',
            audio_codec_path=str(ROOT/'models/dac/weights_24khz_1.5kbps_v1.0.pth'),
            additional_model_config={'seed':SEED,'verbose':False})
    else:
        config=outetts.ModelConfig.auto_config(model=outetts.Models.VERSION_1_0_SIZE_1B,backend=outetts.Backend.HF)
        from huggingface_hub import snapshot_download, hf_hub_download
        from download_models import PINS
        checkpoint=snapshot_download('OuteAI/Llama-OuteTTS-1.0-1B',revision=PINS['outetok'],allow_patterns=['*.json','*.safetensors','*.model','*.txt'])
        config.model_path=checkpoint
        config.tokenizer_path=checkpoint
        config.audio_codec_path=hf_hub_download('ibm-research/DAC.speech.v1.0','weights_24khz_1.5kbps_v1.0.pth',revision=PINS['dac'])
    if not gguf and torch.cuda.is_available() and torch.cuda.get_device_capability()[0]<8:
        config.dtype=torch.float16
    model=outetts.Interface(config=config)
    profile=folder/'references/speaker.json'
    if profile.exists(): voice=model.load_speaker(str(profile))
    else:
        alignment=json.loads((folder/'references/alignment.json').read_text())
        voice=model.audio_processor.create_speaker_from_dict({'audio':{'bytes':Path(path).read_bytes()},**alignment})
        model.save_speaker(voice,str(profile))
    def synth(text):
        if gguf: model.model.model.set_seed(SEED)
        result=model.generate(config=outetts.GenerationConfig(text=text,generation_type=outetts.GenerationType.REGULAR,speaker=voice,sampler_config=outetts.SamplerConfig(temperature=0.4)))
        return result.audio.squeeze().cpu().numpy(),result.sr
    return synth

def run(lang,name,limit):
    folder=ROOT/lang/name
    (folder/'audio').mkdir(exist_ok=True)
    ledger=folder/'synthesis.csv'
    done=load(ledger) if ledger.exists() else []
    seen={r['id'] for r in done}
    if len(seen)==len(load(folder/'texts.csv')):return
    torch.set_num_threads(3)
    generate=engine(lang,name,folder)
    count=0
    for r in load(folder/'texts.csv'):
        if r['id'] in seen: continue
        t=time.monotonic()
        out=dict(id=r['id'],tts_input=r['text'],audio='',duration='',elapsed='',status='tts_error',reason='',reference_id='')
        if name in ['omnivoice','outetts']: out['reference_id']=load(folder/'references/references.csv')[0]['id']
        try:
            if name=='silero': out['tts_input']=stress(r['text'])
            random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
            with torch.inference_mode(): wave,rate=generate(out['tts_input'])
            wave=np.asarray(wave).squeeze()
            if wave.ndim!=1 or not wave.size or not np.isfinite(wave).all(): raise ValueError('Invalid waveform')
            p=folder/'audio'/(r['id']+'.wav')
            sf.write(p.with_suffix('.tmp'),wave,rate,format='WAV',subtype='PCM_16')
            p.with_suffix('.tmp').replace(p)
            out.update(audio=p.relative_to(folder).as_posix(),duration=round(len(wave)/rate,4),status='tts_ok')
        except Exception as e: out['reason']=type(e).__name__+': '+str(e)
        out['elapsed']=round(time.monotonic()-t,3)
        done.append(out); save(ledger,done);count+=1
        if count==1 or count%25==0: print(name,len(done),out['status'],out['reason'],flush=True)
        if count==limit: break
        if len(done)>=3 and all(x['status']=='tts_error' for x in done[-3:]): raise RuntimeError('Three consecutive synthesis errors. Inspect synthesis.csv')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('language',choices=MODELS);p.add_argument('model');p.add_argument('--limit',type=int,default=300)
    a=p.parse_args();assert a.model in MODELS[a.language];run(a.language,a.model,a.limit)
