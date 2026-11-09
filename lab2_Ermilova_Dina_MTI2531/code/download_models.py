"""Download public checkpoint files, pin the Hugging Face revision and resume."""
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import sys
ROOT = Path(__file__).resolve().parents[1]
repos = {'outegg':'OuteAI/Llama-OuteTTS-1.0-1B-GGUF','outetok':'OuteAI/Llama-OuteTTS-1.0-1B','dac':'ibm-research/DAC.speech.v1.0','asr':'aadel4/omniASR-CTC-300M-v2', 'omni':'k2-fsa/OmniVoice', 'mms': 'facebook/mms-tts-ind', 'supertonic': 'Supertone/supertonic-3'}
PINS = {'outetok': 'f445d0d301c7fc9e47fd545eb9f9d2912afe7a77', 'outegg': 'f0a131bce9ca77922d38da5af18db11d29b6b768', 'mms': '50dc8c10320a328391efaab25e4c629f9bf254ec', 'asr': 'f2b3c39a62c061c27623c3f2d803294fbb954324', 'dac': '1ea7f64cd0678415e2d8c32d67b190722cb9b149', 'supertonic': '3cadd1ee6394adea1bd021217a0e650ede09a323', 'omni': 'c5fdb5ccb189668d56333f77ba2629f4cd7535f4'}
def get(name):
    repo = repos[name]
    folder = ROOT / 'models' / name
    folder.mkdir(exist_ok=True, parents=True)
    rev = PINS[name]
    stamp = folder/'revision.txt'
    if stamp.exists() and stamp.read_text().strip() != rev:
        raise ValueError(f"Checkpoint revision differs from the experiment: {folder}")
    info = json.loads(subprocess.check_output([
        'curl', '-fsSL', f'https://huggingface.co/api/models/{repo}/revision/{rev}'
    ]))
    stamp.write_text(rev)
    files = [s['rfilename'] for s in info['siblings']]
    if name in ('mms','asr','omni'):
        files = [f for f in files if f.endswith(('.json','.safetensors','.model','.txt','.yaml'))]
    elif name=='outegg':
        files = [f for f in files if f.endswith('-Q8_0.gguf')]
    elif name=='outetok':
        files = [f for f in files if f.endswith('.json')]
    elif name=='dac':
        files = ['weights_24khz_1.5kbps_v1.0.pth']
    else:
        files = [f for f in files if f.startswith(('onnx/','voice_styles/'))]
    if not files: raise ValueError(f"No weights found in {repo}")
    def fetch(f):
        out = folder/f
        if out.is_file(): return
        out.parent.mkdir(exist_ok=True, parents=True)
        tmp = str(out)+'.part'
        subprocess.run(['curl','-fLsS','--retry','3','-C','-','https://huggingface.co/'+repo+'/resolve/'+rev+'/'+f,'-o',tmp],check=True)
        Path(tmp).replace(out)
    with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(fetch,files))
if __name__=='__main__':
    for arg in sys.argv[1:]:
        if arg=='silero':
            import hashlib
            destination=ROOT/'models/silero.pt'
            destination.parent.mkdir(exist_ok=True, parents=True)
            if not destination.exists():
                tmp=destination.with_suffix('.part')
                subprocess.run(['curl','-fLsS','--retry','3','https://models.silero.ai/models/tts/ru/v5_cis_base_nostress.pt','-o',str(tmp)],check=True)
                tmp.replace(destination)
            assert hashlib.sha256(destination.read_bytes()).hexdigest()=='0405777e332906f0644e08a680f7cfdc2137ea864090079c1fdd30a43c1b8761'
        else: get(arg)
