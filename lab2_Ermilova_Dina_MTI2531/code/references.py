import csv, io, shutil, subprocess, tarfile
from pathlib import Path
from prepare import ROOT, save
def reference(lang, key, models):
    root='https://huggingface.co/datasets/google/fleurs/resolve/main/data/'+key
    raw=subprocess.check_output(['curl','-fsSL',root+'/dev.tsv'])
    metadata={r[1]:r for r in csv.reader(io.StringIO(raw.decode()),delimiter='\t') if len(r)>5}
    with subprocess.Popen(['curl','-fsSL',root+'/audio/dev.tar.gz'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL) as proc:
        try:
            with tarfile.open(fileobj=proc.stdout,mode='r|gz') as tar:
                for member in tar:
                    name=Path(member.name).name
                    rec=metadata.get(name)
                    if rec and 48000<=int(rec[5])<=160000:
                        blob=tar.extractfile(member).read()
                        for model in models:
                            folder=ROOT/lang/model/'references';folder.mkdir(parents=True,exist_ok=True)
                            (folder/name).write_bytes(blob)
                            save(folder/'references.csv',[dict(id='fleurs_'+key,filename=name,transcript=rec[2],source=root+'/audio/dev.tar.gz',metadata=root+'/dev.tsv',license='CC BY 4.0')])
                        print(lang,name,rec[2],flush=True);return
        finally: proc.terminate()
    raise RuntimeError('No eligible recording')
if __name__=='__main__':
    reference('belarusian','be_by',['omnivoice','outetts'])
    reference('indonesian','id_id',['omnivoice'])
