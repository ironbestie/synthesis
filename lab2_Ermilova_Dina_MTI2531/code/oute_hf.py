"""Run OuteTTS with its 64-token repetition penalty evaluated in one GPU operation."""
import os
os.environ['USE_TF']='0'
import torch
import outetts
import transformers.generation.utils as generation
from outetts.models.hf_model import RepetitionPenaltyLogitsProcessorPatch

class WindowPenalty(RepetitionPenaltyLogitsProcessorPatch):
    def __call__(self,input_ids,scores):
        if self.penalty==1.0: return scores
        ids=input_ids[:,-self.penalty_last_n:]
        # Every input ID comes from the model vocabulary. Duplicates see the same original score.
        values=scores.gather(1,ids)
        adjusted=torch.where(values<=0,values*self.penalty,values/self.penalty)
        return scores.scatter_(1,ids,adjusted)

def main():
    generation.RepetitionPenaltyLogitsProcessor=WindowPenalty
    from synthesis import run
    run('belarusian','outetts',300)
if __name__=='__main__':main()
