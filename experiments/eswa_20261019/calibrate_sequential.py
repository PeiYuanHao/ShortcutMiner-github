"""Offline Monte Carlo calibration of the actual frozen paired betting process.

These are simulated paired scores under constant-mean independent groups, not
dataset discoveries, and do not validate semantic preservation or audit FPR.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.paired_confidence_sequence import STAKES,WEIGHTS,log_e_value


def batch_log_e(mu,counts):
    values=np.log(1+STAKES[:,None]*(np.array([-1.,0.,1.])[None,:]-mu))
    return logsumexp(np.log(WEIGHTS)+counts@values.T,axis=-1)


def frequency(mask):
    n=len(mask);k=int(np.sum(mask));p=k/n
    return {'count':k,'replicates':n,'rate':p,
        'monte_carlo_se':float(np.sqrt(p*(1-p)/n)),
        'exact_binomial_95_interval':[
            float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replicates',type=int,default=10000)
    args=parser.parse_args();rng=np.random.default_rng(2026101904)
    looks=[128,256,512,1024,1500];alpha=.05/3;boundary=np.log(1/alpha)
    # Verify the vectorized calculation against the unmodified implementation.
    checks=rng.multinomial(1500,[.1,.8,.1],size=20)
    for mu in (0.,.02,.05):
        assert np.allclose(batch_log_e(mu,checks),[log_e_value(mu,c) for c in checks],rtol=1e-12,atol=1e-12)
    settings=[]
    for discordance in (.1,.3,.6):
        for mu in (0.,.02,.05,.10):
            probs=[(discordance-mu)/2,1-discordance,(discordance+mu)/2]
            counts=np.zeros((args.replicates,3,3),dtype=np.int64)
            missed=np.zeros((args.replicates,3),dtype=bool)
            decisions=np.zeros((args.replicates,3),dtype=np.int8)
            sample_counts=np.full((args.replicates,3),1500,dtype=np.int32)
            previous=0
            for look in looks:
                counts+=rng.multinomial(look-previous,probs,size=(args.replicates,3));previous=look
                missed|=batch_log_e(mu,counts)>=boundary
                effect=(counts[:,:,2]-counts[:,:,0])/look
                rejected=batch_log_e(.02,counts)>=boundary
                active=decisions==0
                expand=active&rejected&(effect>.02)
                stop=active&rejected&(effect<.02)
                sample_counts[expand|stop]=look
                decisions[expand]=1;decisions[stop]=-1
            settings.append({'true_mean':mu,'discordance_probability':discordance,
                'probabilities_gained_unchanged_lost':probs,
                'family_any_true_mean_exclusion':frequency(missed.any(axis=1)),
                'candidate_expand':frequency(decisions[:,0]==1),
                'candidate_stop_below_threshold':frequency(decisions[:,0]==-1),
                'candidate_defer_at_budget':frequency(decisions[:,0]==0),
                'mean_first_candidate_pairs':float(sample_counts[:,0].mean()),
                'family_any_false_expansion':frequency((decisions==1).any(axis=1)) if mu<=.02 else None})
    record={'created_utc':datetime.now(timezone.utc).isoformat(),
        'simulation_seed':2026101904,'replicates_per_setting':args.replicates,
        'looks':looks,'family_alpha':.05,'candidate_alpha':alpha,'candidates':3,
        'minimum_meaningful_effect':.02,'actual_implementation_parity':True,
        'assumptions':'Independent iid paired score streams with a constant conditional mean; independently simulated candidates.',
        'scope':'Checks mathematical interval and allocation behavior under the simulation assumptions. Does not estimate actual shortcut false-confirmation or validate interventions.',
        'stakes':STAKES.tolist(),'weights':WEIGHTS.tolist(),'settings':settings}
    write(args.output,record)
    print(json.dumps({'status':'COMPLETED','settings':len(settings),'replicates_per_setting':args.replicates,
        'max_family_true_mean_exclusion_rate':max(s['family_any_true_mean_exclusion']['rate'] for s in settings),
        'output':str(args.output.resolve())}),flush=True)


if __name__=='__main__':main()
