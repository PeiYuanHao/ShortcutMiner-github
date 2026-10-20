"""Regenerate descriptive final figures from retained exports; no model calls."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--delivery',type=Path,required=True);args=parser.parse_args()
    root=args.delivery;figs=root/'figures';figs.mkdir(exist_ok=True)
    summary=json.loads((root/'SUMMARY.json').read_text());oracle=summary['synthetic_specificity']
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(figs/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.4));groups=summary['main'];x=np.arange(3);names=['Full','Single Agent','Fixed Pipeline'];colors=['#4477AA','#EE6677','#228833']
    axes[0].bar(x,[g['completed'] for g in groups],color=colors)
    for i,g in enumerate(groups):axes[0].text(i,g['completed']+.15,f"{g['completed']}/{g['scheduled']}",ha='center')
    axes[0].set(xticks=x,xticklabels=names,ylim=(0,8),ylabel='Completed audits',title='Execution completion')
    axes[1].bar(x,[g['known_tokens']/g['scheduled']/1e3 for g in groups],color=colors)
    axes[1].set(xticks=x,xticklabels=names,ylabel='Known tokens per audit\n(thousands; lower bound)',title='Usage lower bounds')
    fig.text(.5,.01,'Seven dependent conditions; one source pool. Completion is not discovery efficacy.\nUnknown usage is not zero; cost bars exclude missing provider usage.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,1));save(fig,'execution_and_known_usage')
    fig,ax=plt.subplots(figsize=(6.8,3.3));pet=summary['pet_local'];keys=['background_accuracy_drop','foreground_control_accuracy_drop']
    for i,key in enumerate(keys):
        value=pet['paired_effects'][key];e=value['estimate']*100;lo,hi=np.array(value['pointwise_95_interval'])*100
        ax.errorbar(e,i,xerr=[[e-lo],[hi-e]],fmt='o',capsize=4,color=['#4477AA','#EE6677'][i])
    ax.axvline(0,color='.5',lw=.8);ax.set(yticks=[0,1],yticklabels=['Background blur','Foreground area control'],xlabel='Accuracy drop (percentage points), paired 95% interval',title=f"Pet: evaluator segmentation diagnostics ({pet['n_valid']} pairs)")
    ax.set_ylim(1.5,-.5);fig.text(.5,.01,'Historical checkpoint and pool; annotation-assisted operations.\nInterpret background effects within the supplied interval and operation scope.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.14,1,1));save(fig,'pet_local_diagnostics')
    fig,ax=plt.subplots(figsize=(8,4.2));cases=oracle['cases']
    labels=[]
    for i,case in enumerate(cases):
        title={'no_artificial_cue':'Clean (no artificial cue)','position_patch':'Position patch','border':'Border','watermark':'Watermark'}[case['private_mechanism']]
        labels.append(title if case['rho'] is None else f"{title} / rho={case['rho']}")
        for offset,key,color,marker in [(-.12,'oracle_cue_removal_drop','#4477AA','o'),(.12,'source_content_control_drop','#EE6677','s')]:
            val=case['effects'][key];e=val['estimate']*100;lo,hi=np.array(val['pointwise_95_interval'])*100
            ax.errorbar(e,i+offset,xerr=[[e-lo],[hi-e]],fmt=marker,capsize=3,color=color,label=('Exact cue removal' if offset<0 else 'Non-cue content control') if i==0 else None)
    ax.axvline(0,color='.5',lw=.8);ax.set(yticks=np.arange(len(cases)),yticklabels=labels,xlabel='Accuracy drop (percentage points), paired pointwise 95% interval',title='Evaluator-only measurement validity diagnostics')
    ax.invert_yaxis();ax.legend(fontsize=9,loc='best');fig.text(.5,.01,'500 shared source IDs per available checkpoint; posthoc, dependent conditions.\nPrivate cue support supplied to evaluator. Does not replace autonomous discovery or certify it.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,1));save(fig,'evaluator_specificity')
    (figs/'CAPTIONS.md').write_text(f"""# Figure scope

Execution: {summary['agent_audits']} Agent audits in the supplied export. Completion is operational, not discovery efficacy. Known usage excludes missing provider telemetry.

Pet: {pet['n_valid']} valid pairs of {pet['n_selected']} selected, {pet['n_deferred']} deferred without replacement. Evaluator trimaps and paired bootstrap; historical checkpoint/pool. No-op fidelity does not certify semantics.

Synthetic: exact implanted-support restoration versus a matched intended area content control. Evaluator oracle, not autonomous construction. Use retained fidelity audits before attributing effects to nuisance cues. Shared source IDs are not independent datasets or untouched confirmation.
""",encoding='utf-8')

if __name__=='__main__':main()
