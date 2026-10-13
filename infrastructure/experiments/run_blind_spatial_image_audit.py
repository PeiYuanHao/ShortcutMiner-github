"""Fresh Agent discovery with training localization, geometry and real paired review.

Use the generic entry's CLI. No prior candidate report or dataset truth is loaded.
The caller chooses spatial-quality-v3.1 capability; the Agent chooses candidate,
target coordinates, operation and parameters. Reused data remain development.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import run_autonomous_image_audit as entry
from shortcutminer.experiment.autonomous_construction import select_autonomous_construction,_freeze
from shortcutminer.experiment.spatial_construction_quality import select_spatial_construction
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.inspection.image import load_rgb_image_file


class FrozenConstructionReplay:
    """Replay this run's newly Agent-selected plan only, never historic discovery."""
    def __init__(self,client,proposal):self.client=client;self.proposal=proposal
    def invoke_agent(self,name,payload,**kwargs):
        if name=='quantitative':return dict(self.proposal)
        return self.client.invoke_agent(name,payload,**kwargs)
    def invoke_agent_validated(self,name,payload,*,validator,**kwargs):
        if name=='quantitative':
            result=dict(self.proposal);validator(result);return result
        return self.client.invoke_agent_validated(name,payload,validator=validator,**kwargs)


def select_with_spatial_quality(client,vision_client,**kwargs):
    resources=kwargs['resources'];root=Path(kwargs['root']);path=Path(kwargs['receipt_path'])
    frame=pd.read_csv(root/resources.model_metadata_file,dtype=str,keep_default_na=False)
    frame=frame[frame[resources.split_column].isin(resources.train_split_values)]
    sums={};counts={}
    for row in frame.to_dict('records'):
        source=(root/row[resources.sample_path_column]).resolve(strict=True);source.relative_to(root.resolve())
        rgb=load_rgb_image_file(source).rgb.astype(float)/255
        gray=rgb.mean(axis=2);edge=np.zeros(gray.shape)
        edge[:,1:]+=np.abs(np.diff(gray,axis=1));edge[1:,:]+=np.abs(np.diff(gray,axis=0))
        grid=np.asarray(Image.fromarray(edge.astype('float32')).resize((16,16),Image.Resampling.BOX))
        label=row['y'];sums[label]=sums.get(label,np.zeros((16,16)))+grid;counts[label]=counts.get(label,0)+1
    means=[sums[label]/counts[label] for label in sorted(sums)]
    difference=np.max(means,axis=0)-np.min(means,axis=0)
    quality_path=path.parent/'spatial-quality.json'
    selected=select_spatial_construction(client,candidates=kwargs['candidates'],observations=kwargs['observations'],
        descriptor_keys=kwargs['descriptor_keys'],resources=resources,root=root,receipt_path=quality_path,
        prompt=load_agent_workspace_prompt('quantitative'),check_edge_residual=True,allow_exterior=True,training_diagnostics={
            'absolute_class_edge_difference_16x16':np.round(difference,4).tolist(),
            'grid_definition':'16x16 training image grid; cell column c,row r covers normalized x=[c/16,(c+1)/16],y=[r/16,(r+1)/16].',
            'scope':'Training pixels only; no task-model outcomes or private truth.'})
    proposal={k:v for k,v in selected.model_dump(mode='json').items()
              if k in {'status','candidate_id','descriptor_key','transformation','rationale','intended_invariants','limitations'}}
    proposal['limitations'] += ['Agent-declared region coverage does not certify localization or task-label preservation.',
        *selected.semantic_risks,*selected.processing_controls]
    # Do not repair a frozen geometric plan using visual/model outcomes.
    kwargs['max_revisions']=0
    receipt=select_autonomous_construction(FrozenConstructionReplay(client,proposal),vision_client,**kwargs)
    quality_sha=entry.sha(quality_path)
    if 'spatial_quality_sha256' in receipt:
        if receipt['spatial_quality_sha256']!=quality_sha:raise ValueError('Frozen spatial quality changed')
        return receipt
    receipt.pop('receipt_hash')
    receipt.update(construction_protocol='spatial-quality-v3.3-with-actual-visual-review',target_geometry=selected.target_geometry,
        target_box=selected.target_box.model_dump(mode='json'),spatial_quality_sha256=quality_sha,
        intervention_owner='quantitative_agent',plan_replay_scope='This run training-only Agent plan, not prior-run output')
    return _freeze(path,receipt)


if __name__=='__main__':
    entry.select_autonomous_construction=select_with_spatial_quality
    entry.main()
