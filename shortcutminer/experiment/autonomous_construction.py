"""Training-only autonomous construction with actual paired visual review.

Does not open evaluation images, infer task-model predictions or alter scientific gates.
The returned frozen recipe must still be authorized by execution resources.
"""
import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Literal
import numpy as np
from PIL import Image,ImageDraw
from pydantic import Field,model_validator
from shortcutminer.protocol.identify import StrictProtocolModel,CandidateId,NonEmptyText
from shortcutminer.protocol.model_runtime import ModelImageInput
from shortcutminer.intervention.protocol import AutonomousImageTransform,GlobalColorTransform,LuminanceTransform
from shortcutminer.intervention.engine import InterventionEngine
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.experiment.luminance_calibration import LuminanceCalibrationChoice,calibrate_luminance,training_images
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.experiment.shortcut_knowledge import load_shortcut_knowledge

class ConstructionProposal(StrictProtocolModel):
    status:Literal['selected','deferred']
    candidate_id:CandidateId
    descriptor_key:NonEmptyText
    transformation:AutonomousImageTransform|GlobalColorTransform|None=None
    luminance_calibration:LuminanceCalibrationChoice|None=None
    rationale:NonEmptyText
    intended_invariants:list[NonEmptyText]=Field(min_length=1)
    limitations:list[NonEmptyText]=Field(min_length=1)

    @model_validator(mode='after')
    def one_recipe(self):
        n=int(self.transformation is not None)+int(self.luminance_calibration is not None)
        if n!=(1 if self.status=='selected' else 0):raise ValueError('Selected: exactly one recipe; deferred: no recipe')
        if isinstance(self.transformation,LuminanceTransform) and self.transformation.mode!='offset':
            raise ValueError('Use training calibration for normalization or reflection, not guessed target/pivot values')
        return self

class VisualConstructionReview(StrictProtocolModel):
    status:Literal['supported','unverified','failed']
    reviewed_pair_ids:list[NonEmptyText]=Field(min_length=1)
    cue_change_observation:NonEmptyText
    task_content_observation:NonEmptyText
    artifacts_observation:NonEmptyText
    limitations:list[NonEmptyText]=Field(min_length=1)

class ConstructionDecision(StrictProtocolModel):
    decision:Literal['accept','revise','defer']
    rationale:NonEmptyText
    limitations:list[NonEmptyText]=Field(min_length=1)

def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def _invoke_typed(client,agent,payload,schema,prompt):
    validated=getattr(client,'invoke_agent_validated',None)
    if callable(validated):
        raw=validated(agent,payload,system_prompt=prompt,validator=lambda value:schema.model_validate(value))
    else:
        raw=client.invoke_agent(agent,payload,system_prompt=prompt)
    return schema.model_validate(raw)

def _freeze(path,value):
    value['receipt_hash']=_hash(value)
    _write(path,value)
    return value

def preview_construction(operation,resources,root,directory,*,limit=12):
    """Persist actual training pairs and report changes without semantic pass claims."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    records=[];pictures=[]
    for index,(ref,path) in enumerate(training_images(resources,root,limit=limit)):
        before=load_rgb_image_file(path).rgb
        suffix='.jpg' if operation.type=='compression' else '.png'
        changed_path=directory/(f'pair-{index:03d}-changed'+suffix)
        quality=InterventionEngine().apply(source_path=path,output_path=changed_path,transformation=operation,dataset_root=Path(root))
        after=load_rgb_image_file(changed_path).rgb
        anchor_path=directory/f'pair-{index:03d}-original.png';Image.fromarray(before).save(anchor_path)
        a=np.asarray(Image.fromarray(before).convert('YCbCr')).astype(float)
        b=np.asarray(Image.fromarray(after).convert('YCbCr')).astype(float)
        records.append({'pair_id':f'pair-{index:03d}','source_reference':ref,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'original_reference':anchor_path.name,'changed_reference':changed_path.name,
          'original_sha256':hashlib.sha256(anchor_path.read_bytes()).hexdigest(),'changed_sha256':hashlib.sha256(changed_path.read_bytes()).hexdigest(),
          'same_dimensions':quality.same_dimensions,'changed_pixel_fraction':quality.pixel_change_fraction,
          'original_mean_y':float(a[:,:,0].mean()),'changed_mean_y':float(b[:,:,0].mean()),
          'chroma_mean_absolute_difference':float(np.abs(a[:,:,1:]-b[:,:,1:]).mean()),
          'changed_y_endpoint_fraction':float(np.mean((b[:,:,0]==0)|(b[:,:,0]==255)))})
        centered_a=a[:,:,0]-a[:,:,0].mean();centered_b=b[:,:,0]-b[:,:,0].mean()
        records[-1].update(original_std_y=float(a[:,:,0].std()),changed_std_y=float(b[:,:,0].std()),
            centered_y_mean_absolute_difference=float(np.abs(centered_a-centered_b).mean()),
            spatial_y_correlation=float(np.corrcoef(centered_a.ravel(),centered_b.ravel())[0,1])
                if centered_a.std()>0 and centered_b.std()>0 else None)
        pictures.append((Image.fromarray(before),Image.fromarray(after)))
    if not records or not all(r['same_dimensions'] for r in records):raise ValueError('No valid training pairs')
    if not any(r['changed_pixel_fraction']>0 for r in records):raise ValueError('Proposed transform is a no-op on training previews')
    sheet=Image.new('RGB',(384,160*len(pictures)),(32,32,32));draw=ImageDraw.Draw(sheet)
    for i,(before,after) in enumerate(pictures):
        for j,picture in enumerate((before,after)):
            picture.thumbnail((184,136));sheet.paste(picture,(j*192,i*160+24))
            draw.text((j*192+4,i*160+4),records[i]['pair_id']+(' original' if j==0 else ' changed'),fill='white')
    sheet_path=directory/'paired-contact-sheet.png';sheet.save(sheet_path)
    _write(directory/'pairs.json',{'scope':'training_only','pairs':records,'semantic_preservation':'unverified'})
    return records,sheet_path

def select_autonomous_construction(client,vision_client,*,run_id,candidates,observations,descriptor_keys,
        resources,root,receipt_path,task_description,training_diagnostics=None,max_revisions=2):
    """Quantitative selects; Qualitative sees paired pixels; Executive authorizes."""
    if max_revisions not in {0,1,2}:raise ValueError('Construction revisions bounded to 0..2')
    receipt_path=Path(receipt_path)
    # Resume validates data, proposal inputs and all saved preview pixels before reusing authority.
    source_hashes={ref:hashlib.sha256(path.read_bytes()).hexdigest() for ref,path in training_images(resources,root)}
    metadata_path=(Path(root)/resources.model_metadata_file).resolve(strict=True)
    metadata_path.relative_to(Path(root).resolve())
    if not candidates:raise ValueError('Construction requires frozen discovery candidates')
    base={'run_id':run_id,'mode':'autonomous_visual_intervention','task':task_description,'frozen_candidates':candidates,
       'training_qualitative_observations':observations,'available_descriptor_keys':descriptor_keys,
       'training_diagnostics':training_diagnostics or {},'method_memory':load_shortcut_knowledge(),
       'response_schema':ConstructionProposal.model_json_schema(),
       'construction_policy':'Agent selects mechanism, mode and calibration strategy; Python fits numeric parameters from pooled training pixels only. No held-out data, labels in calibration, truth or model outcomes.',
       'recipe_encoding':{'calibrated_luminance':{'transformation':None,'luminance_calibration':{'mode':'normalize_mean or reflect_mean','strategy':'pooled_median or pooled_mean or mid_quantiles'}},'other_supported_transform':{'transformation':'Complete supported transformation object','luminance_calibration':None},'rule':'Choose exactly one branch. Normalization/reflection MUST use luminance_calibration, never an incomplete transformation object. Mode and strategy remain Agent-selected.'},
       'diagnostic_design_guidance':{'neutralization':'Remove the measured cue by mapping to a pooled center; tests removal, not reversal.',
           'reversal':'Reflect the measured cue around a training-fitted center; tests opposite cue values while retaining spatial variation. Check gamut, content and training support.',
           'selection_rule':'For a photometry candidate compare neutralization and reversal explicitly in rationale, using training evidence only. Select one defensible primary diagnostic before held-out access; do not select based on desired accuracy loss. Both may preserve correlated texture/color cues. Other candidate families may use other supported operations.',
           'reporting':'Prediction-margin changes are auxiliary sensitivity diagnostics, never a substitute for the frozen accuracy-drop certification.'},
       'training_input_hash':_hash(source_hashes),'training_sample_count':len(source_hashes),
       'construction_metadata_sha256':hashlib.sha256(metadata_path.read_bytes()).hexdigest()}
    identity=_hash(base)
    if receipt_path.exists():
        saved=json.loads(receipt_path.read_text())
        if saved.get('receipt_hash')!=_hash({k:v for k,v in saved.items() if k!='receipt_hash'}):
            raise ValueError('Frozen construction receipt changed')
        if saved['input_hash']!=identity:raise ValueError('Frozen construction identity changed')
        for ref,h in saved.get('preview_file_hashes',{}).items():
            path=(receipt_path.parent/ref).resolve(strict=True);path.relative_to(receipt_path.parent.resolve())
            if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError('Frozen preview pixels changed')
        return saved
    feedback=[];remaining=list(candidates)
    for attempt in range(max_revisions+1):
        payload={**base,'frozen_candidates':remaining,'feedback':feedback}
        proposal=_invoke_typed(client,'quantitative',payload,ConstructionProposal,
          load_agent_workspace_prompt('quantitative')+' ConstructionProposal JSON only. Select a currently evidenced nuisance hypothesis and supported recipe. Numeric Y targets/pivots must be fitted by luminance_calibration. A lack of perfect semantic proof alone is not a reason to abandon a scoped diagnostic; disclose uncertainty. Do not guess task labels or certify content preservation.')
        if proposal.candidate_id not in {c['candidate_id'] for c in remaining}:raise ValueError('Proposal changed frozen candidate identity')
        if proposal.descriptor_key not in descriptor_keys:raise ValueError('Unavailable measured descriptor')
        _write(receipt_path.parent/f'construction-attempt-{attempt}.json',{'proposal':proposal.model_dump(mode='json'),'input_hash':identity,'feedback':feedback})
        if proposal.status=='deferred':
            remaining=[c for c in remaining if c['candidate_id']!=proposal.candidate_id]
            feedback.append({'candidate_deferral':proposal.model_dump(mode='json')})
            if remaining:continue
            break
        calibration=None;operation=proposal.transformation
        if proposal.luminance_calibration is not None:
            calibration=calibrate_luminance(proposal.luminance_calibration,resources,root)
            operation=LuminanceTransform.model_validate(calibration['transformation'])
        try:
            records,sheet=preview_construction(operation,resources,root,receipt_path.parent/f'preview-{attempt}')
        except ValueError as error:
            feedback.append({'construction_error':str(error),'prior_proposal':proposal.model_dump(mode='json')})
            continue
        visual_input={'run_id':run_id,'mode':'construction_visual_review','task':task_description,'candidate_id':proposal.candidate_id,
           'candidate':next(c for c in candidates if c['candidate_id']==proposal.candidate_id),
           'transformation':operation.model_dump(mode='json'),'training_pairs':records,
           'response_schema':VisualConstructionReview.model_json_schema(),
           'display':'Pairs in row order, original left and changed right. Labels/padding are display annotations, not image evidence.'}
        if isinstance(operation,LuminanceTransform):
            means=np.asarray([[r['original_mean_y'],r['changed_mean_y']] for r in records])
            visual_input['cue_execution_check']={'scope':'training_preview_only_no_model_outcomes',
                'mode':operation.mode,'original_y_mean_range':np.ptp(means[:,0]).item(),
                'changed_y_mean_range':np.ptp(means[:,1]).item(),
                'mean_y_correlation':float(np.corrcoef(means.T)[0,1]) if np.all(means.std(axis=0)>0) else None,
                'note':'Normalization should collapse mean differences; reflection should reverse their ordering. Neither removes texture/contrast variation. These are diagnostics, not semantic certification or new scientific thresholds.'}
        data=sheet.read_bytes()
        with Image.open(sheet) as decoded:width,height=decoded.size
        image_input=ModelImageInput(sample_id='construction-training-pairs',sample_reference=sheet.name,
            image_sha256=hashlib.sha256(data).hexdigest(),mime_type='image/png',width=width,height=height)
        visual=VisualConstructionReview.model_validate(vision_client.invoke_multimodal(run_id=run_id,agent_name='qualitative',
            payload_for_hash=visual_input,system_prompt=load_agent_workspace_prompt('qualitative')+' In construction_visual_review mode return only VisualConstructionReview JSON. Inspect actual original/changed pairs. Report whether the candidate cue was changed and task content remains visible, including clipping, borders and missing structure. A global pixel change fraction is not evidence that semantics were destroyed. Visual judgment is not ground truth; use unverified when uncertain.',
            content_parts=[{'type':'text','text':json.dumps(visual_input)},
              {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode(),'detail':'high'}}],
            validator=lambda v:VisualConstructionReview.model_validate(v),image_inputs=[image_input]))
        if len(set(visual.reviewed_pair_ids))!=len(visual.reviewed_pair_ids) or not set(visual.reviewed_pair_ids).issubset({r['pair_id'] for r in records}):
            raise ValueError('Visual review cites unavailable pairs')
        decision_input={'run_id':run_id,'mode':'construction_semantic_review','task':task_description,'proposal':proposal.model_dump(mode='json'),
          'fitted_transformation':operation.model_dump(mode='json'),'visual_review':visual.model_dump(mode='json'),
          'pixel_diagnostics':records,'response_schema':ConstructionDecision.model_json_schema(),
          'scope':'Training-only diagnostic authorization; no task-model predictions or held-out outcomes'}
        decision_input['cue_execution_check']=visual_input.get('cue_execution_check')
        decision=_invoke_typed(client,'executive',decision_input,ConstructionDecision,
           load_agent_workspace_prompt('executive')+' ConstructionDecision JSON only. Use the actual Qualitative paired-image review and pixel diagnostics. Global changes are expected for a global cue; never reject solely because most pixels changed. Consider lost class evidence, clipping, cue specificity and scope. An unverified semantic status can authorize a limited dependence diagnostic with retained limitations; it cannot authorize a causal claim. A failed visual review must not be accepted.')
        _write(receipt_path.parent/f'construction-review-{attempt}.json',{'visual':visual.model_dump(mode='json'),'executive':decision.model_dump(mode='json'),'input':decision_input})
        if decision.decision=='accept' and visual.status!='failed':
            saved={'status':'selected','input_hash':identity,'proposal':proposal.model_dump(mode='json'),
                'transformation':operation.model_dump(mode='json'),'calibration':calibration,'visual_review':visual.model_dump(mode='json'),
                'executive_review':decision.model_dump(mode='json'),'truth_seen':False,'heldout_seen':False,
                'cue_execution_check':visual_input.get('cue_execution_check'),
                'scope':'training_only_before_evaluation','semantic_preservation':'unverified',
                'training_source_hashes':source_hashes,
                'construction_metadata_sha256':base['construction_metadata_sha256'],
                'preview_file_hashes':{p.relative_to(receipt_path.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sheet.parent.iterdir() if p.is_file()}}
            return _freeze(receipt_path,saved)
        feedback.append({'visual_review':visual.model_dump(mode='json'),'executive_review':decision.model_dump(mode='json'),
                         'prior_proposal':proposal.model_dump(mode='json')})
        if decision.decision=='defer':
            remaining=[c for c in remaining if c['candidate_id']!=proposal.candidate_id]
            if not remaining:break
    saved={'status':'deferred','input_hash':identity,'training_source_hashes':source_hashes,'feedback':feedback,
        'truth_seen':False,'heldout_seen':False,'reason':'No construction accepted within the bounded training-only planning budget'}
    return _freeze(receipt_path,saved)
