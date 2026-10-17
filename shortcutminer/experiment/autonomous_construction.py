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
from shortcutminer.intervention.protocol import AutonomousImageTransform,LegacyAutonomousImageTransform,GlobalColorTransform,LuminanceTransform,ChannelRatioTransform
from shortcutminer.experiment.channel_ratio_calibration import ChannelRatioCalibrationChoice,calibrate_channel_ratio
from shortcutminer.intervention.engine import InterventionEngine
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.experiment.luminance_calibration import LuminanceCalibrationChoice,calibrate_luminance,training_images
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.experiment.shortcut_knowledge import load_shortcut_knowledge
from shortcutminer.experiment.construction_quality import POLICY as REVIEW_POLICY, BoundedRatioChoice

class ConstructionProposal(StrictProtocolModel):
    status:Literal['selected','deferred']
    candidate_id:CandidateId
    descriptor_key:NonEmptyText
    transformation:LegacyAutonomousImageTransform|GlobalColorTransform|None=None
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


class AlignedConstructionProposal(ConstructionProposal):
    transformation:AutonomousImageTransform|GlobalColorTransform|None=None
    channel_ratio_calibration:ChannelRatioCalibrationChoice|None=None

    @model_validator(mode='after')
    def one_recipe(self):
        n=sum(v is not None for v in (self.transformation,self.luminance_calibration,self.channel_ratio_calibration))
        if n!=(1 if self.status=='selected' else 0):raise ValueError('Selected: exactly one recipe; deferred: no recipe')
        if isinstance(self.transformation,ChannelRatioTransform):
            raise ValueError('Use channel_ratio_calibration, not guessed ratio centers')
        if isinstance(self.transformation,LuminanceTransform) and self.transformation.mode!='offset':
            raise ValueError('Use luminance_calibration, not guessed target/pivot values')
        return self


class ScalarChromaticityProposal(AlignedConstructionProposal):
    """Expose only executable color operations for a non-ratio primary target."""
    transformation:GlobalColorTransform|None=None
    luminance_calibration:None=None
    channel_ratio_calibration:None=None


class ReviewedConstructionProposal(AlignedConstructionProposal):
    channel_ratio_calibration:BoundedRatioChoice|None=None
    mechanism_scope:Literal['global_illumination','global_color_cast','background','foreground','spatial_artifact','texture','acquisition','other']
    mechanism_test:NonEmptyText
    remaining_confounds:list[NonEmptyText]=Field(min_length=1)
    training_selection_criteria:NonEmptyText


class ReviewedScalarProposal(ReviewedConstructionProposal):
    transformation:GlobalColorTransform|None=None
    luminance_calibration:None=None
    channel_ratio_calibration:None=None

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


class VisualConstructionReviewV2(VisualConstructionReview):
    mechanism_specificity:Literal['targeted','partial','unsupported']
    task_preservation:Literal['supported','uncertain','damaged']
    recommended_action:Literal['accept_scoped','revise','defer']
    preservation_risks:list[NonEmptyText]=Field(min_length=1)

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

def preview_construction(operation,resources,root,directory,*,limit=12,quality_policy=None,seek_counterexamples=True):
    """Persist actual training pairs and report changes without semantic pass claims."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    records=[];pictures=[]
    sources=training_images(resources,root,limit=limit)
    if not seek_counterexamples:
        import random
        pool=training_images(resources,root)
        sources=random.Random(2026).sample(pool,min(limit,len(pool)))
    if quality_policy is not None and seek_counterexamples:
        from shortcutminer.experiment.construction_quality import preview_sources
        sources=preview_sources(operation,resources,root)
    for index,(ref,path) in enumerate(sources):
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
        if quality_policy is not None:
            from shortcutminer.experiment.construction_quality import pixel_diagnostics
            records[-1].update(pixel_diagnostics(before,after))
            records[-1]['preview_selection']='seeded_random' if not seek_counterexamples or index<8 else 'training_target_extreme'
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
        resources,root,receipt_path,task_description,training_diagnostics=None,max_revisions=2,measurement_binding=None,quality_policy=None,
        disable_qualitative=False,seek_counterexamples=True):
    """Quantitative selects; Qualitative sees paired pixels; Executive authorizes."""
    if max_revisions not in {0,1,2}:raise ValueError('Construction revisions bounded to 0..2')
    if quality_policy not in {None,REVIEW_POLICY}:raise ValueError('Unknown construction quality policy')
    if quality_policy and measurement_binding is None:raise ValueError('Reviewed construction requires a frozen measurement')
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
    contract=None
    proposal_schema=ConstructionProposal
    if measurement_binding is not None:
        from shortcutminer.experiment.intervention_alignment import binding_contract
        contract=binding_contract(measurement_binding)
        proposal_schema=AlignedConstructionProposal
        if contract['measurement_axis']=='chromaticity' and 'channel_ratio' not in contract['eligible_operation_types']:
            proposal_schema=ScalarChromaticityProposal
        base['response_schema']=proposal_schema.model_json_schema()
        if 'channel_ratio' in (contract['eligible_operation_types'] or []):
            base['recipe_encoding']['calibrated_channel_ratio']={'transformation':None,'luminance_calibration':None,
                'channel_ratio_calibration':{'mode':'normalize_mean or reflect_log_ratio','strategy':'pooled_median or pooled_mean'},
                'scope':'Only a frozen ratio of RGB channel means. Channels come from the measurement; Python fits the center from positive pooled training means in log space. Never send identity channel scales as placeholders for calibration.'}
        elif contract['measurement_axis']=='chromaticity':
            base['primary_chromaticity_guidance']='The primary measurement is not an RGB mean ratio. channel_ratio_calibration is unavailable for this binding: leave it null. For saturation_mean, supported global_color desaturate/grayscale operations can directly change saturation. Select a supported mode and complete its parameters, subject to actual paired visual review; do not substitute a different measured axis.'
        base['recipe_encoding']['rule']='Choose exactly one recipe branch. Photometry normalization/reflection uses luminance_calibration; RGB channel-ratio normalization/reflection uses channel_ratio_calibration. Numeric centers are fitted by Python from training pixels only. Other operations require a complete supported transform.'
        base['measurement_intervention_contract']=contract
        base['primary_target_guidance']='Select an operation eligible for the frozen measured axis. Luminance normalization is not a primary chromaticity test. A global pixel change is insufficient: the frozen primary descriptor must change on training previews. Invalid proposals receive training-only revision feedback within the existing budget. Defer unsupported targets; do not change the frozen measurement or use predictions to choose.'
    if quality_policy:
        proposal_schema=ReviewedScalarProposal if proposal_schema is ScalarChromaticityProposal else ReviewedConstructionProposal
        base['response_schema']=proposal_schema.model_json_schema()
        base['construction_quality_policy']=quality_policy
        base['mechanism_scope_guidance']='Specify the mechanism tested, remaining confounds and training-only selection criteria. Global illumination, global color cast, natural subject color and background color are distinct. Whole-image transforms do not isolate background/foreground. A frozen scalar measurement tests only part of a broad hypothesis. Do not relabel subject coloration as a proven nuisance. Background-only hypotheses require an authorized segmentation-backed tool; none is silently substituted in this generic entry, so defer unsupported isolation.'
        base['training_preview_policy']='8 fixed-seed random training images + 4 primary-target extremes; no labels, held-out pixels or model outcomes select previews. Qualitative must examine all pairs for lost details, severe color casts and clipping. Revise damaged/unsupported constructions within the existing 3 attempts; do not select the strongest corruption.'
        if 'channel_ratio' in (contract['eligible_operation_types'] or []):
            base['recipe_encoding']['calibrated_channel_ratio']['channel_ratio_calibration']['max_log_shift']='Agent-selected positive bound <= log(4), justified from training appearance; no outcome tuning'
            base['recipe_encoding']['calibrated_channel_ratio']['scope']+=' In this version the executable operator is bounded_channel_ratio: per-pixel luminance-preserving chroma change with gamut compression, not the legacy reciprocal clipped RGB operator. Target may be partial; actual preview measurement required.'
    if disable_qualitative:
        base['ablation_qualitative_disabled']=True
        base['training_preview_policy']='Deterministic training-pixel checks only. Semantic paired-image review is disabled; preservation and specificity remain unverified.'
    if not seek_counterexamples:
        base['ablation_active_counterexample_search']=False
        base['training_preview_policy']='12 seeded random training images, no targeted extreme selection. Retain all incidentally observed failures and contradictions.'
    identity=_hash(base)
    if receipt_path.exists():
        saved=json.loads(receipt_path.read_text(encoding='utf-8'))
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
        proposal=_invoke_typed(client,'quantitative',payload,proposal_schema,
          load_agent_workspace_prompt('quantitative')+' ConstructionProposal JSON only. Select a currently evidenced nuisance hypothesis and supported recipe. Numeric Y targets/pivots must be fitted by luminance_calibration. A lack of perfect semantic proof alone is not a reason to abandon a scoped diagnostic; disclose uncertainty. Do not guess task labels or certify content preservation.')
        if proposal.candidate_id not in {c['candidate_id'] for c in remaining}:raise ValueError('Proposal changed frozen candidate identity')
        if proposal.descriptor_key not in descriptor_keys:raise ValueError('Unavailable measured descriptor')
        _write(receipt_path.parent/f'construction-attempt-{attempt}.json',{'proposal':proposal.model_dump(mode='json'),'input_hash':identity,'feedback':feedback})
        if proposal.status=='deferred':
            remaining=[c for c in remaining if c['candidate_id']!=proposal.candidate_id]
            feedback.append({'candidate_deferral':proposal.model_dump(mode='json')})
            if remaining:continue
            break
        if contract is not None:
            from shortcutminer.experiment.intervention_alignment import validate_operation
            ratio_choice=getattr(proposal,'channel_ratio_calibration',None)
            symbolic_operation=({'type':'channel_ratio','numerator_channel':contract['descriptor_key'].removeprefix('rgb_mean_'),
                'denominator_channel':(contract.get('denominator_key') or '').removeprefix('rgb_mean_')}
                if ratio_choice is not None else {'type':'luminance'} if proposal.luminance_calibration is not None
                else proposal.transformation.model_dump(mode='json'))
            try:
                validate_operation(contract,proposal.candidate_id,proposal.descriptor_key,symbolic_operation)
            except ValueError as error:
                feedback.append({'measurement_alignment_error':str(error),'prior_proposal':proposal.model_dump(mode='json')})
                continue
        calibration=None;operation=proposal.transformation
        if proposal.luminance_calibration is not None:
            calibration=calibrate_luminance(proposal.luminance_calibration,resources,root)
            operation=LuminanceTransform.model_validate(calibration['transformation'])
        if getattr(proposal,'channel_ratio_calibration',None) is not None:
            try:
                if quality_policy:
                    from shortcutminer.experiment.construction_quality import calibrate_bounded_ratio
                    from shortcutminer.intervention.protocol import BoundedChannelRatioTransform
                    calibration=calibrate_bounded_ratio(proposal.channel_ratio_calibration,contract,resources,root)
                    operation=BoundedChannelRatioTransform.model_validate(calibration['transformation'])
                else:
                    calibration=calibrate_channel_ratio(proposal.channel_ratio_calibration,contract,resources,root)
                    operation=ChannelRatioTransform.model_validate(calibration['transformation'])
            except ValueError as error:
                feedback.append({'construction_error':str(error),'prior_proposal':proposal.model_dump(mode='json')})
                continue
        try:
            preview_options={} if seek_counterexamples else {'seek_counterexamples':False}
            records,sheet=preview_construction(operation,resources,root,receipt_path.parent/f'preview-{attempt}',quality_policy=quality_policy,**preview_options)
        except ValueError as error:
            feedback.append({'construction_error':str(error),'prior_proposal':proposal.model_dump(mode='json')})
            continue
        target_check=None
        if contract is not None:
            from shortcutminer.experiment.intervention_alignment import measure_preview_target
            target_check=measure_preview_target(contract,records,Path(root),sheet.parent)
            _write(receipt_path.parent/f'measurement-check-{attempt}.json',target_check)
            if target_check['status']!='measured_change':
                feedback.append({'measurement_alignment_error':'The primary target did not measurably change on training previews. Revise the operation or defer; this is not absence of a shortcut.',
                                 'target_measurement_check':target_check,'prior_proposal':proposal.model_dump(mode='json')})
                continue
        visual_schema=VisualConstructionReviewV2 if quality_policy else VisualConstructionReview
        visual_input={'run_id':run_id,'mode':'construction_visual_review','task':task_description,'candidate_id':proposal.candidate_id,
           'candidate':next(c for c in candidates if c['candidate_id']==proposal.candidate_id),
           'transformation':operation.model_dump(mode='json'),'training_pairs':records,
           'response_schema':visual_schema.model_json_schema(),
           'display':'Pairs in row order, original left and changed right. Labels/padding are display annotations, not image evidence.'}
        if target_check is not None:
            visual_input['target_measurement_check']=target_check
        if quality_policy:
            visual_input['mechanism_contract']={k:proposal.model_dump(mode='json')[k] for k in ('mechanism_scope','mechanism_test','remaining_confounds','training_selection_criteria')}
            visual_input['quality_review_instruction']='Review every pair including target extremes. Distinguish nuisance changes from legitimate subject/scene colors. Severe color casts, lost task details or unsupported mechanism isolation require revise/defer. Uncertain semantics may support only a limited scoped diagnostic, never causality. Pixel diagnostics are descriptive; no new scientific threshold is applied.'
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
        visual=None if disable_qualitative else visual_schema.model_validate(vision_client.invoke_multimodal(run_id=run_id,agent_name='qualitative',
            payload_for_hash=visual_input,system_prompt=load_agent_workspace_prompt('qualitative')+' In construction_visual_review mode return only VisualConstructionReview JSON. Inspect actual original/changed pairs. Report whether the candidate cue was changed and task content remains visible, including clipping, borders and missing structure. A global pixel change fraction is not evidence that semantics were destroyed. Visual judgment is not ground truth; use unverified when uncertain.',
            content_parts=[{'type':'text','text':json.dumps(visual_input)},
              {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode(),'detail':'high'}}],
            validator=lambda v:visual_schema.model_validate(v),image_inputs=[image_input]))
        if visual is not None and (len(set(visual.reviewed_pair_ids))!=len(visual.reviewed_pair_ids) or not set(visual.reviewed_pair_ids).issubset({r['pair_id'] for r in records})):
            raise ValueError('Visual review cites unavailable pairs')
        if quality_policy and visual is not None and (set(visual.reviewed_pair_ids)!={r['pair_id'] for r in records} or visual.task_preservation=='damaged' or visual.mechanism_specificity=='unsupported' or visual.recommended_action!='accept_scoped'):
            feedback.append({'quality_review_requires_revision':visual.model_dump(mode='json'),'prior_proposal':proposal.model_dump(mode='json')})
            continue
        decision_input={'run_id':run_id,'mode':'construction_semantic_review','task':task_description,'proposal':proposal.model_dump(mode='json'),
          'fitted_transformation':operation.model_dump(mode='json'),'visual_review':visual.model_dump(mode='json') if visual is not None else None,
          'pixel_diagnostics':records,'response_schema':ConstructionDecision.model_json_schema(),
          'scope':'Training-only diagnostic authorization; no task-model predictions or held-out outcomes'}
        decision_input['cue_execution_check']=visual_input.get('cue_execution_check')
        if target_check is not None:
            decision_input['target_measurement_check']=target_check
        review_instruction=(' Qualitative is experimentally disabled: visual_review=null means NOT PERFORMED, never passed. Use only observed numerical diagnostics to authorize a scoped sensitivity test, or defer. Task preservation remains unverified; no semantic or causal certification is possible.' if disable_qualitative else ' Use the actual Qualitative paired-image review and pixel diagnostics. Global changes are expected for a global cue; never reject solely because most pixels changed. Consider lost class evidence, clipping, cue specificity and scope. An unverified semantic status can authorize a limited dependence diagnostic with retained limitations; it cannot authorize a causal claim. A failed visual review must not be accepted.')
        decision=_invoke_typed(client,'executive',decision_input,ConstructionDecision,
           load_agent_workspace_prompt('executive')+' ConstructionDecision JSON only.'+review_instruction)
        visual_record=visual.model_dump(mode='json') if visual is not None else None
        _write(receipt_path.parent/f'construction-review-{attempt}.json',{'visual':visual_record,'executive':decision.model_dump(mode='json'),'input':decision_input})
        if decision.decision=='accept' and (visual is None or visual.status!='failed'):
            saved={'status':'selected','input_hash':identity,'proposal':proposal.model_dump(mode='json'),
                'transformation':operation.model_dump(mode='json'),'calibration':calibration,'visual_review':visual_record,
                'executive_review':decision.model_dump(mode='json'),'truth_seen':False,'heldout_seen':False,
                'cue_execution_check':visual_input.get('cue_execution_check'),
                'scope':'training_only_before_evaluation','semantic_preservation':'unverified',
                'training_source_hashes':source_hashes,
                'construction_metadata_sha256':base['construction_metadata_sha256'],
                'preview_file_hashes':{p.relative_to(receipt_path.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sheet.parent.iterdir() if p.is_file()}}
            if disable_qualitative:saved['qualitative_review_status']='not_performed_ablation'
            if not seek_counterexamples:saved['active_counterexample_search']=False
            if target_check is not None: saved['measurement_alignment']=target_check
            if quality_policy:saved['construction_quality_policy']=quality_policy
            return _freeze(receipt_path,saved)
        feedback.append({'visual_review':visual_record,'executive_review':decision.model_dump(mode='json'),
                         'prior_proposal':proposal.model_dump(mode='json')})
        if decision.decision=='defer':
            remaining=[c for c in remaining if c['candidate_id']!=proposal.candidate_id]
            if not remaining:break
    saved={'status':'deferred','input_hash':identity,'training_source_hashes':source_hashes,'feedback':feedback,
        'truth_seen':False,'heldout_seen':False,'reason':'No construction accepted within the bounded training-only planning budget'}
    return _freeze(receipt_path,saved)
