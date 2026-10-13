"""Offline DINO -> SAM -> optional ViTMatte. Predicted masks are not ground truth."""
import gc
import hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from shortcutminer.protocol.segmentation import LocalForegroundSpec
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.intervention.segmentation import SegmentationError, SegmentationMask

def verify_model_files(model, *, tokenizer=False):
    directory=Path(model.directory)
    if not directory.is_absolute(): raise SegmentationError('Local model directory must be absolute')
    directory=directory.resolve(strict=True)
    required={'config.json','preprocessor_config.json','model.safetensors'}
    if tokenizer: required.update({'tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt'})
    if set(model.files_sha256)!=required: raise SegmentationError('Local model must freeze every required model/processor file')
    for name,expected in model.files_sha256.items():
        path=(directory/name).resolve(strict=True);path.relative_to(directory)
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for data in iter(lambda:f.read(1048576),b''):digest.update(data)
        if digest.hexdigest()!=expected: raise SegmentationError('Frozen model file hash changed: '+name)
    return directory

class LocalForegroundSegmentationProvider:
    provider_name='local_dino_sam_vitmatte'
    provider_version='local_foreground_v1'
    available=True

    def __init__(self,spec):
        self.spec=LocalForegroundSpec.model_validate(spec)
        if importlib.metadata.version('transformers')!=self.spec.transformers_version:
            raise SegmentationError('Frozen Transformers version differs')
        sam=verify_model_files(self.spec.sam)
        dino=verify_model_files(self.spec.grounding_dino,tokenizer=True)
        matte=verify_model_files(self.spec.vitmatte) if self.spec.vitmatte else None
        import torch
        from transformers import SamModel,SamProcessor,GroundingDinoForObjectDetection,GroundingDinoProcessor,VitMatteForImageMatting,VitMatteImageProcessor
        if not torch.cuda.is_available(): raise SegmentationError('Local foreground protocol requires CUDA')
        self.torch=torch
        self.sam_processor=SamProcessor.from_pretrained(sam,local_files_only=True)
        self.sam=SamModel.from_pretrained(sam,local_files_only=True,use_safetensors=True).eval().to('cuda')
        self.dino_processor=GroundingDinoProcessor.from_pretrained(dino,local_files_only=True)
        self.dino=GroundingDinoForObjectDetection.from_pretrained(dino,local_files_only=True,use_safetensors=True).eval().to('cuda')
        self.matte_processor=VitMatteImageProcessor.from_pretrained(matte,local_files_only=True) if matte else None
        self.matte=VitMatteForImageMatting.from_pretrained(matte,local_files_only=True,use_safetensors=True).eval().to('cuda') if matte else None
        self.last_details={}

    def segment(self,*,image_path,mask_path,max_file_bytes,max_pixels):
        if mask_path is not None: raise SegmentationError('Local foreground must not consume an oracle mask')
        image=Image.fromarray(load_rgb_image_file(image_path,max_file_bytes=max_file_bytes,max_pixels=max_pixels).rgb)
        s=self.spec; torch=self.torch
        inputs=self.dino_processor(images=image,text=s.object_prompt,return_tensors='pt',
            size={'shortest_edge':s.detector_shortest_edge,'longest_edge':s.detector_longest_edge}).to('cuda')
        with torch.inference_mode(): outputs=self.dino(**inputs)
        result=self.dino_processor.post_process_grounded_object_detection(outputs,input_ids=inputs.input_ids,
            threshold=s.box_threshold,text_threshold=s.text_threshold,target_sizes=[image.size[::-1]])[0]
        if len(result['boxes'])==0: raise SegmentationError('Local detector found no foreground; no heuristic fallback')
        indices=result['scores'].argsort(descending=True)[:s.max_boxes]
        boxes=result['boxes'][indices].float().cpu().tolist()
        scores=result['scores'][indices].float().cpu().tolist()
        sam_inputs=self.sam_processor(images=image,input_boxes=[boxes],return_tensors='pt').to('cuda')
        with torch.inference_mode(): sam_outputs=self.sam(**sam_inputs)
        masks=self.sam_processor.image_processor.post_process_masks(sam_outputs.pred_masks.cpu(),
            sam_inputs['original_sizes'].cpu(),sam_inputs['reshaped_input_sizes'].cpu())[0]
        quality=sam_outputs.iou_scores[0].float().cpu().numpy()
        foreground=np.zeros((image.height,image.width),dtype=bool);selected_scores=[]
        for i,row in enumerate(quality):
            best=int(row.argmax());score=float(row[best]);selected_scores.append(score)
            if score < s.minimum_sam_score: raise SegmentationError('SAM predicted mask quality below frozen bound')
            foreground |= masks[i,best].numpy().astype(bool)
        if not foreground.any(): raise SegmentationError('Empty foreground mask')
        alpha=None
        if self.matte is not None:
            binary=Image.fromarray(foreground.astype('uint8')*255)
            trimap=np.zeros(foreground.shape,dtype='uint8')
            trimap[np.asarray(binary.filter(ImageFilter.MaxFilter(s.trimap_kernel)))>0]=128
            trimap[np.asarray(binary.filter(ImageFilter.MinFilter(s.trimap_kernel)))>0]=255
            matte_inputs=self.matte_processor(images=image,trimaps=Image.fromarray(trimap),return_tensors='pt').to('cuda')
            with torch.inference_mode(): alpha=self.matte(**matte_inputs).alphas[0,0].float().cpu().numpy()
            if alpha.shape[0]<image.height or alpha.shape[1]<image.width: raise SegmentationError('Invalid matte dimensions')
            alpha=alpha[:image.height,:image.width]
            if not np.isfinite(alpha).all(): raise SegmentationError('Nonfinite alpha matte')
            # Never shrink the SAM protection region or blend/modify its foreground pixels.
            foreground |= alpha >= s.alpha_protection_threshold
        if s.protection_margin:
            foreground=np.asarray(Image.fromarray(foreground.astype('uint8')*255).filter(ImageFilter.MaxFilter(2*s.protection_margin+1)))>0
        coverage=float(foreground.mean())
        if not s.minimum_coverage <= coverage <= s.maximum_coverage:
            raise SegmentationError('Foreground coverage outside frozen bounds')
        self.last_details={'boxes':boxes,'detector_scores':scores,'sam_predicted_iou_scores':selected_scores,
            'coverage':coverage,'oracle_mask_used':False,'alpha_refinement':self.matte is not None,
            'limitation':'Predicted foreground quality is not independently guaranteed; protected pixels are preserved, segmentation errors may remove actual foreground.'}
        return SegmentationMask(mask=foreground,coverage=coverage,provider=self.provider_name,version=self.provider_version)

    def close(self):
        self.sam=None;self.dino=None;self.matte=None
        gc.collect();self.torch.cuda.empty_cache()
