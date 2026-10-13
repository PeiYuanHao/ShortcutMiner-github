"""Safe CUDA ResNet training template invoked only by quantitative tools.

Trains fresh models within the current run. No baseline-run import, pickle module,
provider code, dataset-specific label mapping, or test-based model selection.
"""
from pathlib import Path
import json
import random
import time
import shutil

import numpy as np
from PIL import Image

from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.protocol.model_reliance import TaskModelArtifact, TaskModelCapabilities, TorchvisionTrainingOptions
from shortcutminer.task_models.reference import TaskModelError, _canonical_hash
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.experiment.waterbirds_paper import (
    torch_runtime, initialization, build_model, save_state, load_state, sha, transforms, seed_worker,
)

BACKEND_VERSION = "torchvision_resnet_matrix_v1"
FEATURE_VERSION = "torchvision_rgb224_v1"


class PathImages:
    def __init__(self, paths, labels, training, augmentation='natural_image'):
        self.paths, self.labels = paths, labels
        if augmentation in {'digit_preserving','spatial_preserving'}:
            from torchvision import transforms as T
            self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
        else:
            self.transform = transforms(training)

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(load_rgb_image_file(self.paths[index]).rgb)), self.labels[index]


class TorchvisionResNetMatrix:
    model_kind = "torchvision_resnet_matrix"
    capabilities = TaskModelCapabilities(predictions=True, probabilities=True, logits=True,
        training=True, checkpoint=True, image_input=True, intervention_inference=True)

    def __init__(self, model_id, options):
        self.model_id = model_id
        self.options = TorchvisionTrainingOptions.model_validate(options)
        self.torch = torch_runtime(self.options.model_dump())
        self.profiles = []
        self.adaptive_report = None
        self.classes_ = np.array([], dtype=str)
        self._cache_key, self._member_cache = None, None

    def _loader(self, paths, labels, training, seed):
        generator = self.torch.Generator().manual_seed(seed)
        return self.torch.utils.data.DataLoader(PathImages(paths, labels, training, self.options.augmentation),
            batch_size=self.options.batch_size, shuffle=training, num_workers=self.options.workers,
            pin_memory=True, worker_init_fn=seed_worker, generator=generator)

    def _build(self, architecture):
        weight_path, _ = initialization(architecture)
        model = build_model(architecture, weight_path)
        if len(self.classes_) != 2:
            model.fc = self.torch.nn.Linear(model.fc.in_features, len(self.classes_)).cuda()
        return model

    def _infer(self, model, paths, labels=None):
        model.eval(); output=[]
        with self.torch.inference_mode():
            for images, _ in self._loader(paths, labels or [0]*len(paths), False, 0):
                output.extend(model(images.cuda(non_blocking=True)).float().cpu().numpy())
        return np.asarray(output)

    def fit_matrix(self, train_paths, train_labels, validation_paths, validation_labels, directory, fingerprint, *, probe_callback=None, shared_cache=None):
        if not validation_paths:
            raise TaskModelError("CUDA training requires an isolated validation partition")
        self.classes_ = np.array(sorted(set(train_labels)), dtype=str)
        if len(self.classes_) < 2 or not set(validation_labels).issubset(self.classes_):
            raise TaskModelError("Invalid training/validation classes")
        mapping={label:i for i,label in enumerate(self.classes_)}
        train_y=[mapping[y] for y in train_labels]; val_y=[mapping[y] for y in validation_labels]
        self._screening_validation_labels = validation_labels
        directory.mkdir(parents=True, exist_ok=True)
        for architecture in self.options.architectures:
            for seed in self.options.seeds:
                name=f"{architecture}-seed{seed}"; output=directory/name; output.mkdir(exist_ok=True)
                if shared_cache is not None:
                    cached = shared_cache / name / 'profile.json'
                    cached.resolve().relative_to(shared_cache.resolve())
                    if cached.exists():
                        record = json.loads(cached.read_text(encoding='utf-8'))
                        source = cached.parent / 'best.npz'
                        source.resolve(strict=True).relative_to(shared_cache.resolve())
                        if record['classes'] != self.classes_.tolist() or sha(source) != record['weights_sha256']:
                            raise TaskModelError('Shared current-run probe cache identity changed')
                        target = output / 'best.npz'; shutil.copyfile(source, target)
                        profile = {k:v for k,v in record.items() if k != 'classes'}
                        profile.update(checkpoint_path=target, shared_current_run_reuse=True)
                        atomic_write_json(output/'progress.json', {'completed_from_shared_cache': True, 'input_hash': fingerprint}, error_type=TaskModelError)
                        self.profiles.append(profile)
                        if len(self.profiles) == 1 and probe_callback is not None:
                            model = self._build(architecture)
                            load_state(target, profile['weights_sha256'], model)
                            expand = probe_callback(model, profile)
                            del model; self.torch.cuda.empty_cache()
                            if not expand: return
                        continue
                random.seed(seed); np.random.seed(seed); self.torch.manual_seed(seed); self.torch.cuda.manual_seed_all(seed)
                model=self._build(architecture)
                optimizer=self.torch.optim.SGD(model.parameters(), lr=self.options.learning_rate,
                    momentum=self.options.momentum, weight_decay=self.options.weight_decay)
                scaler=self.torch.amp.GradScaler("cuda", enabled=self.options.amp)
                state={"input_hash":fingerprint,"epoch":0,"best_accuracy":-1.,"best_epoch":0,"history":[],"early_stopped":False}
                progress=output/'progress.json'
                if progress.exists():
                    state=json.loads(progress.read_text())
                    if state['input_hash'] != fingerprint: raise TaskModelError("Training resume identity changed")
                    load_state(output/state['last_reference'],state['last_sha256'],model,optimizer)
                    scaler.load_state_dict(state['scaler'])
                start=time.monotonic()
                for epoch in range(state['epoch']+1,self.options.epochs+1):
                    if state['early_stopped']: break
                    epoch_seed=seed+100000*epoch
                    random.seed(epoch_seed); np.random.seed(epoch_seed); self.torch.manual_seed(epoch_seed); self.torch.cuda.manual_seed_all(epoch_seed)
                    model.train(); loss_sum=0.; count=0
                    for images, labels in self._loader(train_paths,train_y,True,epoch_seed):
                        images,labels=images.cuda(non_blocking=True),labels.cuda(non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        with self.torch.autocast('cuda',enabled=self.options.amp):
                            loss=self.torch.nn.functional.cross_entropy(model(images),labels)
                        if not self.torch.isfinite(loss): raise TaskModelError("Nonfinite training loss")
                        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                        loss_sum+=float(loss.detach())*len(labels); count+=len(labels)
                    accuracy=float(np.mean(self._infer(model,validation_paths).argmax(1)==val_y))
                    entry={'epoch':epoch,'training_loss':loss_sum/count,'validation_accuracy':accuracy}
                    state['history'].append(entry)
                    if accuracy > state['best_accuracy']:
                        state.update(best_accuracy=accuracy,best_epoch=epoch,best_reference=f'best-epoch{epoch:03d}.npz')
                        state['best_sha256']=save_state(output/state['best_reference'],model)
                    state['epoch']=epoch
                    state['early_stopped']=epoch>=self.options.minimum_epochs and epoch-state['best_epoch']>=self.options.early_stopping_patience
                    if epoch%self.options.checkpoint_interval==0 or epoch==self.options.epochs or state['early_stopped']:
                        state['last_reference']=f'last-epoch{epoch:03d}.npz'
                        state['last_sha256']=save_state(output/state['last_reference'],model,optimizer)
                        state['scaler']=scaler.state_dict()
                        atomic_write_json(progress,state,error_type=TaskModelError)
                    print(json.dumps({'phase':'quantitative_task_model_training','architecture':architecture,'seed':seed,**entry}),flush=True)
                load_state(output/state['best_reference'],state['best_sha256'],model)
                best=output/'best.npz'; digest=save_state(best,model)
                _,init=initialization(architecture)
                self.profiles.append({'architecture':architecture,'seed':seed,'checkpoint_path':best,
                    'weights_sha256':digest,'best_epoch':state['best_epoch'],'epochs_completed':state['epoch'],
                    'validation_accuracy':state['best_accuracy'],'initialization':init,'elapsed_seconds':time.monotonic()-start})
                if shared_cache is not None:
                    shared = shared_cache / name; shared.mkdir(parents=True, exist_ok=True)
                    shared.resolve().relative_to(shared_cache.resolve())
                    shutil.copyfile(best, shared/'best.npz')
                    atomic_write_json(shared/'profile.json', {**{k:v for k,v in self.profiles[-1].items() if k != 'checkpoint_path'},
                        'classes':self.classes_.tolist()}, error_type=TaskModelError)
                if len(self.profiles) == 1 and probe_callback is not None:
                    expand = probe_callback(model, self.profiles[-1])
                    if not expand:
                        del model,optimizer; self.torch.cuda.empty_cache()
                        return
                del model,optimizer; self.torch.cuda.empty_cache()

    def member_logits(self, paths):
        # Inference cache is restricted to this loaded adapter and exact file hashes.
        key=tuple((str(p),sha(Path(p))) for p in paths)
        if key != self._cache_key:
            outputs={}
            for profile in self.profiles:
                model=self._build(profile['architecture'])
                load_state(profile['checkpoint_path'],profile['weights_sha256'],model)
                outputs[f"{profile['architecture']}-seed{profile['seed']}"]=self._infer(model,paths)
                del model; self.torch.cuda.empty_cache()
            self._cache_key,self._member_cache=key,outputs
        return self._member_cache

    def predict_logits(self, paths): return np.mean(list(self.member_logits(paths).values()),axis=0)

    def predict_proba(self, paths):
        logits=self.predict_logits(paths); logits=logits-logits.max(axis=1,keepdims=True)
        probabilities=np.exp(logits); return probabilities/probabilities.sum(axis=1,keepdims=True)

    def predict(self, paths): return self.classes_[self.predict_logits(paths).argmax(1)].tolist()

    def save_safe_artifact(self, *, run_id, run_root, model_reference, training_fingerprint, training_manifest):
        root=run_root.resolve(strict=True); profile_records=[]
        for profile in self.profiles:
            record={k:v for k,v in profile.items() if k!='checkpoint_path'}
            record['weights_reference']=profile['checkpoint_path'].resolve(strict=True).relative_to(root).as_posix()
            profile_records.append(record)
        profile_reference=(Path(model_reference).parent/'profiles.json').as_posix()
        payload={'backend_version':BACKEND_VERSION,'training_fingerprint':training_fingerprint,
                 'classes':self.classes_.tolist(),'options':self.options.model_dump(mode='json'),'profiles':profile_records}
        if self.adaptive_report is not None:
            payload['adaptive_report'] = self.adaptive_report
        model_hash=_canonical_hash(payload)
        atomic_write_json(root/profile_reference,payload,error_type=TaskModelError)
        artifact=TaskModelArtifact(run_id=run_id,model_id=self.model_id,model_kind=self.model_kind,
            backend='torchvision_resnet_matrix',backend_version=BACKEND_VERSION,feature_version=FEATURE_VERSION,
            model_hash=model_hash,training_fingerprint=training_fingerprint,model_reference=model_reference,
            weights_reference=profile_reference,capabilities=self.capabilities,classes=self.classes_.tolist(),
            limitations=['Ensemble metrics are separate from individual architecture/seed results; no test-based checkpoint selection.',
                *(['Adaptive probe samples are excluded from final evaluation; stopped pilot is not proof of no shortcuts.'] if self.adaptive_report else [])])
        atomic_write_json(root/model_reference,{**artifact.model_dump(mode='json'),'profiles_sha256':sha(root/profile_reference),
            'training':training_manifest},error_type=TaskModelError)
        return artifact

    @classmethod
    def load_safe_artifact(cls, model_path, *, run_root, expected_run_id):
        root=run_root.resolve(strict=True); model_path=model_path.resolve(strict=True); model_path.relative_to(root)
        metadata=json.loads(model_path.read_text()); artifact=TaskModelArtifact.model_validate({k:metadata[k] for k in TaskModelArtifact.model_fields if k in metadata})
        if artifact.run_id!=expected_run_id or artifact.backend!='torchvision_resnet_matrix' or artifact.backend_version!=BACKEND_VERSION:
            raise TaskModelError('CUDA model run/backend/version mismatch')
        profiles_path=(root/artifact.weights_reference).resolve(strict=True); profiles_path.relative_to(root)
        if sha(profiles_path)!=metadata['profiles_sha256']: raise TaskModelError('CUDA profile manifest hash mismatch')
        payload=json.loads(profiles_path.read_text())
        if _canonical_hash(payload)!=artifact.model_hash or payload['classes']!=artifact.classes or payload['training_fingerprint']!=artifact.training_fingerprint:
            raise TaskModelError('CUDA model identity mismatch')
        model=cls(artifact.model_id,payload['options']); model.classes_=np.asarray(artifact.classes)
        expected={(a,s) for a in model.options.architectures for s in model.options.seeds}
        actual=[(p['architecture'],p['seed']) for p in payload['profiles']]
        report = payload.get('adaptive_report')
        if report is not None:
            from shortcutminer.experiment.evidence_promotion import stable_hash
            if report.get('report_hash') != stable_hash({k:v for k,v in report.items() if k != 'report_hash'}) or metadata['training'].get('adaptive_report') != report:
                raise TaskModelError('Adaptive decision identity changed')
            if report['decision'] not in {'expand_models', 'stop_low_effect', 'deferred_pair_budget_exhausted', 'deferred_processing_control_failed', 'deferred_processing_task_not_learned', 'deferred_task_not_learned', 'deferred_insufficient_pairs'}:
                raise TaskModelError('Unknown adaptive budget decision')
            if report['decision'] != 'expand_models':
                expected = {(model.options.architectures[0], model.options.seeds[0])}
            model.adaptive_report = report
        if set(actual)!=expected or len(actual)!=len(expected): raise TaskModelError('CUDA profile set incomplete')
        for p in payload['profiles']:
            path=(root/p['weights_reference']).resolve(strict=True); path.relative_to(root)
            if sha(path)!=p['weights_sha256']: raise TaskModelError('CUDA checkpoint hash mismatch')
            model.profiles.append({**p,'checkpoint_path':path})
        return model,artifact
