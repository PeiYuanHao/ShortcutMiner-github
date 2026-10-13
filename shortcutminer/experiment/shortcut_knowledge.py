"""Evidence-led, dataset-neutral literature knowledge shared by the Agents."""
import json
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parents[2] / 'configs/knowledge/shortcut-literature-v1.json'

def load_shortcut_knowledge():
    catalog=json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
    assert catalog['schema_version']=='shortcut-literature-v1'
    ids={p['id'] for p in catalog['papers']}
    for family in catalog['families']:
        if not set(family['sources']).issubset(ids): raise ValueError('Unknown literature source')
    return catalog

def literature_prompt():
    catalog=load_shortcut_knowledge()
    entries=[{k:family[k] for k in ('id','cue','discovery','validation','runtime_status','limitations','sources')}
             for family in catalog['families']]
    return ('Literature method memory (general hypotheses, never evidence about the current dataset). '
            'Use current observations to select and reject hypotheses; never infer a shortcut from a benchmark name. '
            'Distinguish paper-reported examples from our engineering adaptations. Unavailable methods require prerequisites, '
            'not an invented execution result. A robustness failure alone is not a causal shortcut verdict.\n'+
            json.dumps({'families':entries,'papers':catalog['papers']},ensure_ascii=False,separators=(',',':')))

AXES={
 'global_photometry':lambda k:k in {'brightness','intensity_mean','intensity_std','contrast'} or k.startswith('luminance_'),
 'chromaticity':lambda k:k.startswith('rgb_') or k.startswith('saturation_'),
 'texture_structure':lambda k:k in {'grayscale_entropy','edge_density'},
 'spatial_localization':lambda k:any(k.startswith(v+'_') for v in ('border','center','top','bottom','left','right')),
 'acquisition_encoding':lambda k:k in {'file_size_bytes','width','height','aspect_ratio','frame_count'},
}

def hypothesis_coverage_context(visual):
    if visual is None:return {'available':False,'axes':[]}
    axes=[]
    for name,match in AXES.items():
        facts=[]
        for group in visual.group_summaries:
            values={k:v for k,v in group.descriptor_means.items() if match(k)}
            if values: facts.append({'group':group.group,'n':group.decoded_count,'descriptor_means':values})
        if facts: axes.append({'axis':name,'observed_groups':facts})
    return {'available':bool(axes),'scope':'Observed selected training images; descriptive, not model-reliance evidence',
       'instruction':'Inspect independent axes before allocating the candidate budget. Do not copy the top descriptor ranking as hypotheses. Merge only justified representations of the same mechanism. Opaque identity/path fields are not scientific cues.',
       'axes':axes}
