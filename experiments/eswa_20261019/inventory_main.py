"""Export retained main artifacts and incomplete denominators without Agent calls."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.e4_ablation.variants.policies import write
from shortcutminer.experiment.waterbirds_paper import sha


def csv_write(path,rows,fields):
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    study=ROOT/'results_eswa_single';directory=study/'exports/main-v1-inventory'
    if not (study/'extensions/SCOPE_PLAN.json').is_file():raise ValueError('Select extension scope before outcome extraction')
    directory.mkdir(parents=True,exist_ok=False)
    with (study/'FORMAL_TASK_STATUS.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    candidates=[];references=[];profiles=[]
    for row in rows:
        attempt=study/'audits/formal'/row['case_id']/row['variant']/'attempt-01'
        run=attempt/(row['case_id']+'-'+row['variant'].replace('_','-'))
        result=run/'summarize/result.json'
        findings={}
        if result.exists():
            findings={item['candidate_id']:item for item in json.loads(result.read_text())['payload']['findings']}
        dispositions=run/'candidate-validation-dispositions.json'
        discovery=attempt/(run.name+'-discovery')/'identify/output.json'
        items=[]
        if dispositions.exists():items=json.loads(dispositions.read_text())['candidates']
        elif discovery.exists():items=json.loads(discovery.read_text())['candidates']
        for item in items:
            cid=item['candidate_id'];finding=findings.get(cid,{})
            candidates.append({'case_id':row['case_id'],'variant':row['variant'],'candidate_id':cid,
                'audit_status':row['status'],'audit_reason':row['reason'],
                'canonical_summary_exists':result.exists(),'agent_assessment':finding.get('assessment',''),
                'cheap_state':item.get('cheap_state',''),'human_adjudication':'PENDING',
                'source_reference':str(dispositions if dispositions.exists() else discovery),
                'assessment_is_independent_truth':False})
        paths=[attempt/'EXECUTION_RECEIPT.json',run/'audit_report.json',run/'final_result.json',result,dispositions]
        for pattern in ('quantitative/*/result.json','qualitative/*/result.json','models/task/*/training_manifest.json',
            'screening/**/*.json','interventions/**/*.json','experiments/*/*.json'):
            paths.extend(run.glob(pattern))
        for path in sorted(set(paths)):
            if path.is_file():references.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
                'case_id':row['case_id'],'variant':row['variant'],'audit_status':row['status']})
        for path in run.glob('models/task/**/profile.json'):
            profile=json.loads(path.read_text());profiles.append({'case_id':row['case_id'],'variant':row['variant'],
                'path':str(path),'profile':profile})
    csv_write(directory/'TASK_STATUS.csv',rows,list(rows[0]))
    fields=['case_id','variant','candidate_id','audit_status','audit_reason','canonical_summary_exists',
        'agent_assessment','cheap_state','human_adjudication','source_reference','assessment_is_independent_truth']
    csv_write(directory/'CANDIDATE_COVERAGE.csv',candidates,fields)
    csv_write(directory/'HUMAN_ADJUDICATION_PENDING.csv',[
        {key:row[key] for key in ('case_id','variant','candidate_id','human_adjudication')} for row in candidates],
        ['case_id','variant','candidate_id','human_adjudication'])
    write(directory/'ARTIFACT_INDEX.json',{'references':references,'model_profiles':profiles,
        'note':'Index preserves canonical and partial artifacts separately. Agent assessments are not human/oracle labels. Training elapsed is not GPU active kernel time.'})
    write(directory/'INVENTORY_RECEIPT.json',{'api_calls':0,'new_training_runs':0,'scheduled_audits':len(rows),
        'completed':sum(row['status']=='COMPLETED' for row in rows),'failed_retained':sum(row['status']=='FAILED_RETAINED' for row in rows),
        'candidate_rows':len(candidates),'indexed_artifacts':len(references),'human_adjudication':'PENDING',
        'limitations':'No claim validation, human adjudication, oracle matching or pooled independent-dataset inference was performed.'})
    print(json.dumps({'status':'INVENTORIED','audits':len(rows),'candidate_rows':len(candidates),
        'indexed_artifacts':len(references),'api_calls':0,'directory':str(directory)}))


if __name__=='__main__':main()
