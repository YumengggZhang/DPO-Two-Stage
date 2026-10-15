import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import reproduce as r

class ProtocolTests(unittest.TestCase):
    def test_four_stage2_and_thirty_baselines(self):
        jobs=r.jobs()
        self.assertEqual(len(jobs),42)
        self.assertEqual(len({(j['arm'],j['seed']) for j in jobs}),42)
        self.assertEqual({j['seed'] for j in jobs},{2003,2004,2005})
        self.assertNotIn(2002,{j['seed'] for j in jobs})

    def test_objective_routing_and_best_method4(self):
        self.assertEqual(r.objective_paths('dpo','harmful')[0],'dpo')
        self.assertEqual(r.objective_paths('dpo','benign')[0],'dpo')
        self.assertEqual(r.objective_paths('method2_chosen','harmful'),('kl','pi1',('positive',)))
        self.assertEqual(r.objective_paths('method2_chosen','benign'),('kl','initial',('positive',)))
        self.assertEqual(r.objective_paths('method3','harmful')[0],'kl')
        self.assertEqual(r.objective_paths('method3','benign')[0],'dpo')
        self.assertEqual(r.objective_paths('method4','benign'),('kl','initial',('positive','negative')))
        m=r.protocol()['settings']['method4_best']
        self.assertEqual((m['lr'],m['kl_lambda'],m['benign_kl_paths']),(1e-6,1.,'both'))

    def test_seeds_change_order_without_changing_data(self):
        rows=[{'pair_id':str(i)} for i in range(200)]
        a,b=r.ordered(rows,2003),r.ordered(rows,2004)
        self.assertNotEqual(a,b)
        self.assertEqual(a,r.ordered(rows,2003))
        self.assertEqual({v['pair_id'] for v in a},{v['pair_id'] for v in rows})
        self.assertEqual(rows[0]['pair_id'],'0')

    def test_bank_exposure_and_initialization(self):
        p=r.protocol()['settings']
        for arm in ['method1_dpo','method2_chosen','method3','method4_best']:
            self.assertEqual((p[arm]['data'],p[arm]['initial'],p[arm]['reference'],p[arm]['steps']),('stage2','pi1','pi1',247))
        baselines=[v for k,v in p.items() if k.startswith('baseline_')]
        self.assertEqual(len(baselines),10)
        self.assertEqual({v['base_model'] for v in baselines},{'qwen3_06b','qwen3_17b','qwen3_8b','r1_15b','r1_7b'})
        for v in baselines:
            self.assertEqual((v['initial'],v['reference']),('initial','initial'))
            self.assertEqual(v['lr'],1e-5 if v['benign']==0 else 3e-6)
            self.assertEqual((v['rows'],v['steps']),(828,104) if v['benign']==0 else (1828,229))
            self.assertEqual(v['training_gpus'],2 if v['base_model']=='r1_7b' else 1)

    def test_reporting_excludes_development_and_preserves_unknown(self):
        rows=[dict(id=f'h{i}',benchmark='harmful',unsafe=(True if i==0 else None if i==1 else False)) for i in range(863)]
        rows += [dict(id=f'b{i}',benchmark='benign',helpful=(None if i==0 else True),over_refusal=False) for i in range(3338)]
        rows += [dict(id=f'd{i}',benchmark='benign_dev',helpful=False,over_refusal=True) for i in range(200)]
        v=r.report_records(rows)
        self.assertEqual(v['ASR']['yes'],1)
        self.assertEqual(v['ASR']['unknown'],1)
        self.assertEqual(v['helpful']['unknown'],1)
        self.assertEqual(v['over_refusal']['yes'],0)
        self.assertEqual(v['helpful']['n'],3338)
        self.assertGreater(v['ASR']['upper_pct'],v['ASR']['pct'])
        rows[-1]['id']=rows[0]['id']
        with self.assertRaises(AssertionError):r.report_records(rows)

    @unittest.skipUnless((ROOT/'data/manifest.json').exists(),'Frozen data not yet exported')
    def test_frozen_data_integrity(self):
        manifest=r.check_data()
        self.assertEqual(manifest['banks']['stage2']['rows'],1972)

    @unittest.skipUnless((ROOT/'data/manifest.json').exists(),'Frozen data not yet exported')
    def test_prepare_all_fourteen_real_settings_and_reference_identities(self):
        manifest=r.check_data()
        assets=dict(initial='/fixed/pi0',pi1='/fixed/pi1',guard='/fixed/guard',
                    guard_proof=str(ROOT/'provenance/guard_proof.json'),
                    base_models={k:'/fixed/'+k for k in r.load(ROOT/'provenance/baseline_model_identities.json')})
        with tempfile.TemporaryDirectory() as temp:
            schedules=[]
            for job in r.jobs(seeds=[2003]):
                out=Path(temp)/job['arm']
                override=None
                if job.get('base_model') and job['base_model']!='qwen3_17b':
                    with self.assertRaises(AssertionError):r.prepare(out,job,assets,manifest)
                    # Neutral scores exercise preparation plumbing only, not model scoring.
                    values=[dict(pair_id=x['pair_id'],row_sha256=r.digest(x),scores={'positive':0.,'negative':0.}) for x in r.read_rows(ROOT/'data'/manifest['banks'][job['data']]['rows_file'])]
                    fixture=Path(temp)/(job['arm']+'.fixture.jsonl');r.save_rows(fixture,values)
                    override=dict(reference_path=str(fixture),reference_digest=r.digest(values))
                cfg=r.prepare(out,job,assets,manifest,override)
                models=r.load(out/'models.json')
                self.assertEqual(models['pi1']['path'],'/fixed/'+job['base_model'] if job.get('base_model') else '/fixed/pi1')
                self.assertEqual(len(r.read_rows(out/'data/sampled.jsonl')),job['rows'])
                self.assertEqual(cfg,r.prepare(out,job,assets,manifest,override))
                if job['data']=='stage2':schedules.append(r.load(out/'schedule.json')['sha256'])
                other=dict(job,seed=2004)
                with self.assertRaises(AssertionError):r.prepare(out,other,assets,manifest,override)
            self.assertEqual(len(set(schedules)),1)
            from common import identity
            # Evaluator family follows the actual trained model, including R1.
            for job in r.jobs(seeds=[2003]):
                out=Path(temp)/job['arm'];r.save_json(out/'training/full_lr1e5/complete.json',dict(status='complete',model_output='/final',files={},run_fingerprint='test'))
                self.assertEqual(identity(out,'full_lr1e5')['family'],job.get('family','qwen3_17b'))

    def test_two_gpu_lock_acquisition_releases_partial_group(self):
        import fcntl
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            paths=[Path(temp)/f'gpu{i}.lock' for i in (0,1)]
            for path in paths:path.touch()
            checks=iter(['0','1000','0','0'])
            def sleeping(_):
                # Both attempted FDs must already be closed before waiting.
                for path in paths:
                    with path.open('r') as handle:
                        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(r.subprocess,'check_output',side_effect=lambda *a,**k:next(checks)),patch.object(r.time,'sleep',side_effect=sleeping):
                with r.GPU([0,1],temp,count=2) as visible:
                    self.assertEqual(visible,'0,1')
                    for path in paths:
                        with path.open('r') as handle:
                            with self.assertRaises(BlockingIOError):fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for path in paths:
                with path.open('r') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_all_wild_public_data_and_seed_order(self):
        manifest=r.check_data()
        rows=r.read_rows(ROOT/'data/wild_mixed.jsonl')
        harmful=r.read_rows(ROOT/'data/harmful.jsonl')
        lookup={x['pair_id']:x for x in rows}
        self.assertEqual(len(rows),1828)
        self.assertTrue(all(x==lookup[x['pair_id']] for x in harmful))
        benign=[x for x in rows if x['task_type']=='benign']
        self.assertEqual(len(benign),1000)
        self.assertEqual(sum(x['group']=='vanilla_benign' for x in benign),498)
        for x in benign:
            self.assertEqual(x['source_evidence']['chosen']['dataset'],'allenai/wildjailbreak')
            self.assertEqual(x['source_evidence']['rejected']['dataset'],'allenai/wildguardmix')
        for model in ('qwen3_06b','qwen3_17b','qwen3_8b','r1_15b','r1_7b'):
            self.assertEqual(manifest['banks'][model+'_wild_mixed']['rows_digest'],r.digest(rows))
            self.assertEqual(manifest['banks'][model+'_wild_harmful']['rows_digest'],r.digest(harmful))
        for seed in (2003,2004,2005):
            ordered=r.ordered(rows,seed)
            self.assertEqual(len({x['pair_id'] for x in ordered}),1828)
            self.assertEqual(sum(x['task_type']=='harmful' for x in ordered),828)
            self.assertEqual(r.digest(ordered),r.digest(r.ordered(rows,seed)))

    def test_model_reference_cache_reuse_and_wrong_model_rejected(self):
        from unittest.mock import patch
        manifest=r.check_data()
        assets={'base_models':{key:'/fixed/'+key for key in r.load(ROOT/'provenance/baseline_model_identities.json')}}
        job=next(x for x in r.jobs(seeds=[2003]) if x['arm']=='baseline_r1_15b_wild_harmful')
        with tempfile.TemporaryDirectory() as temp:
            def stage(root,*unused):
                cfg=r.load(root/'config.json');entry=cfg['reference_model']
                fp=r.digest(dict(config_sha256=r.digest(cfg),reference_model=entry,rows_sha256=cfg['banks']['sampled']['sha256'],encoding_sha256=cfg['banks']['sampled']['encoded_sha256'],score_reduction='sequence_sum_including_EOS',parameters='FP32',forward='BF16',optimization_steps=0))
                values=[dict(pair_id=x['pair_id'],row_sha256=r.digest(x),scores={'positive':0.,'negative':0.},run_fingerprint=fp) for x in r.read_rows(root/'data/sampled.jsonl')]
                path=root/'reference/scores.jsonl';r.save_rows(path,values)
                r.save_json(root/'reference/complete.json',dict(status='complete',rows=1828,scores_sha256=r.digest(values),file_sha256=r.sha(path),reference_model_files_digest=r.digest(entry['files']),run_fingerprint=fp,optimization_steps=0,new_answers=0))
            with patch.object(r,'run_stage',side_effect=stage):
                result=r.prepare_reference_cache(Path(temp),job,assets,manifest,None)
            self.assertEqual(len(r.read_rows(Path(result['reference_path']))),828)
            with patch.object(r,'run_stage'):
                repeated=r.prepare_reference_cache(Path(temp),dict(job,seed=2004),assets,manifest,None)
            self.assertEqual(repeated,result)
            proof=Path(temp)/'reference_cache/r1_15b/reference/complete.json';value=r.load(proof);value['reference_model_files_digest']='wrong_model';r.save_json(proof,value)
            with patch.object(r,'run_stage'),self.assertRaises(AssertionError):r.prepare_reference_cache(Path(temp),job,assets,manifest,None)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable in this environment')
    def test_reference_scoring_is_gradient_free_and_resumable(self):
        import torch
        from prepare_reference import score_rows
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.arange(21,dtype=torch.float32).reshape(3,7)/10)
            def forward(self,input_ids,use_cache=False):
                from types import SimpleNamespace
                return SimpleNamespace(logits=self.weight[:input_ids.shape[1]].unsqueeze(0))
        model=Tiny();rows=[{'pair_id':str(i)} for i in range(2)]
        enc={x['pair_id']:{side:dict(ids=[1,2,3],start=1) for side in ('positive','negative')} for x in rows}
        emitted=[];done=score_rows(model,rows,enc,[],'fp',emitted.append)
        self.assertEqual(len(emitted),2);self.assertIsNone(model.weight.grad)
        retried=[];again=score_rows(model,rows,enc,emitted,'fp',retried.append)
        self.assertEqual(done,again);self.assertEqual(retried,[])
        wrong=copy.deepcopy(emitted);wrong[0]['run_fingerprint']='other'
        with self.assertRaises(AssertionError):score_rows(model,rows,enc,wrong,'fp',retried.append)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable in this environment')
    def test_cpu_optimizer_matches_adamw_and_resume(self):
        import torch
        from optimizer_factory import optimizer
        for kind in ('cpu','streaming_cpu'):
            torch.manual_seed(1)
            a=torch.nn.Linear(3,2);b=copy.deepcopy(a)
            opt=optimizer(a,{'optimizer':kind},lr=1e-5,weight_decay=1e-4)
            ref=torch.optim.AdamW(b.parameters(),lr=1e-5,weight_decay=1e-4,foreach=False)
            for i in range(3):
                opt.zero_grad();ref.zero_grad()
                x=torch.tensor([[1.,2.,3.]])
                a(x).square().sum().backward();b(x).square().sum().backward()
                opt.step();ref.step()
                self.assertTrue(all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters())))
            state=copy.deepcopy(opt.state_dict());opt.load_state_dict(state)
            self.assertTrue(all(torch.equal(opt.checkpoint_model(a)[k],v) for k,v in a.state_dict().items()))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable in this environment')
    def test_full_vocabulary_kl_loss_and_gradient(self):
        from kl_objective import self_test
        self.assertTrue(self_test()['full_vocabulary_forward_kl'])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable in this environment')
    def test_dpo_sequence_sum_and_both_answer_gradients(self):
        import math,torch
        from standard_dpo_objective_20260912 import standard_dpo_loss
        p=torch.tensor([-.4,-.8],requires_grad=True)
        n=torch.tensor([-1.,-.2,-.3],requires_grad=True)
        loss=standard_dpo_loss(p,n,float(p.detach().sum()),float(n.detach().sum()),.1)
        self.assertAlmostEqual(float(loss.detach()),math.log(2),places=6)
        gp,gn=torch.autograd.grad(loss,(p,n))
        self.assertTrue(torch.all(gp<0) and torch.all(gn>0))

if __name__=='__main__':unittest.main()
