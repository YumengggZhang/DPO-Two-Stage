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
            self.assertEqual((v['initial'],v['reference'],v['lr']),('initial','initial',1e-5))
            self.assertEqual((v['rows'],v['steps']),(1000,125) if v['benign']==0 else (2000,250))
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
                cfg=r.prepare(out,job,assets,manifest)
                models=r.load(out/'models.json')
                self.assertEqual(models['pi1']['path'],'/fixed/'+job['base_model'] if job.get('base_model') else '/fixed/pi1')
                self.assertEqual(len(r.read_rows(out/'data/sampled.jsonl')),job['rows'])
                self.assertEqual(cfg,r.prepare(out,job,assets,manifest))
                if job['data']=='stage2':schedules.append(r.load(out/'schedule.json')['sha256'])
                other=dict(job,seed=2004)
                with self.assertRaises(AssertionError):r.prepare(out,other,assets,manifest)
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

    def test_historical_schedules_and_balanced_new_seeds(self):
        manifest=r.check_data()
        for bank,expected in [('clean1000','fde70b759060fa0826c5f701e40c773f43abf70c23a8fc8453a4178aeb607000'),('mixed2000','633025bd275f62d492a8f7226e34fd91d2d872716e9ef35e6b5591b9f105d5ef')]:
            rows=r.read_rows(ROOT/'data'/manifest['banks']['qwen3_17b_'+bank]['rows_file'])
            ordered=r.ordered(rows,2002,bank=='mixed2000')
            self.assertEqual(r.digest([[x['pair_id'] for x in ordered[i:i+8]] for i in range(0,len(ordered),8)]),expected)
            for seed in (2003,2004,2005):
                ordered=r.ordered(rows,seed,bank=='mixed2000')
                if bank=='mixed2000':
                    self.assertTrue(all(sum(x['task_type']=='harmful' for x in ordered[i:i+8])==4 for i in range(0,2000,8)))

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
