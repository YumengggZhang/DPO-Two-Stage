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
    def test_six_settings_eighteen_unique_new_seeds(self):
        jobs=r.jobs()
        self.assertEqual(len(jobs),18)
        self.assertEqual(len({(j['arm'],j['seed']) for j in jobs}),18)
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
        self.assertEqual((p['dpo_harmful_only']['rows'],p['dpo_harmful_only']['steps']),(828,104))
        self.assertEqual((p['dpo_mixed']['rows'],p['dpo_mixed']['steps'],p['dpo_mixed']['lr']),(1828,229,3e-6))

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
    def test_prepare_all_six_real_banks_and_reference_identities(self):
        manifest=r.check_data()
        assets=dict(initial='/fixed/pi0',pi1='/fixed/pi1',guard='/fixed/guard',
                    guard_proof=str(ROOT/'provenance/guard_proof.json'))
        with tempfile.TemporaryDirectory() as temp:
            schedules=[]
            for job in r.jobs(seeds=[2003]):
                out=Path(temp)/job['arm']
                cfg=r.prepare(out,job,assets,manifest)
                models=r.load(out/'models.json')
                self.assertEqual(models['pi1']['path'],'/fixed/pi1' if job['initial']=='pi1' else '/fixed/pi0')
                self.assertEqual(len(r.read_rows(out/'data/sampled.jsonl')),job['rows'])
                self.assertEqual(cfg,r.prepare(out,job,assets,manifest))
                if job['data']=='stage2':schedules.append(r.load(out/'schedule.json')['sha256'])
                other=dict(job,seed=2004)
                with self.assertRaises(AssertionError):r.prepare(out,other,assets,manifest)
            self.assertEqual(len(set(schedules)),1)

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
