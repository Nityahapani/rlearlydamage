"""Software-only checks: synthetic arrays never enter scientific artifacts or target outcomes."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p14_analysis',ROOT/'scripts/analyze_p14.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class AnalysisChecks(unittest.TestCase):
    def test_practical_threshold_direction(self):
        self.assertEqual(a.directional(-np.ones(32),.5,'less'),0)
        self.assertEqual(a.directional(np.ones(32),-.5,'greater'),0)
        self.assertEqual(a.directional(np.zeros(32),.5,'less'),1)
        self.assertEqual(a.directional(np.zeros(32),-.5,'greater'),1)
        self.assertEqual(a.tost(np.zeros(32),.3),0)
        self.assertEqual(a.tost(np.ones(32),.3),1)
        self.assertEqual(a.tci(np.ones(32)),[1,1,1])
        self.assertEqual(a.bootstrap_ci(np.arange(32),reps=100),a.bootstrap_ci(np.arange(32),reps=100))

    def test_fixed_n_guard_and_full_synthetic_pipeline(self):
        P=json.loads((ROOT/'prereg/phase14.json').read_text())
        old=Path.cwd()
        with tempfile.TemporaryDirectory(prefix='p14-software-only-') as tmp:
            os.chdir(tmp)
            try:
                Path('prereg').mkdir();Path('results/summary').mkdir(parents=True)
                Path('prereg/phase14.json').write_text(json.dumps(P))
                for start in range(0,32,4):
                    base=6+np.arange(start,start+4)[:,None]/100
                    hist=np.tile(base,(1,21));noise=(np.arange(start,start+4)-15.5)[:,None]/200
                    native=hist-1.7+noise;native[:,-5:]=hist[:,-5:]-.8+noise;native[:,0]=-99
                    vals={
                        'fork/clean/donor/CCC':hist,
                        'pause/damaged/donor/CCC':hist,
                        'pause/damaged/donor/DDD':native,
                        'pause/damaged/donor/CDD':hist+.05,
                        'pause/damaged/donor/DCC':hist-1.5+noise,
                        'pause/damaged/zero/CCC':hist,
                        'pause/damaged/zero/CDC':hist-.2,
                        'pause/damaged/zero/DCC':hist-1.4,
                        'pause/damaged/zero/DDC':hist-1.6,
                        'pause/reset/donor/DDD':hist-1.4,
                        'pause/reset/donor/CDD':hist+.1}
                    cells={k:dict(eval_return_mean=v.tolist()) for k,v in vals.items()}
                    for k in P['second_eval_cells']:cells[k]['eval_alt_return_mean']=(vals[k]+.03).tolist()
                    d=dict(seeds=P['training_seeds'][start:start+4],protocol=dict(smoke=False,window=[390,440]),
                           state_fingerprints=dict(C='synthetic-'+str(start),D='synthetic-D',P='synthetic-P'),
                           status='complete',cells=cells,eval_updates=list(range(440,641,10)),
                           checks=dict(DDD_equals_untouched_natural_D=True,CCC_equals_untouched_natural_C=True),
                           fork_eval=dict(updates=list(range(390,441,10)),C=[base[:,0].tolist()]*6),
                           fork_eval_alt=dict(updates=list(range(390,441,10)),C=[(base[:,0]+.03).tolist()]*6))
                    path=Path(f'results/summary/SYNTHETIC-p14-actor-repair-s{1200+start}to{1203+start}-Asterix-MinAtar-software.json')
                    if start<28:
                        path.write_text(json.dumps(d))
                        with self.assertRaises(SystemExit):a.select_complete()
                        self.assertFalse(Path('results/summary/p14_single_result-analysis.json').exists())
                    else:path.write_text(json.dumps(d))
                with contextlib.redirect_stdout(io.StringIO()):a.main()
                r=json.loads(Path('results/summary/p14_single_result-analysis.json').read_text())
                self.assertTrue(r['primary_success']);self.assertEqual(r['n'],32)
                self.assertAlmostEqual(r['primary']['native_AUC_harm']['CI95'][0],-1.475)
                self.assertAlmostEqual(r['primary']['native_late_harm']['CI95'][0],-.8)
                self.assertAlmostEqual(r['fresh_Adam']['actor_main'][0],-1.4)
                self.assertAlmostEqual(r['fresh_Adam']['critic_main'][0],-.2)
                self.assertAlmostEqual(r['fresh_Adam']['actor_in_onset_critic'][0],-1.4)
                self.assertAlmostEqual(r['fresh_Adam']['actor_rescue'][0],1.4)
                self.assertAlmostEqual(r['alternate_evaluation']['native_AUC_harm'][0],-1.475)
                self.assertEqual(r['seed_consistency'],dict(native_AUC_negative=32,actor_rescue_positive=32))
                self.assertEqual(len(r['source_hashes']),8)
            finally:os.chdir(old)


if __name__=='__main__':unittest.main()
