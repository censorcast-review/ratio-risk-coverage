"""Synthetic tests only: never imports or opens external data arrays."""
import copy, json, pathlib, tempfile, unittest
from unittest.mock import patch
import numpy as np
import run
import evaluate as ev


def fixture():
    nitem=6;days=np.arange(1914,1942)
    truth=np.tile(np.repeat([[.1],[.1],[10.],[10.]],28,axis=1),(nitem,1))
    proposal=np.tile(np.repeat([[0.],[0.],[10.],[7.]],28,axis=1),(nitem,1))
    direct=np.tile(np.repeat([[0.],[0.],[0.],[1.]],28,axis=1),(nitem,1)).astype(np.float32)
    error=np.tile(np.repeat([[.25],[.25],[0.],[7.]],28,axis=1),(nitem,1)).astype(np.float32)
    demand=truth.astype(np.float32)
    meta={'item_id':np.repeat([f'i{x}' for x in range(nitem)],4),
          'cat_id':np.repeat(['FOODS','FOODS','HOBBIES','HOBBIES','HOUSEHOLD','HOUSEHOLD'],4)}
    return {'truth':truth,'proposal':proposal,'target_days':days},meta,{'error':error,'demand':demand}


class ExternalTests(unittest.TestCase):
    def approval(self):
        return {'status':'EXPLICIT_USER_APPROVAL_RECORDED','freeze_sha256':'abc',
          'authorization_phrase':'EXAMPLE_TEST_AUTHORIZATION',
          'approve_failed_guardian_external_purpose_amendment':True,
          'approved_external_use_count':1,'approved_policy_count':2,'certificate_issued':False,
          'user_approval_text':'Synthetic test approval; not a real authorization.',
          'approved_utc':'2000-01-01T00:00:00+00:00'}

    def test_formula_matches_frozen_development_arithmetic(self):
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from run_review4_tradeoffs import make_scores
        rng=np.random.default_rng(42)
        e=rng.uniform(-.2,5,(9,28)).astype(np.float32)
        mu=rng.uniform(-.2,4,(9,28)).astype(np.float32)
        e[0,:2]=[0,.2];mu[0,:2]=0
        source=make_scores({'error_x':e,'demand_x':mu},{'x':{'proposal':np.ones_like(e)}},ev.REFERENCE_MEAN)
        actual=ev.mixed_scores({'error':e,'demand':mu})
        for name,key in [('composed_excess_lambda_1','mixed_1'),('mixed_utility_lambda_0_25','mixed_0.25')]:
            np.testing.assert_array_equal(actual[name],source[key]['x'])

    def test_zero_demand_finite_and_no_demand_reference_reestimation(self):
        scores={'error':np.array([[.2,0]],np.float32),'demand':np.zeros((1,2),np.float32)}
        values=ev.mixed_scores(scores)
        np.testing.assert_allclose(values['composed_excess_lambda_1'],[[.2,0]])
        np.testing.assert_allclose(values['mixed_utility_lambda_0_25'],[[.8,0]])
        self.assertEqual(ev.REFERENCE_MEAN,1.3048948713314772)

    def test_nonfinite_heads_rejected(self):
        scores={'error':np.array([[np.nan,np.inf,.2]],np.float32),'demand':np.array([[1,2,0]],np.float32)}
        for value in ev.masks_from_scores(scores).values():
            self.assertFalse(value[0,0]);self.assertFalse(value[0,1])

    def test_missing_approval_never_inspects_ledger(self):
        with patch.object(pathlib.Path,'read_text',side_effect=AssertionError('No ledger read permitted')):
            with self.assertRaises(PermissionError):
                run.begin_opening({'authorization_phrase':'EXAMPLE_TEST_AUTHORIZATION'},'abc',{},pathlib.Path('/nonexistent'),pathlib.Path('/nonexistent'))

    def test_wrong_plan_and_string_boolean_rejected(self):
        plan={'authorization_phrase':'EXAMPLE_TEST_AUTHORIZATION'}
        for key,value in [('freeze_sha256','wrong'),('approved_policy_count',3),
                          ('approve_failed_guardian_external_purpose_amendment','true')]:
            a=self.approval();a[key]=value
            with self.assertRaises(PermissionError):run.validate_authorization(plan,'abc',a)

    def test_one_opening_resumes_only_same_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);ledger=root/'ledger.json';out=root/'out'
            ledger.write_text(json.dumps({'fresh_guardian_use_count':1,'external_opened':False,'certificate_issued':False}))
            plan={'authorization_phrase':'EXAMPLE_TEST_AUTHORIZATION','authorization_scope':'synthetic test'}
            first=run.begin_opening(plan,'abc',self.approval(),ledger,out)
            second=run.begin_opening(plan,'abc',self.approval(),ledger,out)
            self.assertEqual(first,second)
            state=json.loads(ledger.read_text());self.assertEqual(state['review5_external_use_count'],1)
            other=self.approval();other['freeze_sha256']='xyz'
            with self.assertRaises(PermissionError):run.begin_opening(plan,'xyz',other,ledger,out)

    def test_unauthorized_array_loader_never_stats_or_loads(self):
        with patch.object(pathlib.Path,'stat',side_effect=AssertionError('No file access')):
            with self.assertRaises(PermissionError):run.load_external_arrays({},pathlib.Path('/nonexistent'),{},'abc')

    def test_paired_direction_and_48_endpoints(self):
        data,meta,scores=fixture();report=ev.evaluate(data,meta,scores,reps=300)
        primary=report['primary']
        self.assertAlmostEqual(primary['row_coverage_difference'],-.25)
        self.assertGreater(primary['demand_coverage_difference'],0)
        self.assertTrue(primary['directional_confirmation_pass'])
        self.assertEqual(len(report['secondary']),24)
        self.assertEqual(report['secondary_endpoint_count'],48)
        self.assertTrue(report['joint_secondary_operating_check_pass'])
        self.assertEqual(primary['metrics']['composed_excess_lambda_1']['rows'],24*23)

    def test_first_five_targets_excluded_from_confirmation(self):
        data,meta,scores=fixture();before=ev.evaluate(data,meta,scores,reps=100)
        data['truth'][:,:5]=9999;data['proposal'][:,:5]=0
        after=ev.evaluate(data,meta,scores,reps=100)
        self.assertEqual(before['primary'],after['primary'])
        self.assertNotEqual(before['descriptive_full_28_sensitivity'],after['descriptive_full_28_sensitivity'])

    def test_group_reporting_does_not_mutate_masks(self):
        data,meta,scores=fixture();before={k:v.copy() for k,v in scores.items()}
        ev.evaluate(data,meta,scores,reps=100)
        for k,v in scores.items():np.testing.assert_array_equal(v,before[k])

    def test_zero_mass_and_missing_category_fail(self):
        data,meta,scores=fixture();data['truth'][:]=0
        meta['cat_id'][:]='FOODS'
        with np.errstate(invalid='ignore'):
            report=ev.evaluate(data,meta,scores,reps=100)
        self.assertFalse(report['primary']['directional_confirmation_pass'])
        self.assertFalse(report['joint_secondary_operating_check_pass'])
        self.assertIsNone(report['primary']['metrics']['composed_excess_lambda_1']['wape'])
        json.dumps(report,allow_nan=False)


if __name__=='__main__':unittest.main()
