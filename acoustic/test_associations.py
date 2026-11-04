import unittest
import numpy as np
from scipy import stats
from analyze_associations import rank_model,zrank,pearson,permutation_p,bootstrap_rho


class AssociationChecks(unittest.TestCase):
    def test_unadjusted_matches_spearman_with_ties(self):
        x=np.array([1,3,2,3,8,6,6,9,1,5,11,8.])
        y=np.array([0,.5,1,1,3,2,1.5,2,0,2.5,3,2.])
        res=rank_model(np.column_stack([x,y]))
        self.assertAlmostEqual(res['rho'],stats.spearmanr(x,y).statistic,places=12)

    def test_partial_matches_inverse_correlation_formula(self):
        rng=np.random.default_rng(4);v=rng.normal(size=(120,5));v[:,1]+=.7*v[:,0]+v[:,2]
        inv=np.linalg.inv(np.corrcoef(zrank(v).T))
        expected=-inv[0,1]/np.sqrt(inv[0,0]*inv[1,1])
        self.assertAlmostEqual(rank_model(v)['rho'],expected,places=12)

    def test_hc3_matches_direct_sandwich(self):
        rng=np.random.default_rng(9);v=rng.normal(size=(100,4));z=zrank(v)
        X=np.column_stack([np.ones(len(z)),z[:,2:],z[:,0]])
        bread=np.linalg.inv(X.T@X);b=bread@X.T@z[:,1];e=z[:,1]-X@b
        h=np.diag(X@bread@X.T)
        cov=bread@X.T@np.diag(e**2/(1-h)**2)@X@bread
        res=rank_model(v)
        self.assertAlmostEqual(res['score_rank_beta_hc3_se'],np.sqrt(cov[-1,-1]),places=12)

    def test_common_covariate_attenuates_confounding(self):
        rng=np.random.default_rng(12);c=rng.normal(size=1500)
        x=c+rng.normal(size=len(c))*.3;y=c+rng.normal(size=len(c))*.3
        self.assertGreater(rank_model(np.c_[x,y])['rho'],.8)
        self.assertLess(abs(rank_model(np.c_[x,y,c])['rho']),.12)

    def test_permutation_resolution_and_bootstrap(self):
        x=np.arange(20.);v=np.c_[x,x]
        self.assertEqual(permutation_p(v,99,np.random.default_rng(2)),.01)
        lo,hi,n=bootstrap_rho(v,100,np.random.default_rng(3))
        self.assertAlmostEqual(lo,1);self.assertAlmostEqual(hi,1);self.assertEqual(n,100)


if __name__=='__main__':unittest.main()
