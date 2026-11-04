"""Numerical checks for known signals, amplitude scaling and crop/zero handling."""
import unittest
import numpy as np
from extract_features import acoustic_features, prepare_signal


class FeatureChecks(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.t = np.arange(self.sr)/self.sr

    def test_known_pitch(self):
        for hz in [100,150,300]:
            f, _, errors, _ = acoustic_features(.1*np.sin(2*np.pi*hz*self.t),self.sr)
            self.assertFalse(errors)
            self.assertAlmostEqual(f['f0_median_hz'],hz,delta=.2)
            self.assertGreater(f['voiced_fraction'],.95)

    def test_amplitude_scaling(self):
        rng = np.random.default_rng(42)
        x = .2*np.sin(2*np.pi*150*self.t)+.005*rng.normal(size=len(self.t))
        a,_,_,_ = acoustic_features(x,self.sr)
        b,_,_,_ = acoustic_features(x*.1,self.sr)
        self.assertAlmostEqual(a['rms_dbfs']-b['rms_dbfs'],20,places=6)
        self.assertAlmostEqual(a['f0_median_hz'],b['f0_median_hz'],places=4)
        self.assertAlmostEqual(a['cpps_db'],b['cpps_db'],delta=.05)
        self.assertAlmostEqual(a['spectral_slope_db_per_octave'],b['spectral_slope_db_per_octave'],places=6)

    def test_noise_vs_periodic(self):
        rng = np.random.default_rng(7)
        noise,_,_,_ = acoustic_features(.1*rng.normal(size=len(self.t)),self.sr)
        tone,_,_,_ = acoustic_features(.1*np.sin(2*np.pi*150*self.t),self.sr)
        self.assertGreater(tone['cpps_db'],noise['cpps_db']+5)
        self.assertLess(abs(noise['spectral_slope_db_per_octave']),2)

    def test_silence_is_missing_not_zero_pitch(self):
        f,flags,errors,_ = acoustic_features(np.zeros(self.sr),self.sr)
        self.assertIn('silent_or_constant_audio',flags)
        self.assertTrue(np.isnan(f['f0_median_hz']))
        self.assertTrue(np.isnan(f['cpps_db']))
        self.assertFalse(errors)

    def test_center_crop_and_zero_boundaries(self):
        x = np.ones(self.sr*3)
        y,meta = prepare_signal(x,self.sr)
        self.assertEqual(len(y),self.sr)
        self.assertEqual(meta['analysis_start_seconds'],1)
        x = np.r_[np.zeros(800),np.ones(8000),np.zeros(1600)]
        y,meta = prepare_signal(x,self.sr)
        self.assertEqual(len(y),8000)
        self.assertAlmostEqual(meta['analysis_start_seconds'],.05)
        self.assertAlmostEqual(meta['exact_zero_boundary_trim_seconds'],.15)


if __name__ == '__main__':
    unittest.main()
