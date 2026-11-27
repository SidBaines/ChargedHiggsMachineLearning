"""Regressions for relative azimuth and signed working point tie handling."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from hd_interp import backbone_residuals
from physics_rotation import rotate
from physics_samples import operating_points
from models.models import TestNetwork


class PhysicsTests(unittest.TestCase):
    def test_relative_frame_and_reconstruction(self):
        torch.manual_seed(7)
        x = torch.randn(3, 5, 5)
        x[..., 3] = x[..., :3].norm(dim=-1) + 0.7
        # First event wraps the global branch cut, second has no lepton,
        # and third has two leptons so the larger transverse momentum decides.
        types = torch.tensor([[2, 0, 3, 4, 5], [2, 3, 3, 5, 5], [1, 2, 0, 3, 5]])
        model = TestNetwork(hidden_dim=12, num_heads=3, num_attention_blocks=2,
                            num_particle_types=6, add_event_head=True,
                            feature_set=["m", "pt", "eta", "sin_dphi_lep", "cos_dphi_lep", "tag"]).eval()
        with torch.no_grad():
            base = model(x, types)
            rotated = model(rotate(x, 2.7), types)
            for k in base:
                torch.testing.assert_close(base[k], rotated[k], rtol=1e-5, atol=2e-5)
            torch.testing.assert_close(backbone_residuals(model,x,types)[-1], model._backbone_features(x,types))
            idx = torch.tensor([1, 2, 0, 4, 3])
            permuted = model(x[:,idx],types[:,idx])
            torch.testing.assert_close(base["event"], permuted["event"])
            torch.testing.assert_close(base["reco"][:,idx], permuted["reco"])

    def test_signed_threshold_respects_ties_and_empty_tail(self):
        score = np.array([3, 3, 2, 1, 2], dtype=np.float32)
        chan = np.array([1, 1, 1, 1, 2])
        cal = {"label":np.zeros(5,int), "weight":np.array([3.,-1.,3.,1.,20.])}
        test = dict(cal,dsid=np.ones(5,int))
        op = operating_points(cal,test,score,score,chan,chan,target=2)
        self.assertEqual(op["lvbb"]["test_background"], {"n":2,"yield":2.,"sumw2":10.})
        self.assertEqual(op["qqbb"]["test_background"]["n"],0)


if __name__ == "__main__":
    unittest.main()
