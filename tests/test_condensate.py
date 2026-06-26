import unittest
import torch
import math
from nfn.condensate import FractalRFF

class TestFractalRFF(unittest.TestCase):
    def test_initialization(self):
        # Valid initialization
        rff = FractalRFF(d_in=16, n_features=32, n_scales=8)
        self.assertEqual(rff.n_features, 32)
        self.assertEqual(rff.out_dim, 64)
        self.assertEqual(rff.W.shape, (16, 32))
        self.assertEqual(rff.b.shape, (32,))
        self.assertAlmostEqual(rff._norm, math.sqrt(2.0 / 32))

    def test_n_features_indivisible_by_n_scales(self):
        # Auto-rounded to the nearest multiple of n_scales (no longer asserts)
        rff = FractalRFF(d_in=16, n_features=30, n_scales=8)
        self.assertEqual(rff.n_features % 8, 0)
        self.assertGreaterEqual(rff.n_features, 30)

    def test_forward_pass_shape(self):
        rff = FractalRFF(d_in=16, n_features=32, n_scales=8)
        x = torch.randn(4, 10, 16) # [batch, seq_len, d_in]
        out = rff(x)
        self.assertEqual(out.shape, (4, 10, 64))

    def test_forward_pass_values(self):
        rff = FractalRFF(d_in=4, n_features=8, n_scales=2, seed=42)
        x = torch.randn(2, 4)
        out = rff(x)

        # Check that output is bounded and has expected norm properties
        # The outputs are concatenated cos and sin, scaled by _norm
        self.assertTrue(torch.all(torch.abs(out) <= rff._norm + 1e-5))


from nfn.condensate import SpectralCondensate

class TestSpectralCondensate(unittest.TestCase):
    def test_initialization(self):
        # Default initialization should set U as identity for first `rank` dims
        rff_dim = 64
        rank = 16
        sc = SpectralCondensate(rff_dim=rff_dim, rank=rank)
        self.assertEqual(sc.rff_dim, rff_dim)
        self.assertEqual(sc.rank, rank)
        self.assertEqual(sc.U.shape, (rff_dim, rank))
        self.assertEqual(sc.S.shape, (rank,))

        # First min(rank, rff_dim) should be identity
        self.assertTrue(torch.allclose(sc.U[:rank, :rank], torch.eye(rank)))

    def test_condense(self):
        rff_dim = 32
        rank = 8
        sc = SpectralCondensate(rff_dim=rff_dim, rank=rank)

        # Create dummy features [N, rff_dim]
        N = 100
        features = torch.randn(N, rff_dim)
        sc.condense(features)

        self.assertEqual(sc.U.shape, (rff_dim, rank))
        self.assertEqual(sc.S.shape, (rank,))
        # Singular values should be normalized to [0, 1] with the first being ~1.0
        self.assertTrue(torch.all((sc.S >= 0) & (sc.S <= 1.0 + 1e-6)))

    def test_condense_cooccurrence(self):
        rff_dim = 32
        rank = 8
        sc = SpectralCondensate(rff_dim=rff_dim, rank=rank)

        # Create dummy co-occurrence matrix [V, V]
        V = 50
        cooccur = torch.randn(V, V)
        cooccur = cooccur @ cooccur.T # make symmetric semi-definite

        sc.condense_cooccurrence(cooccur)
        self.assertEqual(sc.U.shape, (rff_dim, rank))
        self.assertEqual(sc.S.shape, (rank,))
        self.assertTrue(torch.all((sc.S >= 0) & (sc.S <= 1.0 + 1e-6)))

    def test_forward_pass(self):
        rff_dim = 32
        rank = 8
        sc = SpectralCondensate(rff_dim=rff_dim, rank=rank)

        phi = torch.randn(4, 10, rff_dim) # [batch, seq_len, rff_dim]
        out = sc(phi)

        self.assertEqual(out.shape, (4, 10, rank))


from nfn.condensate import HelmholtzPhaseLocking

class TestHelmholtzPhaseLocking(unittest.TestCase):
    def test_initialization(self):
        hp = HelmholtzPhaseLocking(n_phases=8, n_iter=4, eta=0.1, sparse_k=2)
        self.assertEqual(hp.n_phases, 8)
        self.assertEqual(hp.n_iter, 4)
        self.assertEqual(hp.eta, 0.1)
        self.assertEqual(hp.sparse_k, 2)

    def test_forward_pass_default(self):
        hp = HelmholtzPhaseLocking(n_phases=4, n_iter=2, eta=0.1, sparse_k=0)

        B, L, rank = 2, 5, 8
        z = torch.randn(B, L, rank)

        theta, K_sim = hp(z)

        self.assertEqual(theta.shape, (B, L, 4))
        self.assertEqual(K_sim.shape, (B, L, L))
        # K_sim should be normalized cosine similarity
        self.assertTrue(torch.all((K_sim >= -1.0 - 1e-5) & (K_sim <= 1.0 + 1e-5)))

    def test_forward_pass_sparse_k(self):
        # Setting sparse_k = 2 should ensure that for each row in K_sim,
        # only the top 2 elements are non-zero (unless there are ties/negatives that get zeroed)
        hp = HelmholtzPhaseLocking(n_phases=4, n_iter=2, eta=0.1, sparse_k=2)

        B, L, rank = 2, 5, 8
        z = torch.randn(B, L, rank)

        theta, K_sim = hp(z)

        self.assertEqual(theta.shape, (B, L, 4))
        self.assertEqual(K_sim.shape, (B, L, L))

        # In each row, at most sparse_k elements should be non-zero
        non_zeros_per_row = (K_sim != 0).sum(dim=-1)
        self.assertTrue(torch.all(non_zeros_per_row <= 2))

    def test_forward_pass_with_init_phases(self):
        hp = HelmholtzPhaseLocking(n_phases=4, n_iter=2, eta=0.1, sparse_k=0)

        B, L, rank = 2, 5, 8
        z = torch.randn(B, L, rank)
        init_phases = torch.randn(B, L, 4)

        theta, K_sim = hp(z, init_phases=init_phases)

        self.assertEqual(theta.shape, (B, L, 4))
        self.assertEqual(K_sim.shape, (B, L, L))



from nfn.condensate import CondensateDecoder, NFMCKernelLayer

class TestCondensateDecoder(unittest.TestCase):
    def test_initialization_learnable(self):
        decoder = CondensateDecoder(n_phases=4, rank=8, vocab_size=100, learnable=True)
        # 2*n_phases + rank = 8 + 8 = 16
        self.assertEqual(decoder.proj.weight.shape, (100, 16))
        for p in decoder.parameters():
            self.assertTrue(p.requires_grad)

    def test_initialization_frozen(self):
        decoder = CondensateDecoder(n_phases=4, rank=8, vocab_size=100, learnable=False)
        for p in decoder.parameters():
            self.assertFalse(p.requires_grad)

    def test_forward_pass(self):
        decoder = CondensateDecoder(n_phases=4, rank=8, vocab_size=100)

        theta = torch.randn(2, 5, 4) # [B, L, n_phases]
        z = torch.randn(2, 5, 8)     # [B, L, rank]

        out = decoder(theta, z)
        self.assertEqual(out.shape, (2, 5, 100))

class TestNFMCKernelLayer(unittest.TestCase):
    def test_initialization(self):
        layer = NFMCKernelLayer(d_model=64, n_rff=32, n_scales=4, rank=8, n_phases=4)

        self.assertEqual(layer.rff.n_features, 32)
        self.assertEqual(layer.condensate.rank, 8)
        self.assertEqual(layer.phase_lock.n_phases, 4)

        # 2*n_phases + rank = 8 + 8 = 16
        self.assertEqual(layer.out_proj.weight.shape, (64, 16))

    def test_condense_from_hidden(self):
        layer = NFMCKernelLayer(d_model=64, n_rff=32, n_scales=4, rank=8, n_phases=4)

        h = torch.randn(10, 64) # [N, d_model]
        layer.condense_from_hidden(h)

        self.assertEqual(layer.condensate.U.shape, (64, 8)) # out_dim of rff is 2*n_rff = 64
        self.assertEqual(layer.condensate.S.shape, (8,))
        self.assertTrue(torch.all((layer.condensate.S >= 0) & (layer.condensate.S <= 1.0 + 1e-6)))

    def test_forward_pass(self):
        layer = NFMCKernelLayer(d_model=64, n_rff=32, n_scales=4, rank=8, n_phases=4)

        # B=2, L=5, d_model=64
        x = torch.randn(2, 5, 64)
        out = layer(x)

        self.assertEqual(out.shape, (2, 5, 64))


if __name__ == '__main__':
    unittest.main()
