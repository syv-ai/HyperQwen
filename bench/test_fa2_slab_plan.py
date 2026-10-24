"""Check slab coverage and masking before GPU/model validation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"pthfa/python"))
from pth_flash_attn.fa2_slab_plan import SLAB_TOKENS, slabs, workspace_bytes


class SlabPlanTests(unittest.TestCase):
    def test_every_key_is_covered_once_with_bounded_storage(self):
        for n in (17, 2048, 16383, 16384, 16385, 32767, 32768, 32769, 150528, 153088):
            for q in (1, 17, min(n, 2048)):
                with self.subTest(sequence=n, query=q):
                    plan = slabs(n, q)
                    self.assertEqual((plan[0].start, plan[-1].end), (0, n))
                    self.assertEqual(sum(s.end-s.start for s in plan), n)
                    self.assertTrue(all(0 < s.end-s.start <= SLAB_TOKENS for s in plan))
                    self.assertTrue(all(a.end == b.start for a, b in zip(plan, plan[1:])))
                    self.assertEqual([s.causal for s in plan], [False]*(len(plan)-1)+[True])
                    self.assertTrue(all(s.end <= n-q+1 for s in plan[:-1]))

    def test_slab_mask_matches_complete_bottom_right_causal_mask(self):
        for n, q in ((17, 17), (16385, 17), (32769, 2048), (153088, 2048)):
            for row in (0, q//2, q-1):
                for s in slabs(n, q):
                    for key in {s.start, s.end-1, max(s.start, min(s.end-1, n-q+row))}:
                        local_visible = not s.causal or key-s.start <= (s.end-s.start)-q+row
                        self.assertEqual(local_visible, key <= n-q+row)

    def test_bad_query_lengths_are_rejected(self):
        for n, q in ((0, 0), (10, 11), (4096, 2049), (100, -1)):
            with self.assertRaises(ValueError):
                slabs(n, q)

    def test_workspace_does_not_scale_with_context(self):
        sizes = workspace_bytes()
        self.assertEqual(sizes['kv'], 64*2**20)
        self.assertEqual(sum(sizes.values()), 142802952)


if __name__ == '__main__':
    unittest.main()
