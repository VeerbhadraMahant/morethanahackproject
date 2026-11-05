import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synx.data import load_tracking  # noqa: E402

# First Team in the XML (T2) is home, so slots 0 and 2 are home (0) and slot 1 is away (1).
META = """<?xml version="1.0"?><Metadata>
<Teams><Team id="T2"><Name>H</Name></Team><Team id="T1"><Name>A</Name></Team></Teams>
<Players><Player id="a" teamId="T2"/><Player id="b" teamId="T1"/><Player id="c" teamId="T2"/></Players>
</Metadata>"""
RAW = "\n".join([
    "1:0.5,0.5,1.0;NaN;0.1,0.2,3.0:1.0,1.0,9.0",
    "2:NaN;NaN;NaN:NaN",
    "this is not a frame",
    "5:0.0,1.0,0.0;0.25,0.5,2.0;NaN:0.5,0.5,4.0",
    "6:0.5,0.5,1.0:0.5,0.5,1.0",  # wrong slot count
]) + "\n"


class LoadTrackingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.meta, self.raw = os.path.join(d, "m.xml"), os.path.join(d, "r.txt")
        with open(self.meta, "w") as f:
            f.write(META)
        with open(self.raw, "w") as f:
            f.write(RAW)
        self.kw = dict(paths=(self.meta, self.raw), n_frames=8, cache_dir=os.path.join(d, "c"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse(self):
        t = load_tracking(**self.kw)
        self.assertEqual(t["players"].shape, (8, 3, 2))
        self.assertEqual(t["player_speed"].shape, (8, 3))
        self.assertEqual(t["ball"].shape, (8, 2))
        self.assertEqual(t["n_malformed"], 2)
        self.assertEqual(list(t["team"]), [0, 1, 0])
        np.testing.assert_allclose(t["pitch"], (105.3, 68.0), rtol=1e-6)
        np.testing.assert_allclose(t["ball"][0], (105.3, 68.0), rtol=1e-5)  # frame_id 1 -> row 0
        np.testing.assert_allclose(t["players"][0, 0], (52.65, 34.0), rtol=1e-5)
        self.assertEqual(t["ball_speed"][0], 9.0)
        self.assertEqual(t["player_speed"][4, 1], 2.0)
        np.testing.assert_allclose(t["players"][4, 1], (26.325, 34.0), rtol=1e-5)
        self.assertTrue(np.isnan(t["players"][0, 1]).all())  # NaN player
        self.assertTrue(np.isnan(t["ball"][1]).all())        # NaN ball
        self.assertTrue(np.isnan(t["players"][2:4]).all())   # gap in frame_ids stays NaN
        self.assertTrue(np.isnan(t["ball"][5:]).all())       # malformed / absent frames

    def test_cache_roundtrip_and_invalidation(self):
        a = load_tracking(**self.kw)
        self.assertTrue(os.path.isfile(os.path.join(self.kw["cache_dir"], "BAR-ATH_tracking.npz")))
        b = load_tracking(**self.kw)
        np.testing.assert_array_equal(a["players"], b["players"])
        self.assertEqual(a["n_malformed"], b["n_malformed"])
        # same size and mtime -> still served from the cache
        st = os.stat(self.raw)
        with open(self.raw, "w") as f:
            f.write(RAW.replace("0.25,0.5", "0.75,0.5"))
        os.utime(self.raw, ns=(st.st_atime_ns, st.st_mtime_ns))
        c = load_tracking(**self.kw)
        np.testing.assert_allclose(c["players"][4, 1, 0], 26.325, rtol=1e-5)
        # changed size/mtime -> reparsed
        with open(self.raw, "a") as f:
            f.write("7:NaN;NaN;NaN:NaN\n")
        d = load_tracking(**self.kw)
        np.testing.assert_allclose(d["players"][4, 1, 0], 78.975, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
