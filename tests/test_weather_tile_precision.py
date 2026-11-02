"""Exact retained-weather crop checks across original float16/float32 encodings."""
from copy import deepcopy
import unittest

import numpy as np
from numcodecs import Blosc

from hydra_directflow.local_data import _verify_weather_crop, sha256

CYCLE = "20240606_00Z"


def fixture(dtype):
    tile = np.zeros((18, 150, 150), dtype=dtype)
    crop = np.linspace(0., 1., 957, dtype=np.float64).astype(dtype).reshape(33, 29)
    fill = -10000. if dtype == "<f2" else -9999.
    crop[0, 0], crop[0, 1] = fill, np.nan
    tile[0, 28:61, 14:43] = crop
    raw = bytes(Blosc(cname="lz4", clevel=5, shuffle=1).encode(tile))
    relative = "source_bytes/" + "a" * 64 + ".bin"
    source = {"relative_path": "cycles/" + CYCLE + "/" + relative, "sha256": sha256(raw), "bytes": len(raw),
              "url": "https://example.invalid/field/0.3.9"}
    fields = {"test": {"source": source, "spec": {"chunks": [18, 150, 150], "dtype": dtype,
                      "filters": None, "compressor": {"id": "blosc"}, "fill_value": fill}}}
    values = crop.reshape(1, 957).astype("<f8")
    present = ~(np.isnan(values) | (values == fill))
    records = [{"field_source_id": "test", "lead": 1}]
    return {relative: raw}, fields, records, values, present


class WeatherTilePrecisionTests(unittest.TestCase):
    def test_float16_and_float32_preserve_exact_source_values_masks_and_cell_order(self):
        for dtype in ("<f2", "<f4"):
            with self.subTest(dtype=dtype):
                args = fixture(dtype)
                original = args[3].copy()
                _verify_weather_crop(*args, cycle_id=CYCLE)
                np.testing.assert_equal(args[3], original)
                self.assertEqual(int((~args[4]).sum()), 2)

    def test_float32_cannot_be_silently_rounded_to_float16(self):
        members, fields, records, values, present = fixture("<f4")
        rounded = values.astype("<f2").astype("<f8")
        with self.assertRaisesRegex(ValueError, "values/cell order"):
            _verify_weather_crop(members, fields, records, rounded, present, cycle_id=CYCLE)

    def test_unsupported_dtype_is_rejected_before_decoding(self):
        for dtype in ("<i2", "<f8", ">f4"):
            args = fixture("<f4")
            args[1]["test"]["spec"]["dtype"] = dtype
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "encoding"):
                _verify_weather_crop(*args, cycle_id=CYCLE)

    def test_dtype_expansion_length_mismatch_is_rejected(self):
        args = fixture("<f4")
        args[1]["test"]["spec"]["dtype"] = "<f2"
        with self.assertRaisesRegex(ValueError, "expansion bound"):
            _verify_weather_crop(*args, cycle_id=CYCLE)

    def test_mask_or_crop_order_cannot_change_under_float32(self):
        members, fields, records, values, present = fixture("<f4")
        altered = present.copy(); altered[0, 0] = True
        with self.assertRaisesRegex(ValueError, "masks differ"):
            _verify_weather_crop(members, fields, records, values, altered, cycle_id=CYCLE)
        changed = values.copy(); changed[0, 10], changed[0, 11] = changed[0, 11], changed[0, 10]
        with self.assertRaisesRegex(ValueError, "cell order"):
            _verify_weather_crop(members, fields, records, changed, present, cycle_id=CYCLE)


if __name__ == "__main__":
    unittest.main()
