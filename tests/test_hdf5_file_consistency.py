import pytest
import h5py
import numpy as np
from os.path import abspath, dirname, join

PARENT_DIR = dirname(dirname(abspath(__file__)))


def rounded_equal_with_nans(array_a, array_b):
    a = np.nan_to_num(array_a)
    b = np.nan_to_num(array_b)
    return np.array_equal(np.round(a, 6), np.round(b, 6))


# We've chosen Crusoe as our test dataset. As a sanity check
# after refactoring you can take the current Region_Climate_Crusoe.hdf5
# file as a baseline, and rerun the weather extract on Crusoe_Data.csv.
# This will generate a new Region_Climate_Crusoe.hdf5 which you can rerun
# this test to check that we haven't broke extraction.
# IMPORTANT: RENAME Region_Climate_Crusoe.hdf5 before re-running the extract
# so that you don't overwrite it.
@pytest.mark.skip(
    reason="Only run when comparing hd5f outputs after a significant refactor")
def test_file_consistency():
    cru_latest = h5py.File(
        join(PARENT_DIR, "Climate_Data", "Region_Climate_Crusoe_203d59bd.hdf5"), "r")
    cru_orig = h5py.File(
        join(PARENT_DIR, "Climate_Data", "Region_Climate_Crusoe_d442a20c.hdf5"), "r")

    for k in cru_orig.keys():
        latest = cru_latest[k]
        orig = cru_orig[k]
        assert rounded_equal_with_nans(orig, latest)
    cru_latest.close()
    cru_orig.close()
