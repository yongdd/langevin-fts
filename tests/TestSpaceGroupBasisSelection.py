#!/usr/bin/env python3
"""Basic regression test for SpaceGroup physical basis selection."""

from polymerfts import _core


def expect_error(fn, contains):
    try:
        fn()
    except Exception as exc:
        msg = str(exc)
        assert contains in msg, f"Expected '{contains}' in error, got: {msg}"
        return
    raise AssertionError("Expected error, but function succeeded")


def main():
    nx = [8, 8, 8]

    # Pmmm: should allow Pmmm basis + z-mirror; M3 should fail
    hall_nums = _core.SpaceGroup.hall_numbers_from_symbol("Pmmm")
    assert len(hall_nums) > 0, "Pmmm hall numbers not found"
    sg_pmmm = _core.SpaceGroup(nx, hall_nums[0])

    sg_pmmm.enable_pmmm_physical_basis()
    assert sg_pmmm.using_pmmm_physical_basis(), "Pmmm basis not enabled"
    assert sg_pmmm.get_n_reduced_basis() == (nx[0] // 2) * (nx[1] // 2) * (nx[2] // 2)

    sg_pmmm.enable_z_mirror_physical_basis()
    assert sg_pmmm.using_z_mirror_physical_basis(), "Z-mirror basis not enabled"
    assert sg_pmmm.get_n_reduced_basis() == nx[0] * nx[1] * (nx[2] // 2)

    # Im-3m: should allow M3 basis
    sg_m3 = _core.SpaceGroup(nx, "Im-3m", 529)
    sg_m3.enable_m3_physical_basis()
    assert sg_m3.using_m3_physical_basis(), "M3 basis not enabled"
    assert sg_m3.get_n_reduced_basis() == (nx[0] // 2) * (nx[1] // 2) * (nx[2] // 2)

    # I4_132: no 3m translations
    sg_no_m3 = _core.SpaceGroup(nx, "I4_132", 510)
    expect_error(lambda: sg_no_m3.enable_m3_physical_basis(), "3m")

    print("TestSpaceGroupBasisSelection: PASSED")


if __name__ == "__main__":
    main()
