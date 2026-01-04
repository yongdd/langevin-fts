#!/usr/bin/env python
"""
Test validation utilities for SCFT and L-FTS simulations.

These tests verify that the parameter validation catches common
errors before expensive computations begin.
"""

import unittest
import sys
import os

# Add parent directory to path to import polymerfts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polymerfts.validation import (
    validate_scft_params,
    validate_lfts_params,
    validate_type,
    validate_required_keys,
    validate_positive,
    validate_list_length,
    ValidationError,
)


class TestValidationType(unittest.TestCase):
    """Test type validation function."""

    def test_valid_type(self):
        """Valid types should not raise."""
        validate_type(42, int, "test")
        validate_type(3.14, float, "test")
        validate_type("hello", str, "test")
        validate_type([1, 2], (list, tuple), "test")

    def test_invalid_type(self):
        """Invalid types should raise ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            validate_type("string", int, "my_param")
        self.assertIn("my_param", str(ctx.exception))
        self.assertIn("int", str(ctx.exception))


class TestValidateRequiredKeys(unittest.TestCase):
    """Test required key validation."""

    def test_all_keys_present(self):
        """Should not raise when all keys present."""
        params = {"a": 1, "b": 2, "c": 3}
        validate_required_keys(params, ["a", "b"])

    def test_missing_keys(self):
        """Should raise when keys are missing."""
        params = {"a": 1}
        with self.assertRaises(ValidationError) as ctx:
            validate_required_keys(params, ["a", "b", "c"])
        self.assertIn("b", str(ctx.exception))
        self.assertIn("c", str(ctx.exception))


class TestValidatePositive(unittest.TestCase):
    """Test positive value validation."""

    def test_positive_values(self):
        """Positive values should not raise."""
        validate_positive(1, "test")
        validate_positive(0.001, "test")
        validate_positive(1e10, "test")

    def test_zero_raises(self):
        """Zero should raise ValidationError."""
        with self.assertRaises(ValidationError):
            validate_positive(0, "test")

    def test_negative_raises(self):
        """Negative values should raise ValidationError."""
        with self.assertRaises(ValidationError):
            validate_positive(-1, "test")


class TestValidateListLength(unittest.TestCase):
    """Test list length validation."""

    def test_correct_length(self):
        """Correct length should not raise."""
        validate_list_length([1, 2, 3], 3, "test")

    def test_wrong_length(self):
        """Wrong length should raise ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            validate_list_length([1, 2], 3, "test")
        self.assertIn("3 elements", str(ctx.exception))


class TestValidateSCFTParams(unittest.TestCase):
    """Test SCFT parameter validation."""

    def get_valid_params(self):
        """Return a valid SCFT parameter dictionary."""
        return {
            "nx": [32, 32],
            "lx": [4.0, 4.0],
            "ds": 0.01,
            "segment_lengths": {"A": 1.0, "B": 1.0},
            "chi_n": {"A,B": 20.0},
            "distinct_polymers": [
                {
                    "volume_fraction": 1.0,
                    "blocks": [
                        {"type": "A", "length": 0.5},
                        {"type": "B", "length": 0.5},
                    ],
                }
            ],
        }

    def test_valid_params(self):
        """Valid parameters should not raise."""
        params = self.get_valid_params()
        validate_scft_params(params)

    def test_missing_nx(self):
        """Missing nx should raise."""
        params = self.get_valid_params()
        del params["nx"]
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("nx", str(ctx.exception))

    def test_missing_lx(self):
        """Missing lx should raise."""
        params = self.get_valid_params()
        del params["lx"]
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("lx", str(ctx.exception))

    def test_invalid_nx_dimension(self):
        """nx with wrong dimensions should raise."""
        params = self.get_valid_params()
        params["nx"] = [32, 32, 32, 32]  # 4D is invalid
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("1, 2, or 3", str(ctx.exception))

    def test_negative_nx(self):
        """Negative nx values should raise."""
        params = self.get_valid_params()
        params["nx"] = [32, -16]
        with self.assertRaises(ValidationError):
            validate_scft_params(params)

    def test_lx_dimension_mismatch(self):
        """lx with wrong number of elements should raise."""
        params = self.get_valid_params()
        params["lx"] = [4.0]  # Should match nx length
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("lx", str(ctx.exception))

    def test_negative_lx(self):
        """Negative lx values should raise."""
        params = self.get_valid_params()
        params["lx"] = [4.0, -4.0]
        with self.assertRaises(ValidationError):
            validate_scft_params(params)

    def test_ds_too_large(self):
        """ds > 1 should raise."""
        params = self.get_valid_params()
        params["ds"] = 2.0
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("ds", str(ctx.exception))

    def test_empty_segment_lengths(self):
        """Empty segment_lengths should raise."""
        params = self.get_valid_params()
        params["segment_lengths"] = {}
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("segment_lengths", str(ctx.exception))

    def test_negative_segment_length(self):
        """Negative segment length should raise."""
        params = self.get_valid_params()
        params["segment_lengths"]["A"] = -1.0
        with self.assertRaises(ValidationError):
            validate_scft_params(params)

    def test_empty_polymers(self):
        """Empty distinct_polymers should raise."""
        params = self.get_valid_params()
        params["distinct_polymers"] = []
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("distinct_polymers", str(ctx.exception))

    def test_volume_fraction_sum(self):
        """Volume fractions not summing to 1.0 should raise."""
        params = self.get_valid_params()
        params["distinct_polymers"][0]["volume_fraction"] = 0.5
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("volume fraction", str(ctx.exception).lower())

    def test_missing_blocks(self):
        """Polymer without blocks should raise."""
        params = self.get_valid_params()
        del params["distinct_polymers"][0]["blocks"]
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("blocks", str(ctx.exception))

    def test_invalid_chain_model(self):
        """Invalid chain_model should raise."""
        params = self.get_valid_params()
        params["chain_model"] = "invalid"
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("chain_model", str(ctx.exception))

    def test_invalid_platform(self):
        """Invalid platform should raise."""
        params = self.get_valid_params()
        params["platform"] = "invalid"
        with self.assertRaises(ValidationError) as ctx:
            validate_scft_params(params)
        self.assertIn("platform", str(ctx.exception))


class TestValidateLFTSParams(unittest.TestCase):
    """Test L-FTS parameter validation."""

    def get_valid_params(self):
        """Return a valid L-FTS parameter dictionary."""
        return {
            "nx": [32, 32],
            "lx": [4.0, 4.0],
            "ds": 0.01,
            "segment_lengths": {"A": 1.0, "B": 1.0},
            "chi_n": {"A,B": 20.0},
            "distinct_polymers": [
                {
                    "volume_fraction": 1.0,
                    "blocks": [
                        {"type": "A", "length": 0.5},
                        {"type": "B", "length": 0.5},
                    ],
                }
            ],
            "langevin": {
                "nbar": 1000,
                "dt": 0.01,
            },
        }

    def test_valid_params(self):
        """Valid parameters should not raise."""
        params = self.get_valid_params()
        validate_lfts_params(params)

    def test_missing_langevin(self):
        """Missing langevin dict should raise."""
        params = self.get_valid_params()
        del params["langevin"]
        with self.assertRaises(ValidationError) as ctx:
            validate_lfts_params(params)
        self.assertIn("langevin", str(ctx.exception))

    def test_missing_nbar(self):
        """Missing nbar should raise."""
        params = self.get_valid_params()
        del params["langevin"]["nbar"]
        with self.assertRaises(ValidationError) as ctx:
            validate_lfts_params(params)
        self.assertIn("nbar", str(ctx.exception))

    def test_missing_dt(self):
        """Missing dt should raise."""
        params = self.get_valid_params()
        del params["langevin"]["dt"]
        with self.assertRaises(ValidationError) as ctx:
            validate_lfts_params(params)
        self.assertIn("dt", str(ctx.exception))

    def test_negative_nbar(self):
        """Negative nbar should raise."""
        params = self.get_valid_params()
        params["langevin"]["nbar"] = -1
        with self.assertRaises(ValidationError):
            validate_lfts_params(params)

    def test_negative_dt(self):
        """Negative dt should raise."""
        params = self.get_valid_params()
        params["langevin"]["dt"] = -0.01
        with self.assertRaises(ValidationError):
            validate_lfts_params(params)


if __name__ == "__main__":
    unittest.main()
