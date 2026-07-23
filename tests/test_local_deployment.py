"""Regression tests for the single-host small/medium deployment contract."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values

from gateway.multiplexer import _positive_int_env


ROOT = Path(__file__).resolve().parents[1]


class LocalProfilesTest(unittest.TestCase):
    def _profile(self, name: str) -> dict[str, str]:
        return {
            key: value
            for key, value in dotenv_values(ROOT / "config" / "profiles" / f"{name}.env").items()
            if value is not None
        }

    def test_small_profile_is_bounded_and_uses_local_dispatcher(self):
        profile = self._profile("small")
        self.assertEqual(profile["LOCAL_PROFILE"], "small")
        self.assertEqual(profile["RUNTIME_DISPATCHER"], "local")
        self.assertEqual(profile["S2S_MODE"], "shadow")
        self.assertLessEqual(int(profile["VOICE_MAX_SESSIONS"]), 2)
        self.assertLessEqual(int(profile["VLLM_MAX_NUM_SEQS"]), 4)
        self.assertLessEqual(int(profile["VLLM_MAX_MODEL_LEN"]), 4096)

    def test_medium_profile_has_more_capacity_but_keeps_shadow_safe(self):
        small = self._profile("small")
        medium = self._profile("medium")
        self.assertEqual(medium["LOCAL_PROFILE"], "medium")
        self.assertEqual(medium["RUNTIME_DISPATCHER"], "local")
        self.assertEqual(medium["S2S_MODE"], "shadow")
        self.assertGreater(
            int(medium["VOICE_MAX_SESSIONS"]), int(small["VOICE_MAX_SESSIONS"])
        )
        self.assertGreater(
            int(medium["GATEWAY_GLOBAL_MAX_CONCURRENT"]),
            int(small["GATEWAY_GLOBAL_MAX_CONCURRENT"]),
        )

    def test_positive_int_env_rejects_invalid_or_non_positive_values(self):
        cases = (("abc", 7), ("0", 7), ("-3", 7), ("5", 5))
        for raw, expected in cases:
            with self.subTest(raw=raw), patch.dict(os.environ, {"TEST_LIMIT": raw}):
                self.assertEqual(_positive_int_env("TEST_LIMIT", 7), expected)


if __name__ == "__main__":
    unittest.main()
