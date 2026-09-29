#!/usr/bin/env bash
# Verify separated prefill/decode evidence without requiring a HIGH action.

BRIDGETP_M2_REQUIRE_LOW_TO_HIGH=0 \
BRIDGETP_M2_LOW_GIB_S="${BRIDGETP_M2_LOW_GIB_S:-0.5}" \
bash "$(dirname "$0")/run_m2_low_to_high_a100_smoke.sh"
