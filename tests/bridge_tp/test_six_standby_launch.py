# SPDX-License-Identifier: Apache-2.0
"""Launcher order for engine reuse "standby": standby forms first, asleep."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.bridge_tp import launch_pressure_balanced_pilot as launcher
from tools.bridge_tp import migration_receivers
from tools.bridge_tp import six_conversion as six
from tools.bridge_tp.pressure_cpu import load

d = load("pause_diagnostics")


class StandbyLaunchTests(unittest.TestCase):
    def test_standby_starts_first_and_channels_open_after_warm_up(self):
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[2]
        ) as directory:
            root = Path(directory)
            arrivals = root / "arrivals.json"
            arrivals.write_text(json.dumps([dict(request={})]))
            standby = [
                dict(instance_id="B-g7-tp1", endpoint="http://b7", tp=1),
                dict(instance_id="A-g9-tp2", endpoint="http://a9", tp=2),
            ]
            plan = dict(
                run_dir=root / "scene/controller",
                config_path=root / "scene/config.json",
                config=dict(source_url="source", target_url="target", target_tp_size=2),
                source_argv=["source"],
                target_argv=["target"],
                source_env={},
                target_env={},
                stager_argv=["stager"],
                controller_argv=["python", "controller"],
                standby_engines=standby,
            )
            common = Mock()

            def write(path, value):
                path.write_text(json.dumps(value))

            common.write_json.side_effect = write
            order = []

            def start(name, argv, env, log):
                order.append(("start", name))
                child = SimpleNamespace(
                    name=name, process=Mock(pid=100 + len(order), returncode=0)
                )
                child.process.poll.return_value = 0
                return child

            def cycle(endpoints, actions):
                order.append(("cycle", tuple(endpoints), actions))
                return {e: [] for e in endpoints}

            common.start_process.side_effect = start
            common.wait_healthy.side_effect = lambda url, process, timeout: order.append(
                ("healthy", url)
            )
            common.process_group_alive.return_value = False
            fake_runtime = SimpleNamespace(RuntimeControl=Mock())
            fake_warmup = SimpleNamespace(warmup_prompt_shapes=Mock())
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "tools.bridge_tp.run_phase9_cap0_noop": fake_warmup,
                        "tools.bridge_tp.run_phase9_controller": fake_runtime,
                    },
                ),
                patch.object(launcher.controller, "parse_args"),
                patch.object(launcher.controller, "preflight"),
                patch.object(launcher, "scene_result", return_value="done"),
                patch.object(launcher, "extra_engine", return_value=(["x"], {})),
                patch.object(launcher, "sleep_cycle", side_effect=cycle),
                patch.object(
                    six,
                    "publish_initial_channels",
                    side_effect=lambda p: order.append(("publish",)),
                ),
                patch.object(
                    migration_receivers,
                    "warm_channels",
                    side_effect=lambda p, r, t: order.append(("warm", t)),
                ),
                patch(
                    "tools.bridge_tp.monitor_pressure_host.validate_journal",
                    return_value={},
                ),
                patch.object(d, "verify", return_value={}),
            ):
                result = launcher.run_scene(
                    dict(startup_timeout_s=10, run_timeout_s=10),
                    dict(name="test", arrivals=str(arrivals)),
                    plan,
                    common,
                    None,
                    diagnose_pauses=True,
                )
            self.assertEqual(result, "done")
            starts = [x[1] for x in order if x[0] == "start" and x[1] != "host monitor"]
            # Both standby forms start before any awake engine.
            self.assertEqual(starts[:2], ["B-g7-tp1", "A-g9-tp2"])
            index = {}
            for i, x in enumerate(order):
                index.setdefault(x, i)  # first occurrence
            # Health is checked again after the awake forms wake.
            self.assertGreater(
                max(i for i, x in enumerate(order) if x == ("healthy", "source")),
                order.index(("cycle", ("target", "source"), ("sleep", "wake"))),
            )
            first = index[("cycle", ("http://b7", "http://a9"), ("sleep", "wake", "sleep"))]
            awake = index[("cycle", ("target", "source"), ("sleep", "wake"))]
            self.assertLess(index[("healthy", "http://a9")], first)
            self.assertLess(first, index[("start", "target TP2")])
            self.assertLess(index[("healthy", "source")], awake)
            self.assertLess(awake, index[("publish",)])
            self.assertLess(index[("publish",)], index[("warm", 180)])
            self.assertLess(index[("warm", 180)], index[("start", "pressure controller")])
            prewarm = json.loads((root / "scene/standby_prewarm.json").read_text())
            self.assertEqual(set(prewarm["standby"]), {"http://b7", "http://a9"})
            self.assertEqual(set(prewarm["awake"]), {"target", "source"})
            self.assertIn("channels_warm_unix_s", prewarm)


if __name__ == "__main__":
    unittest.main()
