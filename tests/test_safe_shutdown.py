import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from safe_shutdown import stop_experiment

CID = 'd' * 64


class ShutdownTest(unittest.TestCase):
    def run_case(self, mode):
        commands = []
        now = [0]
        signalled = [False]

        def sleep(seconds):
            now[0] += seconds

        def docker(*args, **kwargs):
            commands.append(args)
            if args[0] == 'inspect':
                return 0, json.dumps({'Name': '/exp-test', 'Id': CID, 'State': {
                    'Running': not signalled[0] or mode == 'stalled',
                    'ExitCode': 137 if mode == 'exit137' else 0}})
            if args[0] == 'kill':
                signalled[0] = True
            if args[0] == 'logs':
                return 0, 'force killing remaining process EngineCore' if signalled[0] and mode == 'forced' else 'clean'
            return 0, ''

        def barrier():
            commands.append(('barrier',))
            if mode == 'busy':
                raise RuntimeError('GPU_RECOVERY_REQUIRED: scheduler/device barrier incomplete')

        args = dict(timeout=2, clock=lambda: now[0], sleep=sleep, barrier=barrier, container_id=CID)
        if mode == 'clean':
            stop_experiment('exp-test', docker, **args)
        else:
            with self.assertRaisesRegex(RuntimeError, 'GPU_RECOVERY_REQUIRED'):
                stop_experiment('exp-test', docker, **args)
        return commands

    def test_busy_server_is_never_signalled(self):
        commands = self.run_case('busy')
        self.assertFalse(any(c[0] in ('kill', 'stop', 'rm') for c in commands))

    def test_stalled_or_forced_worker_preserved_even_on_exit_zero(self):
        for mode in ('stalled', 'forced', 'exit137'):
            commands = self.run_case(mode)
            self.assertIn(('kill', '--signal=SIGTERM', CID), commands)
            self.assertFalse(any(c[0] in ('stop', 'rm') for c in commands))

    def test_clean_shutdown_removes_without_force(self):
        commands = self.run_case('clean')
        self.assertEqual(commands[-1], ('rm', CID))
        self.assertLess(commands.index(('barrier',)), commands.index(('kill', '--signal=SIGTERM', CID)))
        self.assertFalse(any('-f' in c and c[0] == 'rm' for c in commands))

    def test_production_rejected(self):
        with self.assertRaises(ValueError):
            stop_experiment('qwen38-nvfp4-wsl', lambda *_: self.fail('must not call docker'))

    def test_identity_mismatch_or_missing_barrier_never_signals(self):
        for name in ('/exp-test', '/qwen38-nvfp4-wsl'):
            def docker(*args, **kwargs):
                self.assertEqual(args[0], 'inspect')
                return 0, json.dumps({'Name': name, 'Id': CID, 'State': {'Running': True}})
            with self.assertRaisesRegex(RuntimeError, 'GPU_RECOVERY_REQUIRED'):
                stop_experiment('exp-test', docker, container_id=CID)


if __name__ == '__main__':
    unittest.main()
