import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import available_memory as memory
import lifecycle

spec = importlib.util.spec_from_file_location('head_patch', ROOT / 'patches/patch_startup_head.py')
head = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head)


class ReleaseTests(unittest.TestCase):
    def test_budget_reserves_runtime_and_desktop(self):
        g = memory.GIB
        fraction = memory.memory_fraction(31 * g, 32 * g)
        self.assertGreaterEqual(fraction * 32 * g, 27.35 * g)
        self.assertGreaterEqual(31 * g - fraction * 32 * g - 2.2 * g, g)

    def test_floor_never_steals_desktop_reserve(self):
        with self.assertRaises(ValueError):
            memory.memory_fraction(30 * memory.GIB, 32 * memory.GIB)
        for kwargs in ({'reserve': .49 * memory.GIB}, {'cap': float('nan')},
                       {'overhead': float('inf')}, {'reserve': -1}):
            with self.assertRaises(ValueError):
                memory.memory_fraction(31 * memory.GIB, 32 * memory.GIB, **kwargs)

    def test_head_patch_idempotent_and_unknown_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'head.py'
            path.write_text('if w is None or ' + head.OLD + ':\n    pass\n')
            head.patch(path)
            once = path.read_bytes()
            head.patch(path)
            self.assertEqual(path.read_bytes(), once)
            compile(once, str(path), 'exec')
            path.write_text('unknown source\n')
            with self.assertRaises(RuntimeError):
                head.patch(path)
            self.assertEqual(path.read_text(), 'unknown source\n')

    def test_production_and_invalid_name_rejected(self):
        for name in ('qwen38-nvfp4-wsl', '../escape', 'exp-name;true', 'other'):
            with self.assertRaises(ValueError):
                lifecycle.validate(name, 8180)
        lifecycle.validate('qwen38-dflash', 8080)

    def test_unowned_container_cannot_be_adopted(self):
        with patch.object(lifecycle, 'docker', return_value=(0, json.dumps([
                {'Name': '/exp-test', 'Config': {'Labels': {}}}]))):
            with self.assertRaises(RuntimeError):
                lifecycle.inspect('exp-test')

    def test_secret_and_boot_evidence_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(lifecycle, 'inspect', return_value=None):
            root = Path(tmp)
            first = Path(lifecycle.prepare(root, 'exp-test', 8180))
            before = first.read_bytes()
            second = Path(lifecycle.prepare(root, 'exp-test', 8180))
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(first.stat().st_mode & 0o777, 0o600)

    def test_wrong_container_identity_never_signalled(self):
        with patch.object(lifecycle, 'inspect', return_value={'Id': 'a' * 64}), \
                patch.object(lifecycle, 'active', return_value=(Path('.'),
                     {'name': 'exp-test', 'container_id': 'b' * 64})), \
                patch.object(lifecycle, 'stop_experiment') as stop:
            with self.assertRaises(RuntimeError):
                lifecycle.stop(Path('.'), 'exp-test')
            stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
