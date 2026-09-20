"""CPU-only checks for relocatable paths; no model imports or downloads."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectPathsTest(unittest.TestCase):
    def read_paths(self, cwd, **overrides):
        module = ROOT / 'src/project_paths.py'
        self.assertTrue(module.is_file(), 'shared path configuration is missing')
        env = dict(os.environ)
        for key in ('FLASHHEAD_ROOT', 'MINIMIND_REPO'):
            env.pop(key, None)
        env.update(overrides)
        command = (
            'import json, runpy, sys; '
            'p = runpy.run_path(sys.argv[1]); '
            'print(json.dumps({k: str(p[k]) for k in '
            '("PROJECT_ROOT", "FLASHHEAD_ROOT", "MINIMIND_REPO")}))'
        )
        result = subprocess.run([sys.executable, '-c', command, str(module)],
                                cwd=cwd, env=env, text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_defaults_do_not_depend_on_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.read_paths(directory)
        self.assertEqual(paths['PROJECT_ROOT'], str(ROOT))
        self.assertEqual(paths['FLASHHEAD_ROOT'], str(ROOT.parent / 'SoulX-FlashHead'))
        self.assertEqual(paths['MINIMIND_REPO'], str(ROOT.parent / 'minimind-o'))

    def test_relative_and_tilde_overrides_are_absolute(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.read_paths(directory, FLASHHEAD_ROOT='external models/flashhead',
                                    MINIMIND_REPO='~/models/minimind-o')
            self.assertEqual(paths['FLASHHEAD_ROOT'],
                             str(Path(directory) / 'external models/flashhead'))
            self.assertEqual(paths['MINIMIND_REPO'],
                             str(Path.home() / 'models/minimind-o'))
            self.assertEqual(paths['PROJECT_ROOT'], str(ROOT))

    def test_empty_overrides_use_defaults(self):
        paths = self.read_paths(ROOT, FLASHHEAD_ROOT='', MINIMIND_REPO='')
        self.assertEqual(paths['FLASHHEAD_ROOT'], str(ROOT.parent / 'SoulX-FlashHead'))
        self.assertEqual(paths['MINIMIND_REPO'], str(ROOT.parent / 'minimind-o'))

    def test_asr_help_does_not_load_models_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / 'src/asr_transcribe.py'), '--help'],
                cwd=directory, text=True, capture_output=True, check=True)
        self.assertIn('--model_path', result.stdout)

    def test_shell_scripts_find_their_relocated_project_without_models(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / 'relocated project'
            (checkout / 'scripts').mkdir(parents=True)
            (checkout / 'data/raw_videos').mkdir(parents=True)
            env = dict(os.environ, N_SENT='0', N_VOICE='0')
            for name in ('batch_tts_features.sh', 'preprocess_videos.sh'):
                script = checkout / 'scripts' / name
                shutil.copyfile(ROOT / 'scripts' / name, script)
                result = subprocess.run(['bash', str(script)], cwd=directory,
                                        env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((checkout / 'data/tts_batch').is_dir())
            self.assertTrue((checkout / 'data/audio_16k').is_dir())
            self.assertFalse((Path(directory) / 'data').exists())


if __name__ == '__main__':
    unittest.main()
