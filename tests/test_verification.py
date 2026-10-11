from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_portfolio as verify


class VerificationTests(unittest.TestCase):
    def test_git_errors_fail_the_size_check(self):
        for error in (FileNotFoundError('git'), subprocess.CalledProcessError(128, ['git'])):
            with self.subTest(error=type(error).__name__):
                with patch.object(verify.subprocess, 'run', side_effect=error):
                    self.assertFalse(verify.check_tracked_file_sizes().passed)

    def test_unicode_path_is_checked_against_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
            file = root / '한글 자료.txt'
            file.write_bytes(b'0123456789')
            subprocess.run(['git', 'add', '--', file.name], cwd=root, check=True)
            with patch.object(verify, 'ROOT', root), patch.object(verify, 'MAX_TRACKED_FILE_BYTES', 8):
                result = verify.check_tracked_file_sizes()
                self.assertFalse(result.passed)
                self.assertIn(file.name, result.detail)
                file.write_bytes(b'ok')
                self.assertTrue(verify.check_tracked_file_sizes().passed)
