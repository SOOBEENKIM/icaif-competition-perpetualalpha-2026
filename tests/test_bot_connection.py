"""Credential handling and no-order safeguards; uses only a public test key."""

import contextlib
import io
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eth_account import Account

from bot import connection, setup


# Public scalar-one test vector, unrelated to the team's real API wallet.
TEST_KEY = "0" * 63 + "1"
TEST_SIGNER = Account.from_key(TEST_KEY)


class BotConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.key_dir = Path(self.temp.name) / ".secrets"
        self.key_file = self.key_dir / "api_wallet.key"
        self.settings = {
            "network": "testnet", "dex": "par", "mode": "read_only",
            "team_wallet": "0x" + "2" * 40,
            "api_wallet": TEST_SIGNER.address.lower(),
        }
        for target, attribute, value in (
            (connection, "KEY_DIR", self.key_dir),
            (connection, "KEY_FILE", self.key_file),
            (setup, "KEY_FILE", self.key_file),
        ):
            mock = patch.object(target, attribute, value)
            mock.start()
            self.addCleanup(mock.stop)

    def test_key_can_be_reloaded_with_owner_only_permissions(self):
        connection.save_key(TEST_SIGNER)
        self.assertEqual(stat.S_IMODE(self.key_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.key_file.stat().st_mode), 0o600)
        self.assertEqual(connection.load_signer(self.settings).address, TEST_SIGNER.address)

    def test_existing_key_is_never_overwritten(self):
        connection.save_key(TEST_SIGNER)
        with self.assertRaises(FileExistsError):
            connection.save_key(Account.from_key("0" * 63 + "2"))
        self.assertEqual(connection.load_signer(self.settings).address, TEST_SIGNER.address)

    def test_permissive_and_symlink_key_files_are_rejected(self):
        connection.save_key(TEST_SIGNER)
        self.key_file.chmod(0o644)
        with self.assertRaises(connection.SetupError):
            connection.load_signer(self.settings)
        self.key_file.unlink()
        self.key_file.symlink_to(Path(self.temp.name) / "unread-target")
        with self.assertRaises(OSError):
            connection.load_signer(self.settings)

    def test_wrong_wallet_key_is_rejected_without_echoing_key(self):
        settings = dict(self.settings, api_wallet="0x" + "3" * 40)
        with self.assertRaises(connection.SetupError) as captured:
            connection.signer_from_key(TEST_KEY, settings)
        self.assertNotIn(TEST_KEY, str(captured.exception))
        self.assertFalse(self.key_file.exists())

    def test_exchange_submissions_are_blocked_before_network_access(self):
        client = object.__new__(connection.ReadOnlyExchange)
        with patch("hyperliquid.api.API.post") as request:
            with self.assertRaises(connection.SetupError):
                client.post("/exchange", {"action": {"type": "order"}})
            request.assert_not_called()

    def test_setup_reruns_without_prompt_and_redacts_connection_errors(self):
        connection.save_key(TEST_SIGNER)
        output = io.StringIO()
        with patch.object(setup, "load_settings", return_value=self.settings), \
             patch.object(setup.getpass, "getpass") as prompt, \
             patch.object(setup, "build_client", side_effect=ValueError(TEST_KEY)), \
             contextlib.redirect_stdout(output):
            self.assertEqual(setup.main(), 1)
            prompt.assert_not_called()
        self.assertNotIn(TEST_KEY, output.getvalue())
        self.assertEqual(connection.load_signer(self.settings).address, TEST_SIGNER.address)

    def test_non_terminal_setup_never_reads_or_saves_key(self):
        with patch.object(setup, "load_settings", return_value=self.settings), \
             patch.object(setup.sys.stdin, "isatty", return_value=False), \
             patch.object(setup.getpass, "getpass") as prompt, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.main(), 1)
            prompt.assert_not_called()
        self.assertFalse(os.path.lexists(self.key_file))


if __name__ == "__main__":
    unittest.main()
