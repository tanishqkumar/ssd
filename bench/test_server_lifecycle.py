import unittest
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bench.server_lifecycle import ensure_port_available, is_port_in_use


class ServerLifecycleTests(unittest.TestCase):
    @patch("bench.server_lifecycle.socket.socket")
    def test_is_port_in_use_true_when_connect_succeeds(self, socket_cls):
        sock = socket_cls.return_value.__enter__.return_value
        sock.connect_ex.return_value = 0

        self.assertTrue(is_port_in_use(40020))

    @patch("bench.server_lifecycle.socket.socket")
    def test_is_port_in_use_false_when_connect_fails(self, socket_cls):
        sock = socket_cls.return_value.__enter__.return_value
        sock.connect_ex.return_value = 111

        self.assertFalse(is_port_in_use(40020))

    @patch("bench.server_lifecycle.is_port_in_use", return_value=True)
    def test_ensure_port_available_raises_for_busy_port(self, _port_check):
        with self.assertRaises(RuntimeError) as error:
            ensure_port_available(40020, "vLLM")

        self.assertIn("Port 40020 is already in use", str(error.exception))
        self.assertIn("different --port", str(error.exception))


if __name__ == "__main__":
    unittest.main()
