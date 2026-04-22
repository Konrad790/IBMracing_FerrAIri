from __future__ import annotations

import socket
import time

from torcs_jm_par import Client, data_size


class RLTorcsClient(Client):
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 3001,
        sid: str = "SCR",
        steps: int = 100000,
        connect_attempts: int = 20,
        connect_delay: float = 0.5,
    ) -> None:
        self._connect_attempts = max(1, connect_attempts)
        self._connect_delay = max(0.1, connect_delay)
        super().__init__(H=host, p=port, i=sid)
        self.maxSteps = steps

    def parse_the_command_line(self) -> None:
        return

    def setup_connection(self) -> None:
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as exc:
            raise RuntimeError("Could not create UDP socket for TORCS.") from exc

        self.so.settimeout(1.0)
        init_angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
        initmsg = f"{self.sid}(init {init_angles})"

        attempts_left = self._connect_attempts
        while attempts_left > 0:
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
                sockdata, _addr = self.so.recvfrom(data_size)
                decoded = sockdata.decode("utf-8")
            except OSError:
                attempts_left -= 1
                print(
                    f"Waiting for TORCS on UDP {self.port}... attempts left: {attempts_left}"
                )
                time.sleep(self._connect_delay)
                continue

            if "identified" in decoded:
                print(f"Client connected on {self.port}.")
                return

        self.so.close()
        self.so = None
        raise RuntimeError(
            "Could not connect to TORCS. Start `wtorcs.exe` first in `torcs/torcs`, "
            "then run the RL script from a second terminal in `torcs/gym_torcs`."
        )


def request_race_restart(client: RLTorcsClient | None, pause_seconds: float) -> None:
    if client is None or client.so is None:
        return

    client.R.d["meta"] = 1
    try:
        client.respond_to_server()
    except SystemExit:
        pass
    except OSError:
        pass

    time.sleep(max(0.0, pause_seconds))
