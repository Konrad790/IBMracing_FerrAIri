from __future__ import annotations

from torcs_jm_par import Client

from research_driver import BEST_CONFIG_PATH, TunableDriver, load_driver_setup


if __name__ == "__main__":
    setup = load_driver_setup(BEST_CONFIG_PATH)
    driver = TunableDriver(setup)
    client = Client(p=3001)
    for _step in range(client.maxSteps, 0, -1):
        client.get_servers_input()
        driver.drive_client(client)
        client.respond_to_server()
    client.shutdown()
