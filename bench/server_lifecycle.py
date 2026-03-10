import socket


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def ensure_port_available(port: int, service_name: str) -> None:
    if is_port_in_use(port):
        raise RuntimeError(
            f"Port {port} is already in use. Stop the existing {service_name} server or choose a different --port before launching a new benchmark."
        )
