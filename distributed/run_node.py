import uvicorn

from .config import Settings, _parse_host_port
from .node0 import build_app as build_node0
from .node1 import build_app as build_node1
from .node2 import build_app as build_node2


def main() -> None:
    settings = Settings.from_env()
    builders = {0: build_node0, 1: build_node1, 2: build_node2}
    if settings.node_number not in builders:
        raise ValueError(f"Unsupported NODE_NUMBER={settings.node_number}")
    app = builders[settings.node_number](settings)
    
    # Get the IP for this node and parse host/port
    node_ips = [settings.node_0_ip, settings.node_1_ip, settings.node_2_ip]
    host, port = _parse_host_port(node_ips[settings.node_number])
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
