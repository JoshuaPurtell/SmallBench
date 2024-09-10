import io
import asyncio
import docker
import tempfile
import os
from typing import Dict, Optional, Tuple


def execute_code_docker(
    script_to_run_by_name: str,
    scripts_by_name: Dict[str, str],
    dir_name: str,
    python_version="python:3.9-slim",
    packages=None,
    container: Optional[docker.models.containers.Container] = None,
) -> Tuple[str, docker.models.containers.Container]:
    assert (
        script_to_run_by_name in scripts_by_name
    ), "script_to_run_by_name must be in scripts_by_name"
    client = docker.from_env()

    with tempfile.TemporaryDirectory() as temp_dir:
        paths = [os.path.join(temp_dir, name) for name in scripts_by_name.keys()]
        for path, script in zip(paths, scripts_by_name.values()):
            with open(path, "w") as f:
                f.write(script)

        install_command = ""
        if packages:
            install_command = f"pip install {' '.join(packages)} && "

        if container is None:
            container = client.containers.run(
                python_version,
                command=f"/bin/bash -c '{install_command}python -W ignore /app/{script_to_run_by_name}'",
                volumes={temp_dir: {"bind": "/app", "mode": "ro"}},
                detach=True,
            )
        else:
            container.exec_run(
                f"/bin/bash -c '{install_command}python -W ignore /app/{script_to_run_by_name}'"
            )

        try:
            container.wait(timeout=60)
            logs = container.logs().decode("utf-8")
        finally:
            #pass
            container.remove()
    return logs, container
