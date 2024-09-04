import io
import asyncio
import docker
import tempfile
import os
from typing import Dict
import modal



async def execute_code_modal(
    script_to_run_by_name: str,
    scripts_by_name: Dict[str, str],
    dir_name: str,
    python_version="python:3.9-slim",
    packages=None,
) -> str:
    print("Running code in modal")
    try:
        with modal.NetworkFileSystem.ephemeral() as nfs:
            for name, script in scripts_by_name.items():
                await nfs.write_file.aio(name, io.BytesIO(script.encode()))

            install_command = ""
            if packages:
                install_command = f"pip install {' '.join(packages)} && "

            sb = modal.Sandbox.create(
                "bash",
                "-c",
                f"{install_command}cd /vol && python -W ignore {script_to_run_by_name}",
                image=python_version,
                timeout=60,
                cloud="aws",
                network_file_systems={"/vol": nfs},
            )
            await sb.wait.aio()
            stdout = await sb.stdout.read.aio()
            stderr = await sb.stderr.read.aio()
            return stdout
    except modal.SandboxTimeoutError:
        return "Execution timed out after 60 seconds"
