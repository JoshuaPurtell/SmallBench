import io
from typing import Dict, Optional, Tuple

import modal


async def execute_code_modal_async(
    script_to_run_by_name: str,
    scripts_by_name: Dict[str, str],
    dir_name: str,
    python_version="3.9",
    packages=None,
    verbose=False,
) -> Tuple[str, Optional[str]]:
    result = "Modal execution failed"
    sterror = None
    try:
        with modal.NetworkFileSystem.ephemeral() as nfs:
            for name, script in scripts_by_name.items():
                await nfs.write_file.aio(name, io.BytesIO(script.encode()))
            image = modal.Image.debian_slim(python_version=python_version)
            if packages:
                image = image.pip_install("uv")
                package_install_command = " ".join(packages)
                image = image.run_commands(
                    f"uv pip install --system --compile-bytecode {package_install_command}"
                )
            sb = modal.Sandbox.create(
                "bash",
                "-c",
                f"cd /vol && python -W ignore {script_to_run_by_name}",
                image=image,
                timeout=120,
                cloud="aws",
                network_file_systems={"/vol": nfs},
            )
            await sb.wait.aio()
            stdout = await sb.stdout.read.aio()
            stderr = await sb.stderr.read.aio()
            result = stdout
            sterror = stderr
    except modal.exception.SandboxTimeoutError:
        result = "Execution timed out after 60 seconds"
        sterror = None
    return result, sterror


def execute_code_modal_sync(
    script_to_run_by_name: str,
    scripts_by_name: Dict[str, str],
    dir_name: str,
    python_version="3.9",
    packages=None,
    verbose=False,
) -> Tuple[str, Optional[str]]:
    result = "Modal execution failed"
    sterror = None
    try:
        with modal.NetworkFileSystem.ephemeral() as nfs:
            for name, script in scripts_by_name.items():
                nfs.write_file(name, io.BytesIO(script.encode()))
            image = modal.Image.debian_slim(python_version=python_version)
            if packages:
                image = image.pip_install("uv")
                package_install_command = " ".join(packages)
                image = image.run_commands(
                    f"uv pip install --system --compile-bytecode {package_install_command}"
                )
            sb = modal.Sandbox.create(
                "bash",
                "-c",
                f"cd /vol && python -W ignore {script_to_run_by_name}",
                image=image,
                timeout=120,
                cloud="aws",
                network_file_systems={"/vol": nfs},
            )
            sb.wait()
            stdout = sb.stdout.read()
            stderr = sb.stderr.read()
            result = stdout
            sterror = stderr
    except modal.exception.SandboxTimeoutError:
        result = "Execution timed out after 60 seconds"
        sterror = None
    return result, sterror
