import io
from typing import Dict, Optional, Tuple

import modal


async def execute_code_modal(
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

            install_command = ""
            if packages:
                install_command = f"pip install {' '.join(packages)} && "
            image = modal.Image.debian_slim(python_version=python_version)
            sb = modal.Sandbox.create(
                "bash",
                "-c",
                f"{install_command}cd /vol && python -W ignore {script_to_run_by_name}",
                image=image,
                timeout=120,
                cloud="aws",
                network_file_systems={"/vol": nfs},
            )
            await sb.wait.aio()
            stdout = await sb.stdout.read.aio()
            stderr = await sb.stderr.read.aio()
            #if verbose:
                # print("Results:")
                # print("#"* 20)
                # print(stdout)
                # print("---")
                # print(stderr)
                # print("---")
                # print(scripts_by_name)
                # print("----")
                # print(packages)
                # print("#" * 20)
            result = stdout
            sterror = stderr
    except modal.exception.SandboxTimeoutError:
        result = "Execution timed out after 60 seconds"
        sterror = None
    return result, sterror
