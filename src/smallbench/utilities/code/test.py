import modal
import io
import asyncio

app = modal.App()
volume = modal.Volume.from_name("bcb-volume", create_if_missing=True)

@app.function(volumes={"/vol": volume})
async def run_code(scripts_by_name):
    for name, script in scripts_by_name.items():
        with open(f"/vol/{name}", "w") as f:
            f.write(script)
    
    bash_command = "pip install pandas numpy && cd /vol && python -W ignore eval.py"
    python_version = "python:3.9-slim"
    sb = modal.Sandbox.create(
        "bash",
        "-c",
        bash_command,
        image=python_version,
        timeout=60,
        cloud="aws",
    )
    await sb.wait.aio()
    stdout = await sb.stdout.read.aio()
    stderr = await sb.stderr.read.aio()
    return stdout, stderr

async def main():
    scripts_by_name = {'eval.py': '\nimport unittest\nimport io\ndef test_code():\n    path = "script.py"\n    loader = unittest.TestLoader()\n    suite = loader.discover(\'/vol\', pattern=path)\n    runner = unittest.TextTestRunner()\n    assert suite.countTestCases() != 0, "No tests found in script.py"\n    result = runner.run(suite)\n\n    result_dict = {\n        "errors": len(result.errors),\n        "failures": len(result.failures),\n        "testsRun": result.testsRun,\n        "wasSuccessful": result.wasSuccessful()\n    }\n    return result.wasSuccessful(), result_dict\n\nif __name__ == "__main__":\n    success, result = test_code()\n    print("Success:", success)\n    print(result)\n', 'script.py': 'import pandas as pd\nimport numpy as np\n# Constants\nCOLUMNS = [\'column1\', \'column2\', \'column3\', \'column4\', \'column5\']\ndef task_func(df, dct):\n\n    if not isinstance(df, pd.DataFrame):\n        raise ValueError(\'Input must be a DataFrame\')\n    \n    # Replace values in the DataFrame according to the dictionary\n    df_replaced = df.replace(dct)\n    \n    # Ensure all data is numeric for correlation calculation\n    df_numeric = df_replaced.apply(pd.to_numeric, errors=\'coerce\')\n    \n    # Calculate the Pearson correlation coefficient matrix\n    correlation_matrix = df_numeric.corr(method=\'pearson\')\n    \n    return correlation_matrix\n\nimport unittest\nimport pandas\nimport numpy\nimport pandas as pd\nimport numpy as np\nclass TestCases(unittest.TestCase):\n\n    def test_case_0(self):\n        # Test task_func with a simple DataFrame and replacement dictionary.\n\n        var_0 = pd.DataFrame({\'A\': [1, 2, 3], \'B\': [4, 5, 6]})\n        var_1 = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}\n        result = task_func(**{"df":var_0, "dct":var_1})\n        self.assertTrue(np.allclose(result, np.array([[1.0, 1.0], [1.0, 1.0]])))\n'}
    
    result, sterror = await run_code.remote(scripts_by_name)
    return result, sterror

if __name__ == "__main__":
    result, sterror = asyncio.run(main())
    print(result)
    print(sterror)