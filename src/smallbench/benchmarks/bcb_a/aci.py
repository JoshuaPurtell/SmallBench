import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Type, Tuple

from pydantic import BaseModel

from apropos.src.bench.bigcodebench.backends.docker import execute_code_remotely_docker
from apropos.src.bench.bigcodebench.backends.modal import (
    execute_code_remotely_modal,
)  # change to execute_code_remotely_modal asap
from apropos.src.bench.bigcodebench.main import (
    BigCodeBench_Question,
    BigCodeBenchComplete_Benchmark,
)
import sys
from smallbench.utilities.code.docker import execute_code_docker
from smallbench.utilities.code.modal import execute_code_modal
from smallbench.benchmarks.bcb_a.abstractions import (
    CurrentSolution,
    ACIAction,
    Transform,
)
from smallbench.benchmarks.bcb_a.contexts import unit_test_context
from smallbench.core.aci import AgentComputerInterface


class BCBUnitTest(BaseModel):
    test_description: str
    # input_definitions: Dict[str, str]
    input_names: List[str]
    input_types: List[str]
    input_values: List[Any]
    assertion_condition: str
    assertion_type: Literal["assertTrue", "assertRaises"] = "assertTrue"

    def to_python(self, index: int, function_name: str = "task_func"):
        definitions = []
        arguments = {}
        for i, (input_name, input_type, input_value) in enumerate(
            zip(self.input_names, self.input_types, self.input_values)
        ):
            if not input_type == "str":
                definitions.append(f"var_{i} = {input_value}")
            else:
                definitions.append(f"var_{i} = '{input_value}'")
            arguments[input_name] = f"var_{i}"
        args = ", ".join([f""""{k}":{v}""" for k, v in arguments.items()])
        defs = "\n        ".join(definitions)
        if self.assertion_type == "assertTrue":
            return f"""
    def test_case_{index}(self):
        # {self.test_description}

        {defs}
        result = {function_name}(**{{{args}}})
        self.{self.assertion_type}({self.assertion_condition})
"""
        elif self.assertion_type == "assertRaises":
            return f"""
    def test_case_{index}(self):
        # {self.test_description}
        {defs}
        with self.{self.assertion_type}({self.assertion_condition}):
            {function_name}(**{{{args}}})
"""
        else:
            raise ValueError(f"Invalid assertion type: {self.assertion_type}")


class BCBEngine:
    backend: Literal["docker", "modal"] = "docker"
    bcb_question: BigCodeBench_Question

    def __init__(
        self,
        bcb_question: BigCodeBench_Question,
        backend: Literal["docker", "modal"] = "docker",
    ):
        self.bcb_question = bcb_question
        self.backend = backend

    def temp_code_hack(self, headless_submission: str):
        # TEMP HACK
        if "task_func" in headless_submission:
            lines = headless_submission.split("\n")
            i = [i for i, line in enumerate(lines) if "def task_func" in line][0]
            headless_submission = "\n".join(lines[(i + 1) :])
        return headless_submission

    async def execute_final_submission_against_hidden_tests(self, final_submission):
        final_submission = self.temp_code_hack(final_submission)
        all_unique_imports, imports_snippet = self.get_imports()
        if self.backend == "docker":
            success, result = await execute_code_remotely_docker(
                self.bcb_question.information,
                final_submission,
                packages=all_unique_imports,
            )  # TODO: combine this and the unit testing code ASAP
        elif self.backend == "modal":
            success, result = await execute_code_remotely_modal(
                self.bcb_question.information,
                final_submission,
                packages=all_unique_imports,
            )
        else:
            raise ValueError(f"Invalid backend: {self.backend}")
        return success, result

    def get_imports(self) -> Tuple[List[str], str]:
        code_prompt = self.bcb_question.information["eval_info"]["code_prompt"]
        from_imports = re.findall(r"from (\w+) import (\w+)", code_prompt)
        head_imports = re.findall(r"^import (\w+)(?!\s+as)$", code_prompt, re.MULTILINE)
        head_imports_with_alias = re.findall(r"import ([\w.]+) as (\w+)", code_prompt)
        imports_snippet = "\n".join(
            ["import " + imp for imp in head_imports]
            + [f"import {imp} as {alias}" for imp, alias in head_imports_with_alias]
            + [f"from {package} import {class_or_function}" for package, class_or_function in from_imports]
        )
        standard_libs = set(sys.stdlib_module_names)
        all_unique_imports = [imp for imp in head_imports if imp not in standard_libs]
        for imp, _ in head_imports_with_alias:
            package = imp.split('.')[0]
            if package not in all_unique_imports and package not in standard_libs:
                all_unique_imports.append(package)
        for package, _ in from_imports:
            if package not in all_unique_imports and package not in standard_libs:
                all_unique_imports.append(package)
        return all_unique_imports, imports_snippet

    async def execute_submission_against_tests(
        self, headless_submission: str, tests: List[BCBUnitTest]
    ):  # Write new docker code for this
        headless_submission = self.temp_code_hack(headless_submission)
        all_unique_imports, imports_snippet = self.get_imports()
        tests_snippet = f"""
import unittest
{imports_snippet}
class TestCases(unittest.TestCase):
"""
        assert len(tests) > 0, "No tests found"
        for i, test in enumerate(tests):
            tests_snippet += test.to_python(index=i)
        assert len(tests_snippet) > 0, "Tests snippet is empty"

        full_script = (
            self.bcb_question.information["eval_info"]["code_prompt"]
            + "\n"
            + headless_submission
            + "\n"
            + tests_snippet
        )
        if self.backend == "modal":
            location = "/vol"
        elif self.backend == "docker":
            location = "/app"
        else:
            raise Exception("Invalid backend")
        eval_snippet = f"""
import unittest
import io
def test_code():
    path = "script.py"
    loader = unittest.TestLoader()
    suite = loader.discover('{location}', pattern=path)
    runner = unittest.TextTestRunner()
    assert suite.countTestCases() != 0, "No tests found in script.py"
    result = runner.run(suite)

    result_dict = {{
        "errors": len(result.errors),
        "failures": len(result.failures),
        "testsRun": result.testsRun,
        "wasSuccessful": result.wasSuccessful()
    }}
    return result.wasSuccessful(), result_dict

if __name__ == "__main__":
    success, result = test_code()
    print("Success:", success)
    print(result)
"""
        if self.backend == "docker":
            results = await execute_code_docker(
                script_to_run_by_name="eval.py",
                scripts_by_name={"eval.py": eval_snippet, "script.py": full_script},
                python_version="python:3.9-slim",
                packages=all_unique_imports,
                dir_name="bcb",
            )
            logs_pattern = r"([\s\S]*?)(?=Success:)"
            test_results_pattern = r"Success: (True|False)\s*(\{.*\})"

            logs_match = re.search(logs_pattern, results, re.DOTALL)
            test_results_match = re.search(test_results_pattern, results, re.DOTALL)
            if logs_match and test_results_match:
                logs = logs_match.group(1).strip()
                success = test_results_match.group(1) == "True"
                test_results = eval(test_results_match.group(2))
            else:
                print("No match found")
                logs = ""
                success = False
                test_results = {}
            success = (
                success
                and test_results["wasSuccessful"]
                and test_results["testsRun"] > 0
            )
            if success:
                result = "All tests passed"
            elif logs == "":
                result = "Running tests resulted in an execution hardfail"
            else:
                result = f"Not all tests passed. Results summary: {test_results}. \nExecution logs: {logs}"
            # parse these
        elif self.backend == "modal":
            results, sterror = await execute_code_modal(
                script_to_run_by_name="eval.py",
                scripts_by_name={"eval.py": eval_snippet, "script.py": full_script},
                python_version="3.9",
                packages=all_unique_imports,
                dir_name="bcb",
                verbose=True,
            )
            result = results
        else:
            raise ValueError(f"Invalid backend: {self.backend}")
        return result


class BCBAgentComputerInterface(AgentComputerInterface):
    unit_tests: Dict[str, BCBUnitTest]
    sketch: str  # TODO: add this!
    current_solution: CurrentSolution
    bcb_backend: BCBEngine
    actions: List[ACIAction]
    final_submission: str
    final_success: bool
    terminated: bool

    def __init__(self, bcb_backend: BCBEngine):
        self.unit_tests = {}
        self.current_solution = CurrentSolution(lines=[])
        self.bcb_backend = bcb_backend
        self.actions = [
            ACIAction(
                action_name="edit_submission",
                action_arg_spec={"first_line": int, "last_line": int, "new_code": str},
                action_description="Edit the submission code",
                action_context="Edit the submission code. Use this when you want to make changes to the current solution.",
                transform=Transform(callable=self.edit_submission, type="sync"),
            ),
            ACIAction(
                action_name="add_submission",
                action_arg_spec={"submission": str},
                action_description="Add the submission code",
                action_context="Add the submission code. Use this when you want to start from scratch with a new solution.",
                transform=Transform(callable=self.add_submission, type="sync"),
            ),
            ACIAction(
                action_name="add_unit_test",
                action_arg_spec={"unit_test_name": str, "unit_test_dict": Dict},
                action_description="Add a unit test",
                action_context=f"""Add a unit test. The unit test information you submit must be in the format of a BCBUnitTest: \n
class BCBUnitTest(BaseModel):
    test_description: str
    input_names: List[str]
    input_types: List[str]
    input_values: List[Any]
    assertion_condition: str
    assertion_type: Literal["assertTrue", "assertRaises"] = "assertTrue"
\n\n It will be parsed via BCBUnitTest(**unit_test_dict)


{unit_test_context}""",
                transform=Transform(callable=self.add_unit_test, type="sync"),
            ),
            ACIAction(
                action_name="remove_unit_test",
                action_arg_spec={"unit_test_name": str},
                action_description="Remove a unit test",
                action_context="Remove a unit test",
                transform=Transform(callable=self.remove_unit_test, type="sync"),
            ),
            ACIAction(
                action_name="test_submission",
                action_arg_spec={},
                action_description="Test the submission",
                action_context="Test the submission",
                transform=Transform(
                    callable=self._execute_submission_against_tests, type="async"
                ),
            ),
            ACIAction(
                action_name="submit_solution",
                action_arg_spec={},
                action_description="Submit the solution",
                action_context="Submit the solution",
                transform=Transform(callable=self._submit_solution, type="async"),
            ),
        ]
        self.final_success = False
        self.final_submission = None
        self.terminated = False

    async def accept_delta(self, action_name: str, action_args: Dict):
        action = next(
            (action for action in self.actions if action.action_name == action_name),
            None,
        )
        if action is None:
            raise ValueError(f"Action {action_name} not found")
        result = await action._act(action_args)
        return result

    async def _execute_submission_against_tests(self):
        results = await self.bcb_backend.execute_submission_against_tests(
            self.current_solution.for_execution(), list(self.unit_tests.values())
        )
        return results

    def edit_submission(self, first_line: int, last_line: int, new_code: str):
        if not isinstance(new_code, str):
            return "New code must be a string"
        self.current_solution.lines[first_line:last_line] = new_code.split("\n")
        return "Edited submission successfully"

    def add_submission(self, submission: str):
        if not isinstance(submission, str):
            return "Submission must be a string"
        self.current_solution = CurrentSolution(lines=submission.split("\n"))
        return "Added submission successfully"

    def add_unit_test(self, unit_test_name: str, unit_test_dict: Dict):
        try:
            self.unit_tests[unit_test_name] = BCBUnitTest(**unit_test_dict)
            return "Added unit test successfully"
        except Exception as e:
            return f"Failed to add unit test: {e}"

    def remove_unit_test(self, unit_test_name: str):
        try:
            self.unit_tests.pop(unit_test_name)
            return "Removed unit test successfully"
        except Exception as e:
            return f"Failed to remove unit test: {e}"

    async def _submit_solution(self):
        (
            tests_ran,
            result,
        ) = await self.bcb_backend.execute_final_submission_against_hidden_tests(
            self.current_solution.for_execution()
        )
        if result["testsRun"] == 0:
            print("Warning: No final tests ran")
        success = tests_ran and result["wasSuccessful"] and result["testsRun"] > 0

        self.final_submission = self.current_solution.for_execution()
        self.final_success = success
        return f"Solution submitted successfully, Success: {success}"

    def check_termination(self):
        return self.final_success is not False or self.final_submission is not None
