import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Type

from pydantic import BaseModel

from apropos.src.bench.bigcodebench.backends.docker import execute_code_remotely_docker
from apropos.src.bench.bigcodebench.main import (
    BigCodeBench_Question,
    BigCodeBenchComplete_Benchmark,
)

from smallbench.utilities.code.docker import execute_code_docker
from smallbench.benchmarks.bcb_a.abstractions import (
    CurrentSolution,
    ACIAction,
    Transform,
)
from smallbench.benchmarks.bcb_a.contexts import unit_test_context
from smallbench.core.aci import AgentComputerInterface


class BCBUnitTest(BaseModel):
    test_description: str
    input_definitions: Dict[str, str]
    assertion_condition: str
    assertion_type: Literal["assertTrue", "assertRaises"] = "assertTrue"

    def to_python(self, index: int, function_name: str = "task_func"):
        definitions = []
        arguments = {}
        for i, (input_name, input_value) in enumerate(self.input_definitions.items()):
            definitions.append(f"var_{i} = {input_value}")
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
    bcb_question: BigCodeBench_Question

    def __init__(self, bcb_question: BigCodeBench_Question):
        self.bcb_question = bcb_question

    def temp_code_hack(self, headless_submission: str):
        # TEMP HACK
        if "task_func" in headless_submission:
            lines = headless_submission.split("\n")
            i = [i for i, line in enumerate(lines) if "def task_func" in line][0]
            headless_submission = "\n".join(lines[(i + 1) :])
        return headless_submission

    async def execute_final_submission_against_hidden_tests(self, final_submission):
        final_submission = self.temp_code_hack(final_submission)
        success, result = await execute_code_remotely_docker(
            self.bcb_question.information, final_submission
        )
        return success, result

    async def execute_submission_against_tests(
        self, headless_submission: str, tests: List[BCBUnitTest]
    ):  # Write new docker code for this
        headless_submission = self.temp_code_hack(headless_submission)

        head_imports = re.findall(
            r"import (\w+)", self.bcb_question.information["eval_info"]["code_prompt"]
        )
        head_imports_with_alias = re.findall(
            r"import (\w+) as (\w+)",
            self.bcb_question.information["eval_info"]["code_prompt"],
        )
        imports_snippet = "\n".join(
            ["import " + imp for imp in head_imports]
            + [
                "import " + imp + " as " + alias
                for imp, alias in head_imports_with_alias
            ]
        )
        tests_snippet = f"""
import unittest
{imports_snippet}
class TestCases(unittest.TestCase):
"""
        for i, test in enumerate(tests):
            tests_snippet += test.to_python(index=i)

        full_script = (
            self.bcb_question.information["eval_info"]["code_prompt"]
            + "\n"
            + headless_submission
            + "\n"
            + tests_snippet
        )
        eval_snippet = """
import unittest
import io
def test_code():
    path = "script.py"
    loader = unittest.TestLoader()
    suite = loader.discover('/app', pattern=path)
    runner = unittest.TextTestRunner()
    assert suite.countTestCases() != 0, "No tests found in script.py"
    result = runner.run(suite)

    result_dict = {
        "errors": len(result.errors),
        "failures": len(result.failures),
        "testsRun": result.testsRun,
        "wasSuccessful": result.wasSuccessful()
    }
    return result.wasSuccessful(), result_dict

if __name__ == "__main__":
    success, result = test_code()
    print("Success:", success)
    print(result)
"""
        import sys

        standard_libs = set(sys.stdlib_module_names)
        all_unique_imports = [imp for imp in head_imports if imp not in standard_libs]
        results = await execute_code_docker(
            script_to_run_by_name="eval.py",
            scripts_by_name={"eval.py": eval_snippet, "script.py": full_script},
            python_version="python:3.9-slim",
            packages=all_unique_imports,
            dir_name="bcb",
        )
        return results


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
    input_definitions: Dict[str, Any]
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
        return await action._act(action_args)

    async def _execute_submission_against_tests(self):
        results = await self.bcb_backend.execute_submission_against_tests(
            self.current_solution.for_execution(), list(self.unit_tests.values())
        )
        return results

    def edit_submission(self, first_line: int, last_line: int, new_code: str):
        self.current_solution.lines[first_line:last_line] = new_code.split("\n")

    def add_submission(self, submission: str):
        self.current_solution = CurrentSolution(lines=submission.split("\n"))

    def add_unit_test(self, unit_test_name: str, unit_test_dict: Dict):
        self.unit_tests[unit_test_name] = BCBUnitTest(**unit_test_dict)

    def remove_unit_test(self, unit_test_name: str):
        self.unit_tests.pop(unit_test_name)

    async def _submit_solution(self):
        (
            success,
            result,
        ) = await self.bcb_backend.execute_final_submission_against_hidden_tests(
            self.current_solution.for_execution()
        )
        self.final_submission = self.current_solution.for_execution()
        self.final_success = success
        return success, result

    def check_termination(self):
        return self.final_success is not False