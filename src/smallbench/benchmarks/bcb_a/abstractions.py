from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Type

from pydantic import BaseModel


@dataclass
class CurrentSolution:
    lines: List[str]

    def for_show(self):
        return "\n".join(["i: " + line for i, line in enumerate(self.lines)])

    def for_execution(self):
        return "\n".join(self.lines)


class SubmissionEdit(BaseModel):
    first_line: int
    last_line: int
    new_code: str


@dataclass
class Transform:
    callable: Callable
    type: Literal["sync", "async"]


class ActionResult(BaseModel):
    success: bool
    result: Any


class ACIAction(BaseModel):
    action_name: str
    action_arg_spec: Dict[str, Any]
    action_description: str
    action_context: str
    transform: Transform

    def validate_action_args(self, action_args: Dict):
        for arg_name, arg_type in self.action_arg_spec.items():
            if arg_name not in action_args:
                raise ValueError(
                    f"Action {self.action_name} requires argument {arg_name}"
                )
            if not isinstance(action_args[arg_name], arg_type):
                raise ValueError(
                    f"Action {self.action_name} requires argument {arg_name} to be of type {arg_type}"
                )

    def _act_sync(self, action_args: Dict):
        try:
            self.validate_action_args(action_args)
        except Exception as e:
            return ActionResult(success=False, result=str(e))
        result = self.transform.callable(**action_args)
        return ActionResult(success=True, result=result)

    async def _act_async(self, action_args: Dict):
        try:
            self.validate_action_args(action_args)
        except Exception as e:
            return ActionResult(success=False, result=str(e))

        result = await self.transform.callable(**action_args)
        return ActionResult(success=True, result=result)
