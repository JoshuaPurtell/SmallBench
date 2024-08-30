# Def would like a cleaner way to do this ... but no start is perfect!
unit_test_context = """
Unit tests will be compiled from the BCBUnitTest class as follows:
1. For AssertTrue type tests, the test will be compiled as follows:
```python
def test_case(self):
    # {{self.test_description}}

    {{defs}}
    result = {{function_name}}(**{{{{args}}}}})
    self.{{self.assertion_type}}({{self.assertion_condition}})
```
2. For AssertRaises type tests, the test will be compiled as follows:

```python
def test_case(self):
    # {{self.test_description}}
    {{defs}}
    with self.{{self.assertion_type}}({{self.assertion_condition}}):
        {{function_name}}(**{{{{args}}}}})
```

Provide information accordingly.
"""
