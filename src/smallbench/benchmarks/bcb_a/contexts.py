# Def would like a cleaner way to do this ... but no start is perfect!
unit_test_context = """
# Some various notes:
1. If an input should be of a type defined by a specific package, add the package name/alias to the type. E.g. "np.ndarray" or "pd.DataFrame". You still should fully define the value for the input_value field e.g. "pd.DataFrame({'a': [1, 2, 3]})"

2. Unit tests will be compiled from the BCBUnitTest class as follows:
    A. For AssertTrue type tests, the test will be compiled as follows:
    ```python
    def test_case(self):
        # {{self.test_description}}

        {{defs}}
        result = {{function_name}}(**{{{{args}}}}})
        self.{{self.assertion_type}}({{self.assertion_condition}})
    ```
    B. For AssertRaises type tests, the test will be compiled as follows:

    ```python
    def test_case(self):
        # {{self.test_description}}
        {{defs}}
        with self.{{self.assertion_type}}({{self.assertion_condition}}):
            {{function_name}}(**{{{{args}}}}})
    ```

    Provide information accordingly.
"""
