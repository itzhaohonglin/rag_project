import pytest


@pytest.fixture
def sample_text() -> str:
    return "这是第一段内容。\n\n这是第二段内容，包含更多信息。\n\n第三段有一些详细说明。"


@pytest.fixture
def sample_code() -> str:
    return """def hello():
    print("hello world")

class MyClass:
    def method(self):
        pass

def another_function():
    return 42
"""
