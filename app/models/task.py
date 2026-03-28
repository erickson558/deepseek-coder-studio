from enum import Enum


class TaskType(str, Enum):
    CODE_GENERATION = "code_generation"
    BUG_FIXING = "bug_fixing"
    REFACTOR = "refactor"
    TEST_GENERATION = "test_generation"
    CODE_EXPLANATION = "code_explanation"
    FILE_EDITING = "file_editing"
    CHAT = "chat"
