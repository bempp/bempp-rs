import os
import pytest

examples = []
root_dir = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), os.path.join("..", "..")))

for f in os.listdir(root_dir):
    if not f.startswith("."):
        if (
            os.path.isdir(os.path.join(root_dir, f))
            and os.path.isdir(os.path.join(root_dir, os.path.join(f, "examples")))
        ):
            for file in os.listdir(os.path.join(root_dir, os.path.join(f, "examples"))):
                if not file.startswith(".") and file.endswith(".rs"):
                    examples.append(os.path.join(os.path.join(f, "examples"), file))


@pytest.mark.parametrize("example_file", examples)
def test_examples_are_strict(example_file):
    with open(os.path.join(root_dir, example_file)) as f:
        example = f.read()
    strict = '#![cfg_attr(feature = "strict", deny(warnings))]'
    if strict not in example:
        raise ValueError(f"You need to include `{strict}` in the file {example_file}")
