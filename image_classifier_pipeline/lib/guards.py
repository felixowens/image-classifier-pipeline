from typing import List, Optional

from annotated_types import T


def assert_list(labels: List[Optional[T]]) -> List[T]:
    assert all(label is not None for label in labels)
    return [label for label in labels if label is not None]
