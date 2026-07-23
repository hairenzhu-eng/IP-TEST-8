import numpy as np

from laptop import segment_intersects_ellipse


def test_apf_field_only_blocks_a_route_that_enters_it():
    assert segment_intersects_ellipse([0, 0], [10, 0], [5, 0], [1, 0], 1, 1, 0.2)
    assert not segment_intersects_ellipse([0, 0], [10, 0], [5, 3], [1, 0], 1, 1, 0.2)


if __name__ == "__main__":
    test_apf_field_only_blocks_a_route_that_enters_it()
