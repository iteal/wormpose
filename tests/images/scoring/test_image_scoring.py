import numpy as np

from wormpose.images.scoring.image_scoring import calculate_similarity


def test_calculate_similarity():

    rand = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    square = np.full((10, 10), 255, dtype=np.uint8)
    square[4:6, 4:6] = 0

    input_outputs = [
        (np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8), 1, (0, 0)),
        (np.ones((5, 10), dtype=np.uint8), np.ones((5, 10), dtype=np.uint8), 1, (0, 0)),
        (rand, rand, 1, (0, 0)),
        (rand, 255 - rand, 1, (0, 0)),
        (np.full((10, 10), 255, dtype=np.uint8), square, 0, (0, 0)),
        (square, square[3:7, 3:7], 1, (3, 3)),
    ]

    for input_a, input_b, expected_output_a, expected_output_b in input_outputs:
        output_a, output_b = calculate_similarity(input_a, input_b)
        assert np.allclose(output_a, expected_output_a, atol=1e-5, equal_nan=True)
        assert np.allclose(output_b, expected_output_b, atol=1e-5, equal_nan=True)
