import numpy as np

from wormpose.images.scoring.image_scoring import calculate_similarity, fit_bounding_box_to_worm


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


def test_fit_bounding_box_to_worm():

    rect_image = np.full((10, 10), 250, dtype=np.uint8)
    rect_image[3:8, 2:5] = 30

    input_outputs = [
        (rect_image, 250, 0, np.s_[3:8, 2:5]),
        (rect_image, 250, 1, np.s_[2:9, 1:6]),
        (np.full((10, 10), 255, dtype=np.uint8), 255, 0, np.s_[0:10, 0:10]),
        (np.full((5, 10), np.nan, dtype=np.uint8), 255, 0, np.s_[0:5, 0:10]),
    ]

    for input_a, input_b, input_c, expected_output in input_outputs:
        output = fit_bounding_box_to_worm(input_a, input_b, input_c)
        assert output == expected_output
