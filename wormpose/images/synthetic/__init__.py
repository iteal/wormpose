"""
Making images for the synthetic dataset
"""

# synthetic shape generator params:
# draw worm segments of a length 1/16 of the worm length
BODY_SEGMENTS_DIVIDER = 16
# draw patches a little larger than the worm thickness to add some background pixels
THICKNESS_MULTIPLIER = 1.2

# data augmentation parameters (applied when enable_random_augmentations is True):
# extra blurring happens 25% of the time
BLUR_PROBABILITY = 0.25
# draw tail on top of head 50% of the time
DRAW_TAIL_ON_TOP_PROBABILITY = 0.5
# extra translation max 5% of image size
MAX_OFFSET_PERCENT = 0.05
# when applying gaussian blur, minimum window size (must be an odd number)
GAUSSIAN_BLUR_MIN_SIZE = 3
# when applying gaussian blur, maximum window size (must be an odd number)
GAUSSIAN_BLUR_MAX_SIZE = 13
# when applying gaussian blur, maximum window size as a percentage of the image size
GAUSSIAN_BLUR_MAX_SIZE_PERCENT = 0.10
# change thickness multiplier by this amount + or -
THICKNESS_MULTIPLIER_OFFSET = 0.1
# multiply worm length by a random amount between 1 -/+ WORM_LENGTH_MULTIPLIER_OFFSET
WORM_LENGTH_MULTIPLIER_OFFSET = 0.1


from wormpose.images.synthetic.synthetic_dataset import SyntheticDataset
