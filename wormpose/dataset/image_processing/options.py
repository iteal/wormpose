WORM_IS_LIGHTER = "worm_is_lighter"


def add_image_processing_arguments(parser):
    parser.add_argument(f"--{WORM_IS_LIGHTER}", action="store_true")
