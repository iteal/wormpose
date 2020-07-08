"""
Wrapper to save the training data to different file formats
"""


class GenericFileWriter(object):
    """
    Write data to different file formats depending on the open_file and write_file functions
    """

    def __init__(self, open_file=None, write_file=None):
        self.open_file = open_file
        self.write_file = write_file

    def __enter__(self):
        self.f = self.open_file()
        self.f.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.__exit__(exc_type, exc_val, exc_tb)

    def write(self, data):
        self.write_file(self.f, data)
