import os
import glob


def make_doctree(out_dir):

    doctree = ""

    root_dir = os.path.join(os.pardir)

    source_files = sorted(glob.glob(os.path.join(root_dir, "wormpose", "**", "[!_]*.py"), recursive=True))
    for source_file in source_files:
        source_file = os.path.relpath(source_file, root_dir)
        module = source_file.replace(os.sep, ".")
        module = os.path.splitext(module)[0]
        doctree += f"   {module}\n"

    with open(os.path.join("_templates", "header.rst"), "r") as f:
        header = f.read()

    with open(os.path.join(out_dir, "documentation.rst"), "w") as f:
        f.write(header)
        f.write(doctree)


if __name__ == "__main__":
    make_doctree("source")
