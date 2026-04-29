import os
import os.path as osp


def is_macos_metadata_file(filename):
    """Return True for AppleDouble and Finder metadata files."""
    basename = osp.basename(os.fspath(filename))
    return basename.startswith("._") or basename == ".DS_Store"


def listdir_without_metadata(path):
    return [
        filename
        for filename in os.listdir(path)
        if not is_macos_metadata_file(filename)
    ]
