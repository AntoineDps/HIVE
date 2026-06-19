from pathlib import Path


def save_pathing(path, filename, file_id, suffix):

    # make sure path is Path
    path = Path(path)

    # make sure repertory exist
    path.mkdir(parents=True, exist_ok=True)

    # build filename string
    if filename is None:
        fullname = file_id
    else:
        fullname = "_".join([filename, file_id])

    # build path
    full_path = (path / fullname).with_suffix(suffix)

    return full_path
