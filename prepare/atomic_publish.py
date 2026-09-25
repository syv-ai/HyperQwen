"""Crash-safe writes for the scripts that rewrite a model directory in place (#195).

docker/prepare.sh runs these scripts on every container start, under `set -e`. A file
they leave half-written stops every later start, so every writer follows one protocol:

  1. backup_once() each file before its first change. The first backup is the pristine
     copy and nothing overwrites it, so a re-run after a kill cannot back up a modified
     file over the original.
  2. Write each output to <path>.tmp, fsync it, rename it over <path>, fsync the
     directory. A reader sees the old file or the new file, never part of one, and the
     new inode never changes a hardlinked copy (the fast variant links base shards).
  3. Write the shards first, config.json next, and the safetensors index last. The index
     is the commit point: state() in docker/prepare.sh reads only the index to decide
     which steps are done, so a step stays pending until its index lands, and the next
     prepare run completes it.
"""

import json
import os
import shutil


def _fsync_dir(path):
    try:
        fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # some filesystems (9p, drvfs) refuse a directory fsync
    finally:
        os.close(fd)


def publish(tmp, dst):
    """Make the completely written file `tmp` visible as `dst` in one rename."""
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    _fsync_dir(dst)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    publish(tmp, path)


def write_text(path, text):
    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    publish(tmp, path)


def save_tensors(tensors, path, metadata):
    from safetensors.torch import save_file

    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata=metadata)
    publish(tmp, path)


def backup_once(path, suffix):
    """Copy `path` to `path + suffix` unless that backup exists: the first backup is the
    pristine one. The copy is published atomically, so a kill cannot leave a partial
    backup that a re-run would then keep."""
    dst = path + suffix
    if not os.path.exists(dst):
        shutil.copy(path, dst + ".tmp")
        publish(dst + ".tmp", dst)
