"""Recover the append point of resumable self-distillation JSONL output. Stdlib only."""
import json
import errno
import os
from contextlib import contextmanager


class OutputLockedError(RuntimeError):
    pass


@contextmanager
def exclusive_output(path):
    """Hold a nonblocking OS lock across resume, engine construction and all writes.

    Keep the sidecar inode: unlinking it could let a new process lock another inode.
    A process exit releases the lock automatically, so crash recovery needs no deletion.
    """
    with open(f"{path}.lock", "a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            raise OutputLockedError(f"{path} is already being generated; wait for the other process to finish") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def load_completed(path):
    """Keep complete records, drop only a torn final record and separate a complete final line.

    Corruption before the last record is an error; silently skipping it would leave a
    JSONL file that capture.py cannot read. Nothing is changed in that case.
    """
    done = set()
    try:
        output = open(path, "rb+")
    except FileNotFoundError:
        return done
    with output:
        needs_newline = False
        while True:
            offset = output.tell()
            line = output.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except ValueError as exc:
                if output.read(1):
                    raise ValueError(f"{path}: invalid JSONL record at byte {offset}; back up the file, "
                                     f"then repair that record or truncate at byte {offset} and rerun to regenerate the tail") from exc
                output.truncate(offset)
                break
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"{path}: record at byte {offset} has no id")
            done.add(record["id"])
            needs_newline = not line.endswith(b"\n")
        if needs_newline:
            output.write(b"\n")
    return done
