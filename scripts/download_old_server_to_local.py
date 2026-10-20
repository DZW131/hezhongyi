#!/usr/bin/env python
"""Download selected migration artifacts from the old server to local storage.

The script intentionally reads credentials from environment variables so that
passwords do not get written into the repository.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import posixpath
import queue
import shlex
import stat
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import paramiko


@dataclass(frozen=True)
class RemoteItem:
    remote: str
    local: str
    required: bool = True


@dataclass(frozen=True)
class RemoteFileInfo:
    path: str
    size: int
    mtime: int


FIRST_PASS_ITEMS = [
    RemoteItem(
        "/root/datasets/HZY_HSPN_export_ds025",
        "data/HZY_HSPN_export_ds025",
    ),
    RemoteItem(
        "/root/datasets/HZY_HSPN_tiles_glom",
        "data/HZY_HSPN_tiles_glom",
    ),
    RemoteItem(
        "/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom_scene_ds025_resume",
        "checkpoints/hzy_hspn_finetune_glom_scene_ds025_resume",
    ),
    RemoteItem(
        "/root/Pytorch-UNet/Pytorch-UNet-master/checkpoints/hzy_hspn_finetune_glom_scene_ds025",
        "checkpoints/hzy_hspn_finetune_glom_scene_ds025",
    ),
    RemoteItem(
        "/root/Pytorch-UNet/Pytorch-UNet-master/logs",
        "project_artifacts/logs",
    ),
    RemoteItem(
        "/root/Pytorch-UNet/Pytorch-UNet-master/predictions",
        "project_artifacts/predictions",
    ),
]

OPTIONAL_RAW_ITEMS = [
    RemoteItem(
        "/root/datasets/HZY_HSPN_raw",
        "data/HZY_HSPN_raw",
        required=False,
    ),
]


def configure_logging(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    log_path = dest / f"download_old_server_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("OLD_SERVER_HOST", "117.50.198.90"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OLD_SERVER_PORT", "23")))
    parser.add_argument("--user", default=os.environ.get("OLD_SERVER_USER", "root"))
    parser.add_argument("--key-file", type=Path, default=None, help="Private key for SSH auth")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(r"E:\临时存储文件\hezhongyi_unet_segm_old4090_20260616"),
        help="Local destination root",
    )
    parser.add_argument("--include-raw", action="store_true", help="Also download raw CZI/slice.db data")
    parser.add_argument("--dry-run", action="store_true", help="List files and sizes without downloading")
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--retries", type=int, default=5, help="Retry count per file")
    parser.add_argument("--workers", type=int, default=1, help="Parallel SFTP worker count")
    return parser.parse_args()


def connect(args: argparse.Namespace) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    password = os.environ.get("OLD_SERVER_PASSWORD")
    key_filename = str(args.key_file) if args.key_file else None
    if not password and not key_filename:
        password = getpass.getpass(f"Password for {args.user}@{args.host}: ")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        key_filename=key_filename,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    return client, client.open_sftp()


def walk_remote(sftp: paramiko.SFTPClient, remote_path: str) -> Iterable[tuple[str, paramiko.SFTPAttributes]]:
    stack = [remote_path.rstrip("/")]
    while stack:
        current = stack.pop()
        attrs = sftp.stat(current)
        if stat.S_ISDIR(attrs.st_mode):
            for child in sorted(sftp.listdir_attr(current), key=lambda item: item.filename, reverse=True):
                if child.filename in {".", ".."}:
                    continue
                stack.append(posixpath.join(current, child.filename))
        elif stat.S_ISREG(attrs.st_mode):
            yield current, attrs


def list_remote_files_fast(
    client: paramiko.SSHClient,
    item: RemoteItem,
) -> list[RemoteFileInfo]:
    command = (
        "find "
        + shlex.quote(item.remote)
        + r" -type f -printf '%s\t%T@\t%p\0'"
    )
    stdin, stdout, stderr = client.exec_command(command, timeout=300)
    del stdin
    payload = stdout.read()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        if err:
            logging.warning("find stderr for %s: %s", item.remote, err)
        raise FileNotFoundError(item.remote)

    files: list[RemoteFileInfo] = []
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        text = raw.decode("utf-8", errors="surrogateescape")
        size_text, mtime_text, path = text.split("\t", 2)
        files.append(
            RemoteFileInfo(
                path=path,
                size=int(size_text),
                mtime=int(float(mtime_text)),
            )
        )
    return files


def local_for(remote_file: str | RemoteFileInfo, item: RemoteItem, dest_root: Path) -> Path:
    remote_path = remote_file.path if isinstance(remote_file, RemoteFileInfo) else remote_file
    rel = posixpath.relpath(remote_path, item.remote)
    return dest_root / item.local / Path(*rel.split("/"))


def readable_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(
    sftp: paramiko.SFTPClient,
    remote_file: RemoteFileInfo,
    local_path: Path,
    chunk_size: int,
) -> str:
    expected_size = int(remote_file.size)
    if local_path.exists() and local_path.stat().st_size == expected_size:
        return "skip"

    part_path = local_path.with_name(local_path.name + ".part")
    ensure_parent(part_path)

    if local_path.exists() and local_path.stat().st_size != expected_size:
        logging.warning("Existing file has wrong size; resuming via .part: %s", local_path)
        if not part_path.exists():
            local_path.replace(part_path)

    offset = part_path.stat().st_size if part_path.exists() else 0
    if offset > expected_size:
        logging.warning("Discarding oversized partial file: %s", part_path)
        part_path.unlink()
        offset = 0

    with sftp.open(remote_file.path, "rb") as remote_handle:
        remote_handle.seek(offset)
        with part_path.open("ab") as local_handle:
            while offset < expected_size:
                data = remote_handle.read(min(chunk_size, expected_size - offset))
                if not data:
                    raise IOError(f"unexpected EOF while downloading {remote_file.path}")
                local_handle.write(data)
                offset += len(data)

    if part_path.stat().st_size != expected_size:
        raise IOError(f"size mismatch after download: {remote_file.path}")

    part_path.replace(local_path)
    os.utime(local_path, (int(remote_file.mtime), int(remote_file.mtime)))
    return "download"


def reconnect(
    args: argparse.Namespace,
    client: paramiko.SSHClient,
    sftp: paramiko.SFTPClient,
) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    try:
        sftp.close()
    except Exception:
        pass
    try:
        client.close()
    except Exception:
        pass
    logging.info("Reconnecting to %s:%s", args.host, args.port)
    return connect(args)


def download_one_with_retries(
    args: argparse.Namespace,
    client: paramiko.SSHClient,
    sftp: paramiko.SFTPClient,
    remote_file: RemoteFileInfo,
    local_path: Path,
) -> tuple[str | None, Exception | None, paramiko.SSHClient, paramiko.SFTPClient]:
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            status = download_file(sftp, remote_file, local_path, args.chunk_size)
            return status, None, client, sftp
        except Exception as exc:
            last_error = exc
            logging.warning(
                "Attempt %d/%d failed for %s: %r",
                attempt,
                args.retries,
                remote_file.path,
                exc,
            )
            if attempt < args.retries:
                time.sleep(min(60, 5 * attempt))
                client, sftp = reconnect(args, client, sftp)
    return None, last_error, client, sftp


def download_files_sequential(
    args: argparse.Namespace,
    client: paramiko.SSHClient,
    sftp: paramiko.SFTPClient,
    remote_files: list[tuple[RemoteItem, RemoteFileInfo]],
) -> int:
    downloaded = skipped = failed = 0
    copied_bytes = skipped_bytes = 0
    for index, (item, remote_file) in enumerate(remote_files, start=1):
        local_path = local_for(remote_file, item, args.dest)
        size = int(remote_file.size)
        status, error, client, sftp = download_one_with_retries(args, client, sftp, remote_file, local_path)
        if error is None and status == "skip":
            skipped += 1
            skipped_bytes += size
        elif error is None:
            downloaded += 1
            copied_bytes += size
        else:
            failed += 1
            logging.error(
                "Failed permanently: %s",
                remote_file.path,
                exc_info=(type(error), error, error.__traceback__),
            )

        if index == 1 or index % 25 == 0 or status == "download":
            logging.info(
                "[%d/%d] %s %s (%s)",
                index,
                len(remote_files),
                status or "failed",
                remote_file.path,
                readable_size(size),
            )

    logging.info(
        "Done. downloaded=%d (%s), skipped=%d (%s), failed=%d",
        downloaded,
        readable_size(copied_bytes),
        skipped,
        readable_size(skipped_bytes),
        failed,
    )
    return 1 if failed else 0


def download_files_parallel(
    args: argparse.Namespace,
    remote_files: list[tuple[RemoteItem, RemoteFileInfo]],
) -> int:
    task_queue: queue.Queue[tuple[int, RemoteItem, RemoteFileInfo]] = queue.Queue()
    for index, (item, remote_file) in enumerate(remote_files, start=1):
        task_queue.put((index, item, remote_file))

    lock = threading.Lock()
    stats = {
        "processed": 0,
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "copied_bytes": 0,
        "skipped_bytes": 0,
    }

    def worker(worker_id: int) -> None:
        client, sftp = connect(args)
        logging.info("Worker %d connected", worker_id)
        try:
            while True:
                try:
                    index, item, remote_file = task_queue.get_nowait()
                except queue.Empty:
                    return

                local_path = local_for(remote_file, item, args.dest)
                size = int(remote_file.size)
                status, error, client, sftp = download_one_with_retries(args, client, sftp, remote_file, local_path)
                with lock:
                    stats["processed"] += 1
                    processed = stats["processed"]
                    if error is None and status == "skip":
                        stats["skipped"] += 1
                        stats["skipped_bytes"] += size
                    elif error is None:
                        stats["downloaded"] += 1
                        stats["copied_bytes"] += size
                    else:
                        stats["failed"] += 1
                        logging.error(
                            "Failed permanently: %s",
                            remote_file.path,
                            exc_info=(type(error), error, error.__traceback__),
                        )

                    should_log = (
                        processed == 1
                        or processed % 25 == 0
                        or size >= 10 * 1024 * 1024
                        or status == "download"
                    )
                    if should_log:
                        logging.info(
                            "[%d/%d] worker=%d source_index=%d %s %s (%s)",
                            processed,
                            len(remote_files),
                            worker_id,
                            index,
                            status or "failed",
                            remote_file.path,
                            readable_size(size),
                        )
                task_queue.task_done()
        finally:
            try:
                sftp.close()
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass
            logging.info("Worker %d stopped", worker_id)

    worker_count = max(1, args.workers)
    threads = [
        threading.Thread(target=worker, args=(worker_id,), name=f"sftp-worker-{worker_id}", daemon=False)
        for worker_id in range(1, worker_count + 1)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    logging.info(
        "Done. downloaded=%d (%s), skipped=%d (%s), failed=%d",
        stats["downloaded"],
        readable_size(stats["copied_bytes"]),
        stats["skipped"],
        readable_size(stats["skipped_bytes"]),
        stats["failed"],
    )
    return 1 if stats["failed"] else 0


def main() -> int:
    args = parse_args()
    log_path = configure_logging(args.dest)
    logging.info("Destination: %s", args.dest)
    logging.info("Log file: %s", log_path)

    items = list(FIRST_PASS_ITEMS)
    if args.include_raw:
        items.extend(OPTIONAL_RAW_ITEMS)

    client, sftp = connect(args)
    try:
        remote_files: list[tuple[RemoteItem, RemoteFileInfo]] = []
        for item in items:
            try:
                item_files = list_remote_files_fast(client, item)
                remote_files.extend((item, remote_file) for remote_file in item_files)
                logging.info("Listed %d files under %s", len(item_files), item.remote)
            except FileNotFoundError:
                message = f"Remote path missing: {item.remote}"
                if item.required:
                    raise FileNotFoundError(message)
                logging.warning(message)

        total_size = sum(int(remote_file.size) for _, remote_file in remote_files)
        logging.info("Files to consider: %d", len(remote_files))
        logging.info("Remote byte total: %s", readable_size(total_size))
        if args.dry_run:
            for item, remote_file in remote_files[:200]:
                logging.info(
                    "%s -> %s (%s)",
                    remote_file.path,
                    local_for(remote_file, item, args.dest),
                    readable_size(int(remote_file.size)),
                )
            if len(remote_files) > 200:
                logging.info("... %d more files", len(remote_files) - 200)
            return 0

        if args.workers > 1:
            sftp.close()
            client.close()
            return download_files_parallel(args, remote_files)
        return download_files_sequential(args, client, sftp, remote_files)
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
