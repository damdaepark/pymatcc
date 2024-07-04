#############################################
#  Author: Hongwei Fan                      #
#  E-mail: hwnorm@outlook.com               #
#  Homepage: https://github.com/hwfan       #
#############################################
from DriveDownloader.netdrives import get_session
from DriveDownloader.utils import judge_session, MultiThreadDownloader, judge_scheme
import argparse
import os
import sys
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

MAJOR_VERSION = 1
MINOR_VERSION = 6
POST_VERSION = 0
__version__ = f"{MAJOR_VERSION}.{MINOR_VERSION}.{POST_VERSION}"
console = Console(width=72)

multi_progress = Progress(
    TextColumn("[bold blue]Thread {task.fields[proc_id]}: ", justify="left"),
    BarColumn(bar_width=15),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "|",
    DownloadColumn(),
    "|",
    TransferSpeedColumn(),
    "|",
    TimeRemainingColumn(),
    refresh_per_second=10
)
url_scheme_env_key_map = {
        "http": "http_proxy",
        "https": "https_proxy",
}

def parse_args():
    parser = argparse.ArgumentParser(description='Drive Downloader Args')
    parser.add_argument('url', help='URL you want to download from.', default='', type=str)
    parser.add_argument('--filename', '-o', help='Target file name.', default='', type=str)
    parser.add_argument('--thread-number', '-n', help='thread number of multithread.', type=int, default=1)
    parser.add_argument('--version', '-v', action='version', version=__version__, help='Version.')
    parser.add_argument('--force-back-google','-F',help='Force to use the backup downloader for GoogleDrive.', action='store_true')
    args = parser.parse_args()
    return args

def get_env(key):
    value = os.environ.get(key)
    if not value or len(value) == 0:
        return None
    return value;

def download_single_file(url, filename="", thread_number=1, force_back_google=False, list_suffix=None):
    scheme = judge_scheme(url)
    if scheme not in url_scheme_env_key_map.keys():
        raise NotImplementedError(f"Unsupported scheme {scheme}")
    env_key = url_scheme_env_key_map[scheme]
    used_proxy = get_env(env_key)

    session_name = judge_session(url)
    session_func = get_session(session_name)
    google_fix_logic = False
    if session_name == 'GoogleDrive' and thread_number > 1 and not force_back_google:
        thread_number = 1
        google_fix_logic = True
    if thread_number > 1:
        progress_applied = multi_progress
    else:
        progress_applied = Progress(
            TextColumn("[bold blue]Downloading: ", justify="left"),
            BarColumn(bar_width=15),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "|",
            DownloadColumn(),
            "|",
            TransferSpeedColumn(),
            "|",
            TimeRemainingColumn(),
            refresh_per_second=10
        )
    download_session = session_func(used_proxy)
    download_session.connect(url, filename, force_backup=force_back_google if session_name == 'GoogleDrive' else False)
    final_filename = download_session.filename
    download_session.show_info(progress_applied, list_suffix)
    if google_fix_logic:
        console.print('[yellow]Warning: Google Drive URL detected. Only one thread will be created.')

    if thread_number > 1:
        download_session = MultiThreadDownloader(progress_applied, session_func, used_proxy, download_session.filesize, thread_number)
        interrupted = download_session.get(url, final_filename, force_back_google)
        if interrupted:
            return
        download_session.concatenate(final_filename)
    else:
        with progress_applied:
            task_id = progress_applied.add_task("download", filename=final_filename, proc_id=0, start=False)
            interrupted = download_session.save_response_content(progress_bar=progress_applied)
            if interrupted:
                return
    console.print('[green]Done.')
        
def download_filelist(args):
    lines = [line for line in open(args.url, 'r')]
    for line_idx, line in enumerate(lines):
        splitted_line = line.strip().split(" ")
        url, filename = splitted_line[0], splitted_line[1] if len(splitted_line) > 1 else ""
        thread_number = int(splitted_line[2]) if len(splitted_line) > 2 else 1
        list_suffix = "({:d}/{:d})".format(line_idx+1, len(lines))
        download_single_file(url, filename, thread_number, args.force_back_google, list_suffix)

def simple_cli(url, filename):
    download_single_file(url, filename)

if __name__ == '__main__':
    simple_cli()
