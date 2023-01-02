'''
Adapted from https://github.com/universome/HDTF/blob/main/download.py

Usage:
```
$ python download.py --src_file /path/to/data.csv
                     --out_dir /path/to/save/video/files 
                     --num_workers 8
```

Requirements:
- tqdm
- youtube-dl
'''

import os
import argparse
from typing import List, Dict
from multiprocessing import Pool
import subprocess

from tqdm import tqdm
import pandas as pd

def download(src_file: str, out_dir: str, num_workers: int, **process_video_kwargs):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, '_videos_raw'), exist_ok=True)

    src_data = pd.read_csv(src_file)

    download_queue = construct_download_queue(src_data)
    task_kwargs = [dict(
        video_data=vd,
        output_dir=out_dir,
        **process_video_kwargs,
     ) for vd in download_queue]
    pool = Pool(processes=num_workers)
    tqdm_kwargs = dict(total=len(task_kwargs), desc=f'Downloading videos into {out_dir} (note: without sound)')

    for _ in tqdm(pool.imap_unordered(task_proxy, task_kwargs), **tqdm_kwargs):
        pass

    print('Download is finished, you can now (optionally) delete the following directories, since they are not needed anymore and occupy a lot of space:')
    print(' -', os.path.join(out_dir, '_videos_raw'))


def construct_download_queue(src_data: pd.DataFrame) -> List[Dict]:
    download_queue = []

    for speaker, speaker_data in src_data.groupby('Speaker'):
        clip_id = 0
        for video, interval_data in speaker_data.groupby('URL'):
            intervals = []
            for i, row in interval_data.iterrows():
                intervals.append((row['Start'], row['End']))
            download_queue.append({
                'name': f"TR_{''.join([x[0].upper() for x in speaker.split(' ')])}_{clip_id}",
                'url': video,
                'intervals': intervals
            })
            clip_id += 1

    return download_queue


def task_proxy(kwargs):
    return download_and_process_video(**kwargs)


def download_and_process_video(video_data: Dict, output_dir: str):
    """
    Downloads the video and cuts/crops it into several ones according to the provided time intervals
    """
    raw_download_path = os.path.join(output_dir, '_videos_raw', f"{video_data['name']}.mp4")
    raw_download_log_file = os.path.join(output_dir, '_videos_raw', f"{video_data['name']}_download_log.txt")
    download_result = download_video(video_data['url'], raw_download_path, log_file=raw_download_log_file)

    if not download_result:
        print('Failed to download', video_data)
        print(f'See {raw_download_log_file} for details')
        return

    for i, (start, end) in enumerate(video_data['intervals']):
        clip_name = f'{video_data["name"]}_{i:03d}'
        clip_path = os.path.join(output_dir, clip_name + '.mp4')
        crop_success = cut_and_crop_video(raw_download_path, clip_path, start, end)

        if not crop_success:
            print(f'Failed to cut-and-crop clip #{i}', video_data)
            continue


def download_video(video_id, download_path, log_file=None):
    """
    Download video from YouTube.
    :param video_id:        YouTube ID of the video.
    :param download_path:   Where to save the video.
    :param log_file:        Path to a log file for youtube-dl.
    :return:                A bool indicating success.

    Copy-pasted from https://github.com/ytdl-org/youtube-dl
    """
    if os.path.exists(download_path):
        return True

    if log_file is None:
        stderr = subprocess.DEVNULL
    else:
        stderr = open(log_file, "a")
    command = [
        "youtube-dl",
        video_id, "--quiet", "-f",
        f"bestvideo[ext=mp4]+bestaudio[ext=m4a]",
        "--output", download_path,
        "--no-continue"
    ]
    return_code = subprocess.call(command, stderr=stderr)
    success = return_code == 0

    if log_file is not None:
        stderr.close()

    return success


def cut_and_crop_video(raw_video_path, output_path, start, end):
    command = ' '.join([
        "ffmpeg", "-i", raw_video_path,
        "-strict", "-2", # Some legacy arguments
        "-loglevel", "quiet", # Verbosity arguments
        "-qscale", "0", # Preserve the quality
        "-y", # Overwrite if the file exists
        "-ss", str(start), "-to", str(end), # Cut arguments
        output_path
    ])

    return_code = subprocess.call(command, shell=True)
    success = return_code == 0

    if not success:
        print('Command failed:', command)

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from youtube")
    parser.add_argument('-s', '--src_file', type=str, help='Path to the directory of the csv file')
    parser.add_argument('-o', '--out_dir', type=str, help='Save directory for downloaded and cropped videos')
    parser.add_argument('-w', '--num_workers', type=int, default=8, help='Number of workers for downloading')
    args = parser.parse_args()

    download(
        args.src_file,
        args.out_dir,
        args.num_workers,
    )
