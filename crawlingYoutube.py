import photographic_mosaic as pm
import json
from datetime import datetime 
import os

channel = {
    "hane": "https://www.youtube.com/@하네다시보기",
    "onharu": "https://www.youtube.com/@off_haru",
    "kimate": "https://www.youtube.com/@김뒷태의이중생활",
    "otonosori": "https://www.youtube.com/@소리다시보기"
}

with open('up_to_date.json', 'r') as f:
    up2date = json.load(f)

playlist = {}
for key, url in channel.items():
    playlist[key] = pm.getYoutubePlaylist(url)

playlist_filted = {}
for key, value in playlist.items():
    last_video = up2date['crawling'][key]
    playlist_filted[key] = []

    for video in value:
        if last_video == video['url']: break
        
        playlist_filted[key].append(video)

for i, (key, value) in enumerate(playlist_filted.items()):
    os.system('cls')
    print(f"steamer: {key} ({i+1}/4)")

    path = f'source_youtube/{key}'
    for j, video in enumerate(value[::-1]):
        print('\r' + f'process {j/len(value)*100:.1f}% ({j}/{len(value)})', end='')
        pm.getYoutubeScreenshot(video['url'], path)

        up2date['crawling'][key] = video['url']

        with open('up_to_date.json', 'w') as f:
            json.dump(up2date, f)