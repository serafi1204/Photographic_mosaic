import yt_dlp
import cv2
import os

def getYoutubeScreenshot(url, output_dir='', tag='youtube_screenshot', N = 10):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_url = info_dict['url']
        duration = info_dict['duration']
        video_id = info_dict['id']  # YouTube 고유 ID

    interval = duration / (N+1)

    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)

    for i in range(N):
        time_in_seconds = interval * (i+1)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
        success, image = cap.read()

        if success:
            # 시간대를 정수 또는 소수점으로 표현 가능 (여기선 정수 초로 사용)
            time_label = int(time_in_seconds)
            filename = os.path.join(output_dir, f"{video_id}&t={time_label}.jpg")
            cv2.imwrite(filename, image)
            print(f"..{i+1}", end ='')
        else:
            print(f"Failed to capture frame at {time_in_seconds:.2f}s")

    cap.release()
