import yt_dlp

url = 'https://youtu.be/D9tP-Pcvqbc?si=V9VPlE0IkV2m4zZ4'

# Baixar apenas o melhor formato de vídeo disponível (sem áudio)
ydl_opts = {
    'format': 'bv*[ext=mp4]',  # bv = bestvideo, apenas vídeo
    'outtmpl': 'C:\Users\fe_li\Documents\Gymguide\data\sample_videos\pushup_analyze_8_workout.mp4',
    'noplaylist': True,
    'quiet': False,
    'no_warnings': True,
    'skip_download': False
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])