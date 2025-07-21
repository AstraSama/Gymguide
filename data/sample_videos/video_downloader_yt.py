import yt_dlp

url = 'https://youtube.com/shorts/wKauOga6tZ4?si=J2DyuvhWHvhMb4Ot'

# Baixar apenas o melhor formato de vídeo disponível (sem áudio)
ydl_opts = {
    'format': 'bv*[ext=mp4]',  # bv = bestvideo, apenas vídeo
    'outtmpl': '/home/felippemart777/Documentos/atividades/GG/Gymguide/data/sample_videos/pushup_analyze_good_bad_3.mp4',
    'noplaylist': True,
    'quiet': False,
    'no_warnings': True,
    'skip_download': False
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])