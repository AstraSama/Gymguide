# Gymguide

extensions:
    python;
    jupyter;
    black formatter;
    flake8:
        configurar arg:
            --max-line-length=100;
            --exclude=.venv,get-pip.py,test_mediapipe_pose.py, test_video_mediapipe_pose.py, pose_estimator.py;
    pylance;
    python debugger;

miniconda install guide:
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    source ~/.bashrc
    conda create -n nome_projeto python=3.9
    conda activate nome_projeto
    pip install -r requirements.txt
    python3 src/main.py caminho/para/video.mp4
    conda deactivate

dados rotulados, como usar:
    Pressione b para salvar um exemplo bom no CSV
    Pressione r para salvar um exemplo ruim
    Pressione q para sair