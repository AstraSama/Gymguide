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