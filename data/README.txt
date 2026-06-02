This directory is NOT tracked by git.

What goes here:
  data/
  ├── MNIST/                      ← Auto-downloaded by PyTorch (torchvision.datasets.MNIST)
  │   └── raw/                    ← IDX binary files: train/test images and labels
  └── mini_speech_commands/       ← Auto-downloaded by train_export.py on first KWS run
      ├── down/                   ← 1000 x 1-second .wav recordings of "down"
      ├── go/                     ← 1000 x 1-second .wav recordings of "go"
      ├── left/                   ← 1000 x 1-second .wav recordings of "left"
      ├── no/                     ← 1000 x 1-second .wav recordings of "no"
      ├── right/                  ← 1000 x 1-second .wav recordings of "right"
      ├── stop/                   ← 1000 x 1-second .wav recordings of "stop"
      ├── up/                     ← 1000 x 1-second .wav recordings of "up"
      └── yes/                    ← 1000 x 1-second .wav recordings of "yes"

How to populate:
  Run `uv run bash scripts/run_tests.sh` from the project root.
  Both datasets will be downloaded automatically on the first run.

  Sources:
    MNIST             → torchvision (auto via train_export.py --model mnist)
    Mini Speech Cmds  → http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip
