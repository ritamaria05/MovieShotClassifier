# MovieShotClassifier

Analyze a movie frame by shot size, framing, color and lighting. The demo runs locally.

## Setup

You need internet access, a browser, Python 3.12 with `venv` and `pip`, and at least 15 GB of free disk space. Unzip the project first and replace the example path below with the actual project folder.

### Linux

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
cd "/path/to/MovieShotClassifier"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r demo_requirements.txt
```

### macOS

Install [Python 3.12](https://www.python.org/downloads/), then run in Terminal:

```bash
cd "/path/to/MovieShotClassifier"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r demo_requirements.txt
```

### Windows

Install [Python 3.12](https://www.python.org/downloads/), then run in Command Prompt:

```cmd
cd /d "C:\path\to\MovieShotClassifier"
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -r demo_requirements.txt
```

Installation may take roughly **5 minutes**, depending on your connection and computer.


## Start the demo

With the environment active and the terminal still in the project folder:

```bash
python -m uvicorn scripts.demo.app:app --host 127.0.0.1 --port 8000
```

On the **first start**, allow roughly **5 minutes**: the app downloads approximately 4.9 GB of trained checkpoints, plus the pretrained backbone and image processor. Time varies substantially with connection speed. Wait for the message `Application startup complete` before opening http://127.0.0.1:8000.


## Analyze a frame

1. Open http://127.0.0.1:8000 in your browser.
2. Click the upload area or drag an image onto it, then click **Analyze**.
3. Use the **Shot size**, **Framing**, **Color**, and **Light** tabs to explore the results. On the model tabs, use **Model focus** to toggle the heatmap;
4. Choose **Change image** to analyze another frame.

Uploading a local image is usually quick, but analysis can take roughly **1-3 minutes on CPU**.


## Downloads

The five shot-size checkpoints and the framing checkpoint are cached and reused from [Hugging Face](https://huggingface.co/hugomelo17/MovieShotClassifier-models). Reference images are downloaded individually from the [Hugging Face dataset](https://huggingface.co/datasets/hugomelo17/MovieShotClassifier-dataset) and cached.

This design choice was made to reduce the program size, since models are 1GB each and the dataset is approximatelly 13GB.