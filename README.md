# HeartByte
Term_Project_2-2

Interactive heartbeat signal processing and ECG simulation studio, built with Streamlit.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/diBya990/HeartByte.git
cd HeartByte
```

### 2. (Optional) Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install streamlit plotly numpy scipy pandas librosa soundfile scikit-learn joblib matplotlib reportlab
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at http://localhost:8501. Run the command from the project folder so the app can find the `archive/` recordings and the `models/` folder.

### 5. (Optional) Retrain the heartbeat classifier

The trained model is already included in `models/heartbeat_classifier.joblib`. To retrain it from the recordings in `archive/`:

```bash
python train_heartbeat_classifier.py
```
