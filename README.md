Therapy Session Helper – Multi-Modal Analysis

A comprehensive emotional analysis tool that uses video, audio, and text to provide insights during therapy sessions.

⭐ Features
🎥 Video Analysis

Real-time facial expression recognition.

🎵 Audio Analysis

Voice pattern and speech content analysis.

📝 Text Analysis

Depression severity prediction from written text.

💬 Complete Session

Multi-modal analysis combining all inputs for deeper insights.

🚀 Quick Start
Prerequisites

Python 3.8+

Webcam (for video analysis)

Microphone (for audio analysis)

🔧 Installation

1. Clone or download the project.
2. Create a virtual environment (recommended)
   python -m venv therapy_env

# On Windows:

therapy_env\Scripts\activate

# On Mac/Linux:

source therapy_env/bin/activate

3. Install required packages
   pip install streamlit pandas numpy plotly opencv-python tensorflow pillow librosa googletrans langdetect scikit-learn

4. (Optional) Install audio recording support

Windows

pip install pyaudio

Mac

brew install portaudio
pip install pyaudio

Linux

sudo apt-get install python3-pyaudio

▶️ Running the Application

Ensure you're in the project directory containing:

app.py
facial_analyzer.py
depression_text_predictor.py
audio_depression_detector.py

Run:

streamlit run app.py

Open the displayed URL (usually http://localhost:8501
).

📘 Usage Guide
📝 Text Analysis

Select Text Analysis in the sidebar

Type/paste your text

Click Analyze Text

View depression severity results

🎥 Video Analysis

Select Video Analysis

Choose a therapy question

Click Start Video Recording

Speak to the camera

Press Q to stop

View facial expression analysis

🎵 Audio Analysis

Select Audio Analysis

Click Start Recording

Speak normally

Click Stop Recording

View audio-based depression results

💬 Complete Session

Select Complete Session

Follow the combined video/audio prompts

Receive multi-modal analysis output

📂 File Structure
therapy-app/
├── app.py # Main application
├── facial_analyzer.py # Video analysis module
├── depression_text_predictor.py # Text analysis module
├── audio_depression_detector.py # Audio analysis module
└── requirements.txt # Dependencies

🤖 Model Files

The text analysis component checks for:

depression_model.h5

tokenizer.pkl

label_encoder.pkl

If they are missing, the system uses demo mode automatically.

🛠 Troubleshooting
Webcam not working

Ensure no other app is using the camera

Check browser permissions

Audio not recording

Microphone permission required

Demo mode will automatically activate if needed

Import errors

Reinstall dependencies

pip install --upgrade pip

TensorFlow warnings

Normal — they do not affect app functionality.

📌 Support

The application includes robust error handling.
If any module fails (video/audio/model loading), the app switches to demo mode so you can still use the core features.

⚠️ Disclaimer

This tool assists mental health professionals and is not a substitute for medical diagnosis or treatment.
Always consult qualified healthcare providers for medical concerns.
