# app.py (COMPLETE UPDATED VERSION WITH PROPER GREEN DARK THEME)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import time
from datetime import datetime

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Therapy Session Helper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback basic styling
        st.markdown("""
        <style>
        .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        .stApp {background-color: #0a0a0a; color: white;}
        h1, h2, h3 {color: #00ff88 !important;}
        .stButton>button {background-color: #00ff88; color: black;}
        </style>
        """, unsafe_allow_html=True)

def safe_import(module_name, class_name):
    """Safely import modules with detailed error handling"""
    try:
        module = __import__(module_name, fromlist=[class_name])
        class_obj = getattr(module, class_name)
        return class_obj, True
    except ImportError as e:
        st.warning(f"❌ Could not import {module_name}: {e}")
        return None, False
    except AttributeError as e:
        st.warning(f"❌ Could not find {class_name} in {module_name}: {e}")
        return None, False
    except Exception as e:
        st.warning(f"❌ Unexpected error importing {module_name}.{class_name}: {e}")
        return None, False

# Import available components
FacialExpressionAnalyzer, FACE_ANALYSIS_AVAILABLE = safe_import('facial_analyzer', 'FacialExpressionAnalyzer')
DepressionSeverityPredictor, TEXT_ANALYSIS_AVAILABLE = safe_import('depression_text_predictor', 'DepressionSeverityPredictor')

# Try to import our simplified audio detector
try:
    from audio_depression_detector import AudioDepressionDetector
    AUDIO_ANALYSIS_AVAILABLE = True
except ImportError as e:
    st.warning(f"❌ Could not import audio_depression_detector: {e}")
    AUDIO_ANALYSIS_AVAILABLE = False

# Initialize predictors with caching
@st.cache_resource
def load_predictor():
    if not TEXT_ANALYSIS_AVAILABLE:
        return None
    try:
        predictor = DepressionSeverityPredictor()
        return predictor
    except Exception as e:
        st.error(f"❌ Error loading text analysis model: {e}")
        return None

@st.cache_resource
def load_facial_analyzer():
    if not FACE_ANALYSIS_AVAILABLE:
        return None
    try:
        analyzer = FacialExpressionAnalyzer()
        return analyzer
    except Exception as e:
        st.error(f"❌ Error loading facial analyzer: {e}")
        return None

@st.cache_resource
def load_audio_detector():
    if not AUDIO_ANALYSIS_AVAILABLE:
        return None
    try:
        detector = AudioDepressionDetector()
        return detector
    except Exception as e:
        st.error(f"❌ Error loading audio detector: {e}")
        return None

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        # Navigation
        'current_page': 'main',
        
        # Video analysis states
        'video_recording': False,
        'video_results': None,
        'video_analyzing': False,
        'video_recording_started': False,
        'video_stop_clicked': False,
        'recording_start_time': None,
        
        # Video upload states
        'video_upload_analyzing': False,
        'video_upload_processed': False,
        'uploaded_video_file': None,
        
        # Audio analysis states
        'audio_recording': False,
        'audio_results': None,
        'audio_analyzing': False,
        'audio_recording_started': False,
        'audio_stop_clicked': False,
        
        # Audio upload states
        'audio_upload_analyzing': False,
        'audio_upload_processed': False,
        'uploaded_audio_file': None,
        
        # Complete session states
        'complete_session_active': False,
        'complete_session_results': None,
        'complete_session_analyzing': False,
        'complete_session_mode': 'record',  # 'record' or 'upload'
        'complete_session_video_file': None,
        'complete_session_audio_file': None,
        'complete_session_upload_processed': False,
        
        # General states
        'last_action': None,
        'action_timestamp': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    # Load custom CSS
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #1a1a1a;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.header("🧠 Therapy Session Helper")
        st.markdown("---")
        
        # Navigation options
        nav_options = {
            "🏠 Main Dashboard": "main",
            "📝 Text Analysis": "text",
            "🎥 Video Analysis": "video", 
            "🎵 Audio Analysis": "audio",
            "💬 Complete Session": "complete"
        }
        
        # Create navigation
        for label, page in nav_options.items():
            if st.button(label, use_container_width=True, 
                        type="primary" if st.session_state.current_page == page else "secondary"):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("---")
        
        # Model status
        st.subheader("Model Status")
        text_predictor = load_predictor()
        facial_analyzer = load_facial_analyzer()
        audio_detector = load_audio_detector()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("📝" if text_predictor else "❌")
        with col2:
            st.markdown("🎥" if facial_analyzer else "❌")
        with col3:
            st.markdown("🎵" if audio_detector else "❌")
    
    # Page routing
    if st.session_state.current_page == "main":
        show_main_dashboard()
    elif st.session_state.current_page == "text":
        text_analysis(load_predictor())
    elif st.session_state.current_page == "video":
        video_analysis(load_facial_analyzer())
    elif st.session_state.current_page == "audio":
        audio_analysis(load_audio_detector())
    elif st.session_state.current_page == "complete":
        complete_session(load_facial_analyzer(), load_audio_detector())

def show_main_dashboard():
    """Main dashboard with app explanation and feature navigation"""
    
    # Header section
    col1 = st.columns([1])[0]
    with col1:
        st.title("🧠 Therapy Session Helper")
        st.markdown("### Multi-Modal Mental Health Analysis Platform")
        
    st.markdown("---")
    
    # App description
    st.markdown("""
    <div style="background-color: #1a1a1a; border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; margin: 1rem 0;">
    <h3 style="color: #00ff88; margin-bottom: 1rem;">🌟 About This Application</h3>
    <p style="color: #ffffff; margin-bottom: 1rem;">This advanced therapeutic tool uses artificial intelligence to analyze multiple modalities of emotional expression, 
    providing comprehensive insights into mental well-being through state-of-the-art machine learning models.</p>
    
    <p style="color: #ffffff; margin-bottom: 0.5rem;"><strong>🔒 Privacy First:</strong> All analysis happens locally on your device - your data never leaves your computer.</p>
    <p style="color: #ffffff; margin-bottom: 0.5rem;"><strong>🧠 Science Backed:</strong> Based on established psychological assessment methods and clinical research.</p>
    <p style="color: #ffffff; margin-bottom: 0;"><strong>💡 Supportive Tool:</strong> Designed to complement professional care, not replace it.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.subheader("🎯 Available Analysis Methods")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a1a 0%, #1a2a1a 100%); border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; height: 100%; transition: all 0.3s ease;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">📝 Text Analysis</h4>
        <p style="color: #b0b0b0; margin-bottom: 1rem;">Analyze written emotional content using advanced NLP models to detect depression patterns in text.</p>
        <small style="color: #b0b0b0;"><strong>Best for:</strong> Journal entries, written thoughts</small>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Text Analysis", key="nav_text", use_container_width=True):
            st.session_state.current_page = "text"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a1a 0%, #1a2a1a 100%); border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; height: 100%; transition: all 0.3s ease;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">🎥 Video Analysis</h4>
        <p style="color: #b0b0b0; margin-bottom: 1rem;">Facial expression analysis using computer vision to detect emotional states from video recordings.</p>
        <small style="color: #b0b0b0;"><strong>Best for:</strong> Facial emotion tracking</small>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Video Analysis", key="nav_video", use_container_width=True):
            st.session_state.current_page = "video"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a1a 0%, #1a2a1a 100%); border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; height: 100%; transition: all 0.3s ease;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">🎵 Audio Analysis</h4>
        <p style="color: #b0b0b0; margin-bottom: 1rem;">Voice pattern analysis detecting depression indicators through acoustic features and speech content.</p>
        <small style="color: #b0b0b0;"><strong>Best for:</strong> Voice recordings, speech patterns</small>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Audio Analysis", key="nav_audio", use_container_width=True):
            st.session_state.current_page = "audio"
            st.rerun()
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a1a 0%, #1a2a1a 100%); border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; height: 100%; transition: all 0.3s ease;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">💬 Complete Session</h4>
        <p style="color: #b0b0b0; margin-bottom: 1rem;">Comprehensive multi-modal analysis combining video, audio, and contextual data for holistic assessment.</p>
        <small style="color: #b0b0b0;"><strong>Best for:</strong> Complete emotional assessment</small>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Complete Session", key="nav_complete", use_container_width=True):
            st.session_state.current_page = "complete"
            st.rerun()
    
    # How it works section
    st.markdown("---")
    st.subheader("🔬 How Each Analysis Works")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text", "🎥 Video", "🎵 Audio", "💬 Complete"])
    
    with tab1:
        st.markdown("""
        <div style="background-color: #1a1a1a; border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; margin: 1rem 0;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">Text Analysis Pipeline</h4>
        <ol style="color: #ffffff;">
        <li style="margin-bottom: 0.5rem;"><strong>Input:</strong> User provides written text about their feelings</li>
        <li style="margin-bottom: 0.5rem;"><strong>Processing:</strong> NLP model analyzes linguistic patterns</li>
        <li style="margin-bottom: 0.5rem;"><strong>Detection:</strong> Identifies depression indicators in:
            <ul style="margin-top: 0.5rem;">
            <li>Word choice and sentiment</li>
            <li>Sentence structure complexity</li>
            <li>Emotional tone and themes</li>
            </ul>
        </li>
        <li style="margin-bottom: 0;"><strong>Output:</strong> Severity assessment with confidence scores</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div style="background-color: #1a1a1a; border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; margin: 1rem 0;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">Video Analysis Pipeline</h4>
        <ol style="color: #ffffff;">
        <li style="margin-bottom: 0.5rem;"><strong>Input:</strong> Live recording or uploaded video</li>
        <li style="margin-bottom: 0.5rem;"><strong>Processing:</strong> Computer vision detects facial landmarks</li>
        <li style="margin-bottom: 0.5rem;"><strong>Analysis:</strong> Emotion classification based on:
            <ul style="margin-top: 0.5rem;">
            <li>Facial muscle movements</li>
            <li>Micro-expressions</li>
            <li>Emotion distribution over time</li>
            </ul>
        </li>
        <li style="margin-bottom: 0;"><strong>Output:</strong> Emotional state profile and insights</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div style="background-color: #1a1a1a; border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; margin: 1rem 0;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">Audio Analysis Pipeline</h4>
        <ol style="color: #ffffff;">
        <li style="margin-bottom: 0.5rem;"><strong>Input:</strong> Voice recording or audio file</li>
        <li style="margin-bottom: 0.5rem;"><strong>Processing:</strong> Audio feature extraction and speech recognition</li>
        <li style="margin-bottom: 0.5rem;"><strong>Analysis:</strong> Dual analysis of:
            <ul style="margin-top: 0.5rem;">
            <li><strong>Acoustic Features:</strong> Pitch, tone, speech rate, pauses</li>
            <li><strong>Semantic Content:</strong> Speech content and themes</li>
            </ul>
        </li>
        <li style="margin-bottom: 0;"><strong>Output:</strong> Comprehensive audio-based assessment</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div style="background-color: #1a1a1a; border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; margin: 1rem 0;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">Complete Session Pipeline</h4>
        <ol style="color: #ffffff;">
        <li style="margin-bottom: 0.5rem;"><strong>Input:</strong> Combined video and audio recording</li>
        <li style="margin-bottom: 0.5rem;"><strong>Processing:</strong> Synchronized multi-modal analysis</li>
        <li style="margin-bottom: 0.5rem;"><strong>Integration:</strong> Combines insights from:
            <ul style="margin-top: 0.5rem;">
            <li>Facial expression analysis</li>
            <li>Voice pattern analysis</li>
            <li>Speech content analysis</li>
            </ul>
        </li>
        <li style="margin-bottom: 0;"><strong>Output:</strong> Holistic emotional assessment with cross-validation</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Usage guidelines
    st.markdown("---")
    st.subheader("📋 Usage Guidelines")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #1a1a1a; border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; margin: 1rem 0;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">✅ Best Practices</h4>
        <ul style="color: #ffffff;">
        <li>Use in quiet, well-lit environments</li>
        <li>Speak naturally and authentically</li>
        <li>Allow genuine emotional expression</li>
        <li>Record for sufficient duration (1-3 minutes)</li>
        <li>Use regularly for tracking over time</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #1a1a1a; border: 1px solid #00cc6a; border-radius: 12px; padding: 2rem; margin: 1rem 0;">
        <h4 style="color: #00ff88; margin-bottom: 1rem;">⚠️ Important Notes</h4>
        <ul style="color: #ffffff;">
        <li>This tool is for support, not diagnosis</li>
        <li>Consult professionals for clinical assessment</li>
        <li>Results are indicators, not definitive</li>
        <li>Privacy is maintained - data stays local</li>
        <li>Use as part of comprehensive self-care</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def text_analysis(predictor):
    """Text-based depression analysis"""
    st.header("📝 Text Analysis")
    
    # Add navigation back to main
    if st.button("← Back to Main", key="text_back"):
        st.session_state.current_page = "main"
        st.rerun()
    
    if predictor is None:
        st.error("❌ Text analysis model not available")
        return
    
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Describe how you're feeling or paste text to analyze...",
        key="text_input"
    )
    
    if st.button("Analyze Text", type="primary", key="analyze_text") and text_input:
        with st.spinner("Analyzing text content..."):
            try:
                # Use the available method
                if hasattr(predictor, 'predict_severity'):
                    result = predictor.predict_severity(text_input)
                elif hasattr(predictor, 'predict'):
                    result = predictor.predict([text_input])[0]
                else:
                    st.error("No prediction method available")
                    return
                
                display_text_results(result)
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")

def video_analysis(facial_analyzer):
    """Video-based emotion analysis with manual recording"""
    st.header("🎥 Video Emotion Analysis")
    
    # Add navigation back to main
    if st.button("← Back to Main", key="video_back"):
        st.session_state.current_page = "main"
        st.rerun()
    
    if facial_analyzer is None:
        st.error("❌ Video analysis not available")
        return
    
    # Display usage instructions
    st.subheader("📋 How to Use Video Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **🎯 Step-by-Step Guide:**
        
        1. **🎬 Start Recording** - Click below to begin video capture
        2. **🗣️ Speak Naturally** - Express your feelings to the camera
        3. **⏹️ Stop Recording** - Click to end and analyze
        4. **📊 View Results** - See facial expression analysis
        
        **💡 Tips for Best Results:**
        - Ensure good lighting on your face
        - Position yourself clearly in frame
        - Speak naturally about your feelings
        - Allow natural facial expressions
        - Record for at least 30 seconds for best analysis
        """)
        
        st.subheader("Recording Controls")
        
        # Handle recording state transitions
        handle_video_recording_states(facial_analyzer)
        
        # Display current status
        display_video_status()
        
    with col2:
        st.subheader("Status")
        display_video_current_status()
        
        st.subheader("Analysis Output")
        st.markdown("""
        - Facial expression percentages
        - Dominant emotions detected
        - Emotional insights
        - Expression timeline
        """)
        
        # Video upload option
        st.subheader("📁 Upload Video")
        handle_video_upload(facial_analyzer)
    
    # Display recording status and instructions
    if st.session_state.video_recording:
        display_recording_interface()
    
    # Display results if available
    if st.session_state.video_results and not st.session_state.video_analyzing:
        display_video_final_results()

def handle_video_recording_states(facial_analyzer):
    """Handle video recording state transitions"""
    # Start Recording
    if not st.session_state.video_recording and not st.session_state.video_analyzing:
        if st.button("🎬 Start Video Recording", type="primary", use_container_width=True, key="start_video_recording"):
            try:
                result = facial_analyzer.start_video_recording()
                if isinstance(result, dict) and result.get('status') == 'recording_started':
                    st.session_state.video_recording = True
                    st.session_state.recording_start_time = time.time()
                    st.session_state.video_recording_started = True
                    st.session_state.last_action = "start_recording"
                    st.session_state.action_timestamp = time.time()
                    st.rerun()
                else:
                    st.error("❌ Could not start video recording")
            except Exception as e:
                st.error(f"❌ Error starting recording: {e}")
    
    # Stop Recording
    elif st.session_state.video_recording and not st.session_state.video_analyzing:
        if st.button("⏹️ Stop Video Recording", type="secondary", use_container_width=True, key="stop_video_recording"):
            st.session_state.video_recording = False
            st.session_state.video_analyzing = True
            st.session_state.video_stop_clicked = True
            st.session_state.last_action = "stop_recording"
            st.session_state.action_timestamp = time.time()
            st.rerun()
    
    # Handle analysis after stop
    if st.session_state.video_stop_clicked and st.session_state.video_analyzing:
        st.session_state.video_stop_clicked = False
        
        with st.spinner("🔄 Processing video frames and detecting emotions..."):
            try:
                emotion_results, video_path = facial_analyzer.stop_video_recording()
                if emotion_results is not None:
                    st.session_state.video_results = {
                        'emotions': emotion_results,
                        'video_path': video_path
                    }
                else:
                    st.error("❌ No emotion data collected from video")
            except Exception as e:
                st.error(f"❌ Error during video analysis: {e}")
            finally:
                st.session_state.video_analyzing = False
                st.rerun()

def display_video_status():
    """Display current video recording status"""
    if st.session_state.video_recording:
        st.info("🔴 **Recording in progress** - Speak naturally about your feelings")
    elif st.session_state.video_analyzing:
        st.warning("🔄 **Analyzing video** - Please wait...")
    else:
        st.success("✅ **Ready to record**")

def display_video_current_status():
    """Display current status in the right column"""
    if st.session_state.video_recording:
        st.error("🔴 **Recording Active**")
        st.write("A video window is open. Speak naturally about your feelings.")
    elif st.session_state.video_analyzing:
        st.warning("🔄 **Analyzing Video**")
        st.write("Processing recorded video...")
    else:
        st.success("✅ **Ready to Record**")

def handle_video_upload(facial_analyzer):
    """Handle video file upload and analysis - FIXED VERSION"""
    uploaded_video = st.file_uploader("Or upload existing video", 
                                    type=['mp4', 'avi', 'mov'],
                                    help="Upload MP4, AVI, or MOV files for analysis",
                                    key="video_uploader")
    
    # Store uploaded file in session state
    if uploaded_video and uploaded_video != st.session_state.get('uploaded_video_file'):
        st.session_state.uploaded_video_file = uploaded_video
        st.session_state.video_upload_processed = False
    
    # Only show analyze button if we have a file that hasn't been processed
    if (st.session_state.uploaded_video_file and 
        not st.session_state.video_analyzing and 
        not st.session_state.video_upload_analyzing and
        not st.session_state.video_upload_processed):
        
        if st.button("🎭 Analyze Uploaded Video", use_container_width=True, key="analyze_uploaded_video"):
            st.session_state.video_upload_analyzing = True
            st.session_state.video_upload_processed = False
            st.rerun()
    
    # Handle uploaded video analysis - ONLY when explicitly triggered and not already processed
    if (st.session_state.video_upload_analyzing and 
        st.session_state.uploaded_video_file and
        not st.session_state.video_upload_processed):
        
        # Mark as processing immediately to prevent re-triggering
        st.session_state.video_upload_processed = True
        
        with st.spinner(f"🔄 Analyzing {st.session_state.uploaded_video_file.name}..."):
            try:
                emotion_results = facial_analyzer.analyze_uploaded_video(st.session_state.uploaded_video_file)
                st.session_state.video_results = {
                    'emotions': emotion_results,
                    'video_path': "uploaded_file"
                }
            except Exception as e:
                st.error(f"❌ Error analyzing video: {e}")
            finally:
                st.session_state.video_upload_analyzing = False
                st.rerun()

def display_recording_interface():
    """Display recording interface when recording is active"""
    st.markdown("---")
    st.subheader("🎥 Live Recording Active")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **In the video window:**
        - Look directly at the camera
        - Speak naturally about your feelings
        - Express emotions genuinely
        - Continue for 30+ seconds
        """)
    with col2:
        st.markdown("""
        **Current Analysis:**
        - Real-time face detection
        - Live emotion tracking
        - Expression monitoring
        - Voice recording (if available)
        """)
    
    # Recording timer
    if st.session_state.recording_start_time:
        recording_time = time.time() - st.session_state.recording_start_time
        mins, secs = divmod(int(recording_time), 60)
        st.metric("Recording Duration", f"{mins:02d}:{secs:02d}")

def display_video_final_results():
    """Display final video analysis results"""
    st.markdown("---")
    display_video_results(
        st.session_state.video_results['emotions'],
        "Facial Expression Analysis"
    )
    
    # Additional insights based on results
    emotions = st.session_state.video_results['emotions']
    st.subheader("🧠 Emotional Insights")
    
    if emotions.get('sad', 0) > 25:
        st.warning("**Elevated Sadness**: Higher than typical sadness levels detected in facial expressions")
    elif emotions.get('sad', 0) > 15:
        st.info("**Moderate Sadness**: Some sadness indicators present")
    
    if emotions.get('happy', 0) > 40:
        st.success("**Positive Expression**: Strong positive emotional expressions detected")
    elif emotions.get('happy', 0) > 25:
        st.info("**Balanced Expression**: Good mix of positive expressions")
    
    if emotions.get('neutral', 0) > 70:
        st.info("**Neutral Dominance**: Predominantly neutral emotional expressions")
    
    if emotions.get('fear', 0) > 10 or emotions.get('angry', 0) > 10:
        st.warning("**Elevated Negative Emotions**: Increased fear or anger expressions detected")
    
    # Add a button to clear results and start over
    if st.button("🔄 New Video Analysis", type="secondary", key="new_video_analysis"):
        reset_video_state()
        st.rerun()

def reset_video_state():
    """Reset all video-related session states"""
    st.session_state.video_results = None
    st.session_state.video_recording = False
    st.session_state.video_analyzing = False
    st.session_state.video_upload_analyzing = False
    st.session_state.video_recording_started = False
    st.session_state.video_stop_clicked = False
    st.session_state.video_upload_processed = False
    st.session_state.uploaded_video_file = None

def audio_analysis(audio_detector):
    """Audio-based depression analysis with manual recording and file upload"""
    st.header("🎵 Audio Analysis")
    
    # Add navigation back to main
    if st.button("← Back to Main", key="audio_back"):
        st.session_state.current_page = "main"
        st.rerun()
    
    if audio_detector is None:
        st.error("❌ Audio analysis not available")
        return
    
    # Language selection
    st.subheader("🌐 Language Selection")
    language_choice = st.radio(
        "Select speech language:",
        ["English", "Indonesian"],
        horizontal=True,
        key="audio_language"
    )
    
    # Set the selected language in the detector
    audio_detector.set_language(language_choice.lower())
    
    # Display analysis options
    st.subheader("📋 Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎙️ Live Recording:**
        - Record your voice in real-time
        - Speak naturally about your feelings
        - Get instant analysis
        """)
        
        st.subheader("Recording Controls")
        handle_audio_recording_states(audio_detector)
    
    with col2:
        st.markdown("""
        **📁 Upload Audio File:**
        - Upload existing MP3 recordings
        - Analyze pre-recorded sessions
        - Same comprehensive analysis
        """)
        
        st.subheader("File Upload")
        handle_audio_upload(audio_detector)
    
    # Display current status
    display_audio_current_status()
    
    # Display recording timer
    if st.session_state.audio_recording and st.session_state.recording_start_time:
        display_audio_recording_timer()
    
    # Display results if available
    if st.session_state.audio_results and not st.session_state.audio_analyzing:
        display_audio_final_results()
    
    # Display usage tips when idle
    if (not st.session_state.audio_recording and 
        not st.session_state.audio_analyzing and 
        not st.session_state.audio_results):
        display_audio_tips()

def handle_audio_recording_states(audio_detector):
    """Handle audio recording state transitions"""
    # Start Recording
    if not st.session_state.audio_recording and not st.session_state.audio_analyzing:
        if st.button("🎙️ Start Recording", type="primary", use_container_width=True, key="start_audio_recording"):
            result = audio_detector.start_recording()
            if "started" in result:
                st.session_state.audio_recording = True
                st.session_state.recording_start_time = time.time()
                st.session_state.audio_recording_started = True
                st.session_state.last_action = "start_audio_recording"
                st.rerun()
            else:
                st.error("❌ Could not start recording")
    
    # Stop Recording
    elif st.session_state.audio_recording and not st.session_state.audio_analyzing:
        if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True, key="stop_audio_recording"):
            st.session_state.audio_recording = False
            st.session_state.audio_analyzing = True
            st.session_state.audio_stop_clicked = True
            st.session_state.last_action = "stop_audio_recording"
            st.rerun()
    
    # Handle analysis after stop
    if st.session_state.audio_stop_clicked and st.session_state.audio_analyzing:
        st.session_state.audio_stop_clicked = False
        
        with st.spinner("🔄 Analyzing audio... This may take a moment."):
            try:
                st.session_state.audio_results = audio_detector.stop_recording()
            except Exception as e:
                st.error(f"❌ Error during audio analysis: {e}")
            finally:
                st.session_state.audio_analyzing = False
                st.rerun()

def handle_audio_upload(audio_detector):
    """Handle audio file upload and analysis - FIXED VERSION"""
    uploaded_audio = st.file_uploader(
        "Choose audio file", 
        type=['mp3', 'wav', 'm4a'],
        help="Upload MP3, WAV, or M4A files for analysis",
        key="audio_uploader"
    )
    
    # Store uploaded file in session state
    if uploaded_audio and uploaded_audio != st.session_state.get('uploaded_audio_file'):
        st.session_state.uploaded_audio_file = uploaded_audio
        st.session_state.audio_upload_processed = False
    
    # Only show analyze button if we have a file that hasn't been processed
    if (st.session_state.uploaded_audio_file and 
        not st.session_state.audio_analyzing and 
        not st.session_state.audio_upload_analyzing and
        not st.session_state.audio_upload_processed):
        
        if st.button("🔍 Analyze Uploaded Audio", type="primary", use_container_width=True, key="analyze_uploaded_audio"):
            st.session_state.audio_upload_analyzing = True
            st.session_state.audio_upload_processed = False
            st.rerun()
    
    # Handle uploaded audio analysis - ONLY when explicitly triggered and not already processed
    if (st.session_state.audio_upload_analyzing and 
        st.session_state.uploaded_audio_file and
        not st.session_state.audio_upload_processed):
        
        # Mark as processing immediately to prevent re-triggering
        st.session_state.audio_upload_processed = True
        
        with st.spinner(f"🔄 Analyzing {st.session_state.uploaded_audio_file.name}..."):
            try:
                st.session_state.audio_results = audio_detector.analyze_uploaded_audio(st.session_state.uploaded_audio_file)
            except Exception as e:
                st.error(f"❌ Error analyzing audio: {e}")
            finally:
                st.session_state.audio_upload_analyzing = False
                st.rerun()

def display_audio_current_status():
    """Display current audio analysis status"""
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.audio_recording:
            st.warning("🔴 **Currently Recording**")
            st.info("Speak naturally about your feelings. Click 'Stop Recording' when finished.")
        elif st.session_state.audio_analyzing or st.session_state.audio_upload_analyzing:
            st.warning("🔄 **Analyzing Audio**")
            st.info("Please wait while we analyze your audio...")
        else:
            st.success("✅ **Ready**")
    
    with col2:
        if st.session_state.uploaded_audio_file:
            st.info(f"📁 **File Ready**: {st.session_state.uploaded_audio_file.name}")

def display_audio_recording_timer():
    """Display audio recording timer"""
    st.markdown("---")
    st.subheader("🎙️ Recording in Progress...")
    
    recording_time = time.time() - st.session_state.recording_start_time
    mins, secs = divmod(int(recording_time), 60)
    st.metric("Recording Time", f"{mins:02d}:{secs:02d}")
    
    # Show recording tips
    st.info("💡 **Tip**: Speak for at least 30 seconds for best results. Describe your feelings naturally.")

def display_audio_final_results():
    """Display final audio analysis results"""
    st.markdown("---")
    display_audio_results(st.session_state.audio_results)
    
    # Add a button to clear results and start over
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Analysis", type="secondary", use_container_width=True, key="new_audio_analysis"):
            reset_audio_state()
            st.rerun()
    with col2:
        if st.button("📊 Show Detailed Analysis", use_container_width=True, key="detailed_audio_analysis"):
            st.info("Detailed analysis features coming soon!")

def reset_audio_state():
    """Reset all audio-related session states"""
    st.session_state.audio_results = None
    st.session_state.audio_recording = False
    st.session_state.audio_analyzing = False
    st.session_state.audio_upload_analyzing = False
    st.session_state.audio_recording_started = False
    st.session_state.audio_stop_clicked = False
    st.session_state.audio_upload_processed = False
    st.session_state.uploaded_audio_file = None

def display_audio_tips():
    """Display audio analysis tips"""
    st.markdown("---")
    st.subheader("💡 Tips for Best Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **For Live Recording:**
        - Speak clearly and naturally
        - Record in a quiet environment
        - Speak for 1-3 minutes
        - Express genuine emotions
        - Use normal speaking volume
        """)
    
    with col2:
        st.markdown("""
        **For File Upload:**
        - MP3 format recommended
        - Clear audio quality
        - Minimum 30 seconds duration
        - Single speaker preferred
        - Minimal background noise
        """)

def complete_session(facial_analyzer, audio_detector):
    """Complete multi-modal analysis session - Both recording and video upload options"""
    st.header("💬 Complete Therapy Session")
    
    # Add navigation back to main
    if st.button("← Back to Main", key="complete_back"):
        st.session_state.current_page = "main"
        st.rerun()
    
    if not facial_analyzer or not audio_detector:
        st.error("❌ Both video and audio analysis required for complete session")
        return
    
    st.info("""
    **Complete Multi-Modal Analysis:**
    - **🎥 Video Analysis**: Facial expression analysis
    - **🎵 Audio Analysis**: Voice and speech content analysis  
    - **🧠 Combined**: Comprehensive emotional assessment
    
    **Choose your method:**
    - **🎥 Record Live**: Record video with audio simultaneously
    - **📁 Upload Video**: Upload existing video file for analysis
    """)
    
    # Language selection for complete session
    st.subheader("🌐 Language Selection")
    language_choice = st.radio(
        "Select speech language:",
        ["English", "Indonesian"],
        horizontal=True,
        key="complete_session_language"
    )
    
    # Set the selected language in the audio detector
    audio_detector.set_language(language_choice.lower())
    
    # Session mode selection
    st.subheader("🎯 Session Mode")
    session_mode = st.radio(
        "Select session mode:",
        ["Record Live Session", "Upload Video File"],
        horizontal=True,
        key="session_mode"
    )
    
    # Store session mode in state
    st.session_state.complete_session_mode = 'record' if session_mode == "Record Live Session" else 'upload'
    
    if st.session_state.complete_session_mode == 'record':
        handle_live_complete_session(facial_analyzer, audio_detector, language_choice)
    else:
        handle_upload_complete_session(facial_analyzer, audio_detector, language_choice)
    
    # Display results if available
    if st.session_state.complete_session_results and not st.session_state.complete_session_analyzing:
        display_complete_session_results()

def handle_live_complete_session(facial_analyzer, audio_detector, language_choice):
    """Handle live recording for complete session"""
    st.subheader("🎥🎙️ Live Recording Session")
    
    st.markdown("""
    **Live Recording Instructions:**
    1. Click **Start Complete Session** to begin recording
    2. A video window will open - speak naturally about your feelings
    3. Both video and audio will be recorded simultaneously
    4. Click **Stop Complete Session** when finished
    5. Get comprehensive analysis from both modalities
    """)
    
    # Complete session controls
    st.subheader("🎯 Session Controls")
    
    if not st.session_state.complete_session_active and not st.session_state.complete_session_analyzing:
        if st.button("🚀 Start Complete Session", type="primary", use_container_width=True):
            # Start both video and audio recording
            try:
                # Start video recording
                video_result = facial_analyzer.start_video_recording()
                # Start audio recording
                audio_result = audio_detector.start_recording()
                
                if (isinstance(video_result, dict) and video_result.get('status') == 'recording_started' and
                    "started" in audio_result):
                    st.session_state.complete_session_active = True
                    st.session_state.recording_start_time = time.time()
                    st.session_state.last_action = "start_complete_session"
                    st.rerun()
                else:
                    st.error("❌ Could not start complete session")
            except Exception as e:
                st.error(f"❌ Error starting complete session: {e}")
    
    elif st.session_state.complete_session_active and not st.session_state.complete_session_analyzing:
        if st.button("⏹️ Stop Complete Session", type="secondary", use_container_width=True):
            st.session_state.complete_session_active = False
            st.session_state.complete_session_analyzing = True
            st.session_state.last_action = "stop_complete_session"
            st.rerun()
    
    # Handle analysis after stopping complete session
    if st.session_state.complete_session_analyzing and st.session_state.complete_session_mode == 'record':
        with st.spinner("🔄 Analyzing complete session data... This may take a moment."):
            try:
                # Stop both recordings and get results
                video_emotions, video_path = facial_analyzer.stop_video_recording()
                audio_results = audio_detector.stop_recording()
                
                # Combine results
                combined_results = {
                    'video_emotions': video_emotions,
                    'audio_results': audio_results,
                    'combined_score': calculate_combined_score(video_emotions, audio_results),
                    'timestamp': datetime.now().isoformat(),
                    'language': language_choice.lower(),
                    'mode': 'live'
                }
                
                st.session_state.complete_session_results = combined_results
                st.session_state.complete_session_analyzing = False
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during complete session analysis: {e}")
                st.session_state.complete_session_analyzing = False
                st.rerun()
    
    # Display current status
    display_complete_session_status()
    
    # Display recording timer if active
    if st.session_state.complete_session_active and st.session_state.recording_start_time:
        display_complete_session_timer()

def handle_upload_complete_session(facial_analyzer, audio_detector, language_choice):
    """Handle video upload for complete session - extracts both video and audio from single file"""
    st.subheader("📁 Upload Video for Complete Analysis")
    
    st.markdown("""
    **Video Upload Instructions:**
    - Upload a video file containing both visual and audio content
    - We'll extract facial expressions from video frames
    - We'll extract and analyze the audio track from the video
    - Get comprehensive analysis from both modalities
    """)
    
    uploaded_video = st.file_uploader(
        "Upload video file for complete analysis",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file with both visual and audio content",
        key="complete_session_video"
    )
    
    # Store uploaded file in session state
    if uploaded_video and uploaded_video != st.session_state.get('complete_session_video_file'):
        st.session_state.complete_session_video_file = uploaded_video
        st.session_state.complete_session_upload_processed = False
    
    # Show analyze button when video is uploaded
    if (st.session_state.complete_session_video_file and 
        not st.session_state.complete_session_analyzing and
        not st.session_state.complete_session_upload_processed):
        
        if st.button("🔍 Run Complete Analysis", type="primary", use_container_width=True):
            st.session_state.complete_session_analyzing = True
            st.session_state.complete_session_upload_processed = False
            st.rerun()
    
    # Handle complete session analysis for upload
    if (st.session_state.complete_session_analyzing and 
        st.session_state.complete_session_mode == 'upload' and
        st.session_state.complete_session_video_file and
        not st.session_state.complete_session_upload_processed):
        
        # Mark as processing immediately to prevent re-triggering
        st.session_state.complete_session_upload_processed = True
        
        with st.spinner("🔄 Running complete multi-modal analysis..."):
            try:
                # Save the uploaded video to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                    temp_video.write(st.session_state.complete_session_video_file.read())
                    video_path = temp_video.name
                
                # Step 1: Analyze video for facial expressions
                st.info("🎥 Analyzing facial expressions...")
                video_emotions = facial_analyzer.process_video_file(video_path)
                
                # Step 2: Extract and analyze audio from the video
                st.info("🎵 Extracting and analyzing audio...")
                audio_results = extract_and_analyze_audio_from_video(audio_detector, video_path, language_choice.lower())
                
                # Step 3: Combine results
                combined_results = {
                    'video_emotions': video_emotions,
                    'audio_results': audio_results,
                    'combined_score': calculate_combined_score(video_emotions, audio_results),
                    'timestamp': datetime.now().isoformat(),
                    'language': language_choice.lower(),
                    'mode': 'video_upload',
                    'filename': st.session_state.complete_session_video_file.name
                }
                
                st.session_state.complete_session_results = combined_results
                
                # Clean up temporary file
                try:
                    os.unlink(video_path)
                except:
                    pass
                
            except Exception as e:
                st.error(f"❌ Error during complete session analysis: {e}")
            finally:
                st.session_state.complete_session_analyzing = False
                st.rerun()
    
    # Display current status for upload mode
    display_complete_session_status()

def extract_and_analyze_audio_from_video(audio_detector, video_path, language):
    """Extract audio from video and analyze it"""
    try:
        # Create temporary audio file
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        
        # Extract audio from video using pydub
        from pydub import AudioSegment
        
        # Load video and extract audio
        video = AudioSegment.from_file(video_path, format="mp4")
        
        # Export as WAV for analysis
        video.export(temp_audio_path, format="wav")
        
        # Analyze the extracted audio
        audio_results = audio_detector.analyze_audio(temp_audio_path)
        
        # Clean up temporary audio file
        try:
            os.unlink(temp_audio_path)
        except:
            pass
        
        return audio_results
        
    except Exception as e:
        st.error(f"❌ Error extracting audio from video: {e}")
        # Return fallback audio results
        return {
            'phq_score': 10,
            'acoustic_score': 10,
            'semantic_score': 10,
            'depression_percentage': 37.0,
            'transcript': 'Audio extraction failed',
            'acoustic_insights': ['Could not analyze audio features'],
            'semantic_insights': ['Could not analyze speech content']
        }

def calculate_combined_score(video_emotions, audio_results):
    """Calculate combined emotional score from video and audio analysis"""
    try:
        # Video component: focus on sadness and negative emotions
        video_score = 0
        if video_emotions:
            sadness = video_emotions.get('sad', 0)
            fear = video_emotions.get('fear', 0)
            angry = video_emotions.get('angry', 0)
            
            # Convert emotion percentages to score (0-27 scale)
            video_score = (sadness * 0.15 + fear * 0.1 + angry * 0.05) * 0.27
        
        # Audio component: use PHQ-like score
        audio_score = audio_results.get('phq_score', 0) if audio_results else 0
        
        # Combine with weights (adjust as needed)
        combined = (video_score * 0.4) + (audio_score * 0.6)
        return min(27, combined)
        
    except Exception as e:
        st.error(f"Error calculating combined score: {e}")
        return 0

def display_complete_session_status():
    """Display current complete session status"""
    st.markdown("---")
    
    if st.session_state.complete_session_analyzing:
        st.warning("🔄 **Analyzing Complete Session**")
        if st.session_state.complete_session_mode == 'record':
            st.info("Processing live recording data...")
        else:
            st.info("Processing uploaded video file...")
    elif st.session_state.complete_session_active:
        st.error("🔴 **Complete Session Active**")
        st.info("Both video and audio recording are in progress. Speak naturally about your feelings.")
    elif st.session_state.complete_session_video_file and not st.session_state.complete_session_results:
        st.success(f"✅ **Ready to Analyze**: {st.session_state.complete_session_video_file.name}")
    elif st.session_state.complete_session_results:
        st.success("✅ **Analysis Complete**")
    else:
        mode = "Live Recording" if st.session_state.complete_session_mode == 'record' else "Video Upload"
        st.info(f"🎯 **Ready for {mode}**")

def display_complete_session_timer():
    """Display complete session recording timer"""
    st.markdown("---")
    st.subheader("🎥🎙️ Complete Session Recording...")
    
    recording_time = time.time() - st.session_state.recording_start_time
    mins, secs = divmod(int(recording_time), 60)
    st.metric("Session Duration", f"{mins:02d}:{secs:02d}")
    
    # Show recording tips
    st.info("""
    💡 **Complete Session Tips:**
    - Speak naturally about your feelings
    - Maintain eye contact with the camera
    - Express emotions genuinely
    - Continue for 1-3 minutes for best results
    - Describe both thoughts and feelings
    """)

def display_complete_session_results():
    """Display complete session analysis results"""
    st.markdown("---")
    st.header("📊 Complete Session Analysis")
    
    results = st.session_state.complete_session_results
    
    # Show mode information
    if results.get('mode') == 'upload':
        st.info(f"📁 **Analyzed File:** {results.get('filename', 'Unknown')}")
    else:
        st.info("🎥 **Live Recording Analysis**")
    
    # Overall score
    combined_score = results.get('combined_score', 0)
    depression_percentage = (combined_score / 27) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Depression Score", f"{combined_score:.1f}/27")
    
    with col2:
        st.metric("Depression Percentage", f"{depression_percentage:.1f}%")
    
    with col3:
        severity = "Unknown"
        if combined_score < 5:
            severity = "Minimal"
        elif combined_score < 10:
            severity = "Mild"
        elif combined_score < 15:
            severity = "Moderate"
        elif combined_score < 20:
            severity = "Moderately Severe"
        else:
            severity = "Severe"
        st.metric("Overall Severity", severity)
    
    # Component analysis
    st.subheader("🔍 Component Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🎥 Video Analysis")
        if results.get('video_emotions'):
            display_video_results(results['video_emotions'], "Facial Expressions")
        else:
            st.warning("No video analysis data available")
    
    with col2:
        st.write("### 🎵 Audio Analysis")
        if results.get('audio_results'):
            audio_results = results['audio_results']
            st.write(f"**PHQ-like Score:** {audio_results.get('phq_score', 0):.1f}/27")
            st.write(f"**Acoustic Score:** {audio_results.get('acoustic_score', 0):.1f}/27")
            st.write(f"**Semantic Score:** {audio_results.get('semantic_score', 0):.1f}/27")
            
            # Show key insights
            st.write("**Voice Insights:**")
            for insight in audio_results.get('acoustic_insights', [])[:3]:
                st.write(f"• {insight}")
            
            st.write("**Content Insights:**")
            for insight in audio_results.get('semantic_insights', [])[:3]:
                st.write(f"• {insight}")
                
            # Show transcript if available
            if audio_results.get('transcript') and audio_results.get('transcript') not in ['Speech recognition unavailable', 'Analysis failed']:
                with st.expander("View Speech Transcript"):
                    st.write(audio_results['transcript'])
        else:
            st.warning("No audio analysis data available")
    
    # Combined insights
    st.subheader("🧠 Combined Emotional Insights")
    
    # Generate insights based on combined data
    insights = generate_combined_insights(results)
    for insight in insights:
        if "warning" in insight.lower() or "severe" in insight.lower():
            st.warning(insight)
        elif "positive" in insight.lower() or "good" in insight.lower():
            st.success(insight)
        else:
            st.info(insight)
    
    # Recommendation
    st.subheader("💡 Recommendations")
    display_recommendations(combined_score, results)
    
    # Reset button
    if st.button("🔄 New Complete Session", type="secondary", use_container_width=True):
        reset_complete_session_state()
        st.rerun()

def generate_combined_insights(results):
    """Generate insights from combined video and audio analysis"""
    insights = []
    
    video_emotions = results.get('video_emotions', {})
    audio_results = results.get('audio_results', {})
    combined_score = results.get('combined_score', 0)
    
    # Video-based insights
    sadness_video = video_emotions.get('sad', 0)
    if sadness_video > 25:
        insights.append("⚠️ **High facial sadness**: Elevated sadness detected in facial expressions")
    elif sadness_video > 15:
        insights.append("ℹ️ **Moderate facial sadness**: Some sadness indicators in expressions")
    
    neutral_video = video_emotions.get('neutral', 0)
    if neutral_video > 70:
        insights.append("😐 **Flat affect**: Predominantly neutral facial expressions detected")
    
    # Audio-based insights
    audio_score = audio_results.get('phq_score', 0)
    if audio_score > 20:
        insights.append("⚠️ **Severe vocal indicators**: Strong depression indicators in speech patterns")
    elif audio_score > 15:
        insights.append("ℹ️ **Moderate vocal indicators**: Some depression indicators in speech")
    
    # Combined insights
    if combined_score > 20:
        insights.append("🚨 **High overall concern**: Multiple indicators suggest significant emotional distress")
    elif combined_score > 15:
        insights.append("⚠️ **Moderate overall concern**: Several indicators of emotional distress present")
    elif combined_score < 5:
        insights.append("✅ **Positive indicators**: Minimal distress signals detected across modalities")
    
    # Consistency check
    if (sadness_video > 20 and audio_score > 15):
        insights.append("🔍 **Consistent indicators**: Both facial and vocal analysis show consistent emotional patterns")
    
    return insights if insights else ["📊 **Baseline analysis**: Standard emotional patterns detected"]

def display_recommendations(combined_score, results):
    """Display recommendations based on complete session analysis"""
    if combined_score >= 20:
        st.error("""
        **🚨 Immediate Recommendation:**
        - Consider consulting with a mental health professional
        - Reach out to support networks
        - Practice self-care and stress management
        - Consider crisis resources if needed
        """)
    elif combined_score >= 15:
        st.warning("""
        **⚠️ Moderate Concern Recommendation:**
        - Monitor emotional patterns regularly
        - Consider speaking with a counselor
        - Practice mindfulness and relaxation techniques
        - Maintain social connections
        """)
    elif combined_score >= 10:
        st.info("""
        **ℹ️ Mild Concern Recommendation:**
        - Continue self-monitoring
        - Practice stress management
        - Maintain healthy routines
        - Consider preventive mental health practices
        """)
    else:
        st.success("""
        **✅ Maintenance Recommendation:**
        - Continue current emotional wellness practices
        - Regular self-check-ins
        - Maintain healthy lifestyle habits
        - Continue social engagement
        """)

def reset_complete_session_state():
    """Reset complete session states"""
    st.session_state.complete_session_active = False
    st.session_state.complete_session_results = None
    st.session_state.complete_session_analyzing = False
    st.session_state.complete_session_upload_processed = False
    st.session_state.complete_session_video_file = None

def display_audio_results(results):
    """Display audio analysis results"""
    st.subheader("🎵 Audio Analysis Results")
    
    if 'error' in results:
        st.error(f"Analysis error: {results['error']}")
        return
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        depression_pct = results.get('depression_percentage', 0)
        st.metric("Depression Percentage", f"{depression_pct:.1f}%")
    
    with col2:
        phq_score = results.get('phq_score', 0)
        st.metric("PHQ-like Score", f"{phq_score:.1f}/27")
    
    with col3:
        severity = "Unknown"
        phq_score = results.get('phq_score', 0)
        if phq_score < 5:
            severity = "Minimal"
        elif phq_score < 10:
            severity = "Mild"
        elif phq_score < 15:
            severity = "Moderate"
        elif phq_score < 20:
            severity = "Moderately Severe"
        else:
            severity = "Severe"
        st.metric("Severity Level", severity)
    
    # Component scores
    st.subheader("Component Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Acoustic Analysis**")
        st.write(f"Score: {results.get('acoustic_score', 0):.1f}/27")
        
        st.write("**Voice Insights:**")
        for insight in results.get('acoustic_insights', []):
            st.write(f"• {insight}")
    
    with col2:
        st.write("**Semantic Analysis**")
        st.write(f"Score: {results.get('semantic_score', 0):.1f}/27")
        
        st.write("**Content Insights:**")
        for insight in results.get('semantic_insights', []):
            st.write(f"• {insight}")
    
    # Transcript
    if results.get('transcript'):
        with st.expander("View Speech Transcript"):
            st.write(results['transcript'])
    
    # Language info
    if results.get('selected_language'):
        st.info(f"🌐 Analysis performed in: {results['selected_language'].title()}")
    
    # Visualization
    st.subheader("Score Distribution")
    scores_data = {
        'Component': ['Acoustic', 'Semantic', 'Combined'],
        'Score': [
            results.get('acoustic_score', 0),
            results.get('semantic_score', 0), 
            results.get('phq_score', 0)
        ]
    }
    df_scores = pd.DataFrame(scores_data)
    
    fig = px.bar(df_scores, x='Component', y='Score', 
                 title="Depression Score Components",
                 color='Component',
                 range_y=[0, 27])
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional info
    if results.get('analysis_timestamp'):
        st.caption(f"Analysis performed at: {results['analysis_timestamp']}")

def display_video_results(emotion_results, question):
    """Display video analysis results"""
    st.subheader("🎭 Facial Expression Analysis")
    
    if not emotion_results:
        st.warning("No emotion data collected")
        return
    
    # Convert to DataFrame for plotting
    emotions_data = []
    for emotion, percentage in emotion_results.items():
        emotions_data.append({
            'Emotion': emotion.capitalize(),
            'Percentage': percentage
        })
    
    df = pd.DataFrame(emotions_data)
    
    # Create bar chart
    fig = px.bar(df, x='Emotion', y='Percentage', 
                 title="Facial Expression Distribution",
                 color='Emotion')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    dominant_emotion = max(emotion_results.items(), key=lambda x: x[1])
    
    with col1:
        st.metric("Dominant Emotion", dominant_emotion[0].capitalize())
    
    with col2:
        st.metric("Dominant Confidence", f"{dominant_emotion[1]:.1f}%")
    
    with col3:
        sadness_pct = emotion_results.get('sad', 0)
        st.metric("Sadness Level", f"{sadness_pct:.1f}%")
        
        # Color code based on sadness level
        if sadness_pct > 25:
            st.error("High sadness level detected")
        elif sadness_pct > 15:
            st.warning("Moderate sadness level detected")
    
    # Emotional insights
    st.subheader("Emotional Insights")
    if emotion_results.get('sad', 0) > 20:
        st.warning("Elevated sadness detected in facial expressions")
    if emotion_results.get('happy', 0) > 30:
        st.success("Positive emotional expressions detected")
    if emotion_results.get('neutral', 0) > 60:
        st.info("Predominantly neutral emotional expressions")

def display_text_results(result):
    """Display text analysis results"""
    st.subheader("📊 Text Analysis Results")
    
    if 'error' in result:
        st.error(f"Analysis error: {result['error']}")
        return
    
    # Main result
    severity = result.get('severity', 'unknown').capitalize()
    confidence = result.get('confidence', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Detected Severity", severity)
    
    with col2:
        st.metric("Confidence", f"{confidence:.2f}")
    
    # Probabilities chart
    if 'probabilities' in result:
        probabilities = result['probabilities']
        
        prob_df = pd.DataFrame({
            'Severity': [s.capitalize() for s in probabilities.keys()],
            'Probability': probabilities.values()
        })
        
        fig = px.bar(prob_df, x='Severity', y='Probability',
                     title="Severity Probability Distribution",
                     color='Severity')
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional info
    if result.get('was_translated'):
        st.info("🌐 Text was translated from Indonesian to English for analysis")
    
    if result.get('detected_language'):
        st.write(f"**Detected Language:** {result['detected_language'].upper()}")

if __name__ == "__main__":
    main()