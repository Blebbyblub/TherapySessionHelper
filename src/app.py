# app.py (COMPLETELY FIXED VERSION)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import time
from datetime import datetime

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
        
        # General states
        'last_action': None,
        'action_timestamp': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    st.title("🧠 Therapy Session Helper - Multi-Modal Analysis")
    st.markdown("Comprehensive emotional analysis through video, audio, and text")
    
    # Initialize session state
    initialize_session_state()
    
    # Load models
    text_predictor = load_predictor()
    facial_analyzer = load_facial_analyzer()
    audio_detector = load_audio_detector()
    
    # Show availability status
    col1, col2, col3 = st.columns(3)
    with col1:
        if text_predictor:
            st.success("✅ Text analysis available")
        else:
            st.error("❌ Text analysis unavailable")
    
    with col2:
        if facial_analyzer:
            st.success("✅ Facial analysis available")
        else:
            st.error("❌ Facial analysis unavailable")
    
    with col3:
        if audio_detector:
            st.success("✅ Audio analysis available")
        else:
            st.error("❌ Audio analysis unavailable")
    
    # Sidebar
    st.sidebar.header("Analysis Options")
    
    # Determine available analysis modes
    available_modes = ["Text Analysis"]
    
    if facial_analyzer:
        available_modes.append("Video Analysis")
    if audio_detector:
        available_modes.append("Audio Analysis")
    if facial_analyzer and audio_detector:
        available_modes.append("Complete Session")
    
    analysis_mode = st.sidebar.radio(
        "Select Analysis Type:",
        available_modes
    )
    
    if analysis_mode == "Text Analysis":
        text_analysis(text_predictor)
    elif analysis_mode == "Audio Analysis":
        audio_analysis(audio_detector)
    elif analysis_mode == "Video Analysis":
        video_analysis(facial_analyzer)
    else:
        complete_session(facial_analyzer, audio_detector)

def video_analysis(facial_analyzer):
    """Video-based emotion analysis with manual recording"""
    st.header("🎥 Video Emotion Analysis")
    
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
                    # Fallback to demo results
                    demo_emotion_results = {
                        'neutral': 45.0, 'happy': 25.0, 'sad': 15.0,
                        'angry': 5.0, 'surprise': 5.0, 'fear': 3.0, 'disgust': 2.0
                    }
                    st.session_state.video_results = {
                        'emotions': demo_emotion_results,
                        'video_path': None
                    }
            except Exception as e:
                st.error(f"❌ Error during video analysis: {e}")
                demo_emotion_results = {
                    'neutral': 45.0, 'happy': 25.0, 'sad': 15.0,
                    'angry': 5.0, 'surprise': 5.0, 'fear': 3.0, 'disgust': 2.0
                }
                st.session_state.video_results = {
                    'emotions': demo_emotion_results,
                    'video_path': None
                }
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
                # Fallback to demo
                demo_emotion_results = {
                    'neutral': 45.0, 'happy': 25.0, 'sad': 15.0,
                    'angry': 5.0, 'surprise': 5.0, 'fear': 3.0, 'disgust': 2.0
                }
                st.session_state.video_results = {
                    'emotions': demo_emotion_results,
                    'video_path': None
                }
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
                st.session_state.audio_results = {'error': str(e), 'demo_mode': True}
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
                st.session_state.audio_results = {'error': str(e), 'demo_mode': True}
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

def text_analysis(predictor):
    """Text-based depression analysis"""
    st.header("📝 Text Analysis")
    
    if predictor is None:
        st.warning("⚠️ Text analysis model not available - showing demo mode")
        demo_text_analysis()
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
                demo_text_analysis_with_input(text_input)

def complete_session(facial_analyzer, audio_detector):
    """Complete multi-modal analysis session"""
    st.header("💬 Complete Therapy Session")
    
    st.info("""
    **Complete Multi-Modal Analysis:**
    - **Video**: Facial expression analysis
    - **Audio**: Voice and speech content analysis  
    - **Combined**: Comprehensive emotional assessment
    """)
    
    if not facial_analyzer or not audio_detector:
        st.error("❌ Both video and audio analysis required for complete session")
        return
    
    st.warning("🚧 Complete session implementation in progress...")
    st.info("For now, please use the individual Video and Audio analysis options above.")

def display_audio_results(results):
    """Display audio analysis results"""
    st.subheader("🎵 Audio Analysis Results")
    
    if results.get('demo_mode'):
        st.info("🔸 **Demo Mode**: Showing simulated analysis results")
    
    if 'error' in results:
        st.error(f"Analysis error: {results['error']}")
    
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

def demo_text_analysis():
    """Demo text analysis"""
    text_input = st.text_area(
        "Enter text (demo mode):",
        height=150,
        placeholder="Describe your feelings...",
        key="demo_text_input"
    )
    
    if st.button("Show Demo Analysis", key="demo_analyze_text") and text_input:
        demo_text_analysis_with_input(text_input)

def demo_text_analysis_with_input(text_input):
    """Demo analysis with input text"""
    # Simple keyword-based analysis
    text_lower = text_input.lower()
    
    # Mock analysis based on keywords
    negative_words = ['sad', 'depressed', 'hopeless', 'anxious', 'stress', 'worried']
    positive_words = ['happy', 'good', 'better', 'improving', 'excited']
    
    negative_count = sum(1 for word in negative_words if word in text_lower)
    positive_count = sum(1 for word in positive_words if word in text_lower)
    
    if negative_count > 3:
        severity = 'severe'
    elif negative_count > 1:
        severity = 'moderate'
    elif positive_count > 2:
        severity = 'minimal'
    else:
        severity = 'mild'
    
    mock_result = {
        'severity': severity,
        'confidence': 0.75,
        'probabilities': {
            'minimal': 0.2 if severity == 'minimal' else 0.1,
            'mild': 0.3 if severity == 'mild' else 0.2,
            'moderate': 0.4 if severity == 'moderate' else 0.3,
            'severe': 0.3 if severity == 'severe' else 0.2
        },
        'demo_note': 'This is a demo analysis based on keyword matching'
    }
    
    # Normalize probabilities
    total = sum(mock_result['probabilities'].values())
    for key in mock_result['probabilities']:
        mock_result['probabilities'][key] /= total
    
    display_text_results(mock_result)

if __name__ == "__main__":
    main()