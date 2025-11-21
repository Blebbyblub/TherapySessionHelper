# app.py
import streamlit as st

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Therapy Session Helper",
    page_icon="🧠",
    layout="wide"
)

# Now import other modules
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import time

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

# Import handlers with better error handling
FacialExpressionAnalyzer, FACE_ANALYSIS_AVAILABLE = safe_import('facial_analyzer', 'FacialExpressionAnalyzer')
DepressionSeverityPredictor, TEXT_ANALYSIS_AVAILABLE = safe_import('depression_text_predictor', 'DepressionSeverityPredictor')

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

def main():
    st.title("🧠 Therapy Session Helper")
    st.markdown("Record video responses to therapy questions and get comprehensive emotional analysis")
    
    # Load models
    text_predictor = load_predictor()
    facial_analyzer = load_facial_analyzer()
    
    # Show availability status
    col1, col2 = st.columns(2)
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
    
    # Sidebar
    st.sidebar.header("Session Options")
    
    # Determine available analysis modes
    available_modes = ["Text Analysis"]  # Always available for demo
    
    if facial_analyzer:
        available_modes.append("Video Response Session")
    if text_predictor and facial_analyzer:
        available_modes.append("Complete Therapy Session")
    
    analysis_mode = st.sidebar.radio(
        "Select Session Type:",
        available_modes
    )
    
    if analysis_mode == "Text Analysis":
        text_analysis(text_predictor)
    elif analysis_mode == "Video Response Session":
        video_response_session(facial_analyzer)
    else:
        complete_therapy_session(text_predictor, facial_analyzer)

def text_analysis(predictor):
    """Text-based depression analysis"""
    st.header("Text Analysis")
    
    if predictor is None:
        st.warning("⚠️ Text analysis model not available - showing demo mode")
        demo_text_analysis()
        return
    
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Describe how you're feeling or paste text to analyze..."
    )
    
    if st.button("Analyze Text", type="primary") and text_input:
        with st.spinner("Analyzing text..."):
            result = predictor.predict_single(text_input)
            
            if 'error' in result:
                st.error(f"❌ Analysis error: {result['error']}")
                return
            
            display_text_results(result)

def demo_text_analysis():
    """Demo text analysis when model is not available"""
    text_input = st.text_area(
        "Enter text (demo mode):",
        height=150,
        placeholder="This is a demo. Real analysis requires the depression model."
    )
    
    if st.button("Show Demo Analysis") and text_input:
        # Mock analysis for demo
        mock_result = {
            'severity': 'mild',
            'confidence': 0.75,
            'probabilities': {
                'minimum': 0.1,
                'mild': 0.4,
                'moderate': 0.3,
                'severe': 0.2
            }
        }
        display_text_results(mock_result)

def video_response_session(analyzer):
    """Video response recording and analysis"""
    if analyzer is None:
        st.error("Facial analysis model not available")
        return
    
    st.header("🎥 Video Response Session")
    
    # Therapy questions database
    therapy_questions = [
        "How have you been feeling over the past week?",
        "What's been on your mind lately that's been causing stress or worry?",
        "Can you describe a recent situation that made you feel particularly happy or sad?",
        "How have your sleep patterns and appetite been recently?",
        "What activities or relationships bring you the most joy right now?",
        "Is there anything you've been avoiding or putting off?",
        "How would you describe your energy levels and motivation recently?",
        "What are you most looking forward to in the coming weeks?"
    ]
    
    # Session setup
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Session Configuration")
        question_index = st.selectbox(
            "Select therapy question:",
            range(len(therapy_questions)),
            format_func=lambda x: therapy_questions[x]
        )
        
        recording_duration = st.slider(
            "Recording duration (seconds):",
            min_value=15,
            max_value=120,
            value=45,
            help="How long to record your response"
        )
    
    with col2:
        st.subheader("Instructions")
        st.info("""
        🎯 **How to use:**
        1. Read the selected question
        2. Click 'Start Recording'
        3. Answer naturally while looking at the camera
        4. Speak clearly and express your feelings
        5. Recording will stop automatically
        """)
    
    # Selected question display
    st.subheader("Your Question:")
    st.markdown(f"### ❝{therapy_questions[question_index]}❞")
    
    # Recording section
    if st.button("🎬 Start Recording Response", type="primary", use_container_width=True):
        with st.spinner(f"Preparing to record for {recording_duration} seconds..."):
            time.sleep(2)  # Give user time to prepare
            
            # Create temporary file for recording
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                video_path = tmp_file.name
            
            try:
                # Record video response
                st.info("🔴 Recording started... Speak your response naturally")
                st.warning("⚠️ A new window will open for recording. Please allow camera access.")
                
                emotion_results, recorded_path = analyzer.record_video_response(
                    duration_seconds=recording_duration,
                    output_path=video_path
                )
                
                st.success("✅ Recording completed!")
                
                # Display analysis results
                display_video_response_results(emotion_results, recorded_path, therapy_questions[question_index])
                
            except Exception as e:
                st.error(f"❌ Recording error: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(video_path):
                    os.unlink(video_path)

def complete_therapy_session(text_predictor, facial_analyzer):
    """Complete therapy session with both text and video analysis"""
    st.header("💬 Complete Therapy Session")
    
    st.info("""
    This session combines video responses with text analysis for comprehensive emotional assessment.
    You'll record video responses to therapy questions and also provide written reflections.
    """)
    
    if text_predictor is None or facial_analyzer is None:
        st.error("❌ Both text and facial analysis models are required for complete session")
        return
    
    # Rest of the complete_therapy_session function remains the same...
    # [Previous implementation continues...]

# [Rest of the display functions remain the same...]

if __name__ == "__main__":
    main()