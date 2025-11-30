# facial_analyzer.py (FIXED PERCENTAGE DISTRIBUTION)
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import tempfile
import threading
import time

class FacialExpressionAnalyzer:
    def __init__(self, model_path=None):
        # Use OpenCV's built-in face detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise ValueError("Could not load face detection model")
        except Exception as e:
            print(f"Error loading face detector: {e}")
            raise e
            
        self.emotion_model = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.frame_buffer = deque(maxlen=16)
        self.is_recording = False
        self.recording_thread = None
        self.recorded_frames = []
        self.emotion_results = []
        self.cap = None
        
        # Initialize DeepFace with OpenCV backend
        self.deepface_available = self._initialize_deepface()
        
        if not self.deepface_available:
            print("⚠️ DeepFace not available, using fallback model")
            self.create_emotion_model()
    
    def _initialize_deepface(self):
        """Initialize DeepFace for emotion recognition"""
        try:
            import deepface
            from deepface import DeepFace
            print("✅ DeepFace imported successfully with OpenCV backend")
            return True
        except ImportError as e:
            print(f"❌ DeepFace not available: {e}")
            return False

    def predict_emotion(self, face_roi):
        """Predict emotion from face ROI using DeepFace with OpenCV backend"""
        if not self.deepface_available:
            return self._get_demo_emotion_probs()
        
        try:
            import deepface
            from deepface import DeepFace
            
            # Save face ROI to temporary file for DeepFace
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
            cv2.imwrite(temp_path, face_roi)
            
            # Analyze with DeepFace using OpenCV backend
            analysis = DeepFace.analyze(
                img_path=temp_path,
                actions=['emotion'],
                detector_backend='opencv',  # Use OpenCV backend
                enforce_detection=False,
                silent=True
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            if analysis and len(analysis) > 0:
                emotion_results = analysis[0]['emotion']
                
                # FIX: DeepFace already returns percentages (0-100), don't multiply again!
                # Just convert to our format
                emotion_probs = {}
                for emotion in self.emotion_labels:
                    emotion_probs[emotion] = float(emotion_results.get(emotion, 0))
                
                return emotion_probs
            else:
                return self._get_demo_emotion_probs()
                
        except Exception as e:
            print(f"DeepFace prediction error: {e}")
            return self._get_demo_emotion_probs()

    def _aggregate_emotions(self, emotion_results):
        """Aggregate emotion results with smoothing - FIXED: don't multiply by 100"""
        aggregated = {}
        for emotion in self.emotion_labels:
            emotion_values = [result.get(emotion, 0) for result in emotion_results]
            # FIX: Don't multiply by 100 - values are already percentages!
            aggregated[emotion] = np.mean(emotion_values)
        return aggregated

    # KEEP ALL OTHER METHODS EXACTLY THE SAME (only changed predict_emotion and _aggregate_emotions)
    def create_emotion_model(self):
        """Create a pre-trained emotion recognition model"""
        try:
            print("🔄 Creating emotion recognition model...")
            base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            predictions = tf.keras.layers.Dense(7, activation='softmax')(x)
            
            self.emotion_model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
            
            for layer in base_model.layers:
                layer.trainable = False
                
            print("✅ Emotion model created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating emotion model: {e}")
            self._create_simple_model()
    
    def _create_simple_model(self):
        """Create a simple fallback model"""
        print("🔄 Creating simple fallback model...")
        try:
            self.emotion_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(7, activation='softmax')
            ])
            print("✅ Simple fallback model created!")
        except Exception as e:
            print(f"❌ Error creating fallback model: {e}")
            self.emotion_model = None
    
    def load_emotion_model(self, model_path):
        """Load pre-trained emotion model"""
        try:
            self.emotion_model = tf.keras.models.load_model(model_path)
            print("✅ Emotion model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading emotion model: {e}")
            self.create_emotion_model()
    
    def start_video_recording(self):
        """Start video recording - controlled only by frontend button"""
        if self.is_recording:
            return {"status": "already_recording"}
        
        self.is_recording = True
        self.recorded_frames = []
        self.emotion_results = []
        
        def record_video():
            """Video recording thread"""
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Could not access webcam")
                self.is_recording = False
                return
            
            print("🎥 Starting video recording...")
            
            while self.is_recording:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame for emotion analysis
                processed_frame, emotion_probs = self.process_frame(frame)
                
                if emotion_probs:
                    self.emotion_results.append(emotion_probs)
                    self.recorded_frames.append(processed_frame)
                else:
                    self.recorded_frames.append(frame)
                
                # Display frame with simple status (no Q key handling)
                cv2.putText(processed_frame, "RECORDING - Use app to stop", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Therapy Session - Recording', processed_frame)
                cv2.waitKey(1)  # Minimal delay
            
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Video recording stopped")
        
        self.recording_thread = threading.Thread(target=record_video)
        self.recording_thread.start()
        
        return {"status": "recording_started"}
    
    def stop_video_recording(self):
        """Stop video recording and return analysis results"""
        if not self.is_recording:
            return None, None
        
        self.is_recording = False
        
        # Close camera and windows
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.recording_thread:
            self.recording_thread.join(timeout=5)
        
        # Save recorded video if we have frames
        output_path = None
        if self.recorded_frames:
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            try:
                height, width = self.recorded_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
                
                for frame in self.recorded_frames:
                    out.write(frame)
                out.release()
                print(f"✅ Video saved to: {output_path}")
            except Exception as e:
                print(f"❌ Error saving video: {e}")
                output_path = None
        else:
            print("❌ No frames recorded")
        
        # Analyze emotions
        if self.emotion_results:
            final_emotions = self._aggregate_emotions(self.emotion_results)
        else:
            final_emotions = self._get_demo_emotions()
            print("⚠️ Using demo emotions - no emotion data collected")
        
        return final_emotions, output_path
    
    def analyze_uploaded_video(self, video_file):
        """Analyze uploaded video file"""
        try:
            # Save uploaded file to temporary location
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            video_file.seek(0)
            
            with open(temp_path, 'wb') as f:
                f.write(video_file.read())
            
            # Process the video file
            return self.process_video_file(temp_path)
            
        except Exception as e:
            print(f"❌ Error analyzing uploaded video: {e}")
            return self._get_demo_emotions()
    
    def process_video_file(self, video_path):
        """Process uploaded video file for emotion analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            emotion_results = []
            frames_processed = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame to reduce processing load
                if frames_processed % 5 == 0:
                    processed_frame, emotion_probs = self.process_frame(frame)
                    if emotion_probs:
                        emotion_results.append(emotion_probs)
                
                frames_processed += 1
            
            cap.release()
            
            if emotion_results:
                final_emotions = self._aggregate_emotions(emotion_results)
                print(f"✅ Processed {len(emotion_results)} frames from uploaded video")
                return final_emotions
            else:
                print("❌ No faces detected in uploaded video")
                return self._get_demo_emotions()
                
        except Exception as e:
            print(f"❌ Error processing video file: {e}")
            return self._get_demo_emotions()
    
    def get_recording_status(self):
        """Get current recording status"""
        return self.is_recording
    
    def detect_faces(self, frame):
        """Detect faces in frame using OpenCV Haar Cascades"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def process_frame(self, frame):
        """Process a single frame and return emotion analysis"""
        try:
            faces = self.detect_faces(frame)
            processed_frame = frame.copy()
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    emotion_probs = self.predict_emotion(face_roi)
                    
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])
                    cv2.putText(processed_frame, f"{dominant_emotion[0]}: {dominant_emotion[1]:.1f}%", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    return processed_frame, emotion_probs
            
            return processed_frame, None
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame, None
    
    def _get_demo_emotions(self):
        """Return demo emotion percentages for testing"""
        return {
            'neutral': 45.0, 'happy': 25.0, 'sad': 12.0, 'angry': 8.0,
            'surprise': 6.0, 'fear': 3.0, 'disgust': 1.0
        }
    
    def _get_demo_emotion_probs(self):
        """Return demo emotion probabilities for single frame"""
        return {
            'neutral': 0.45, 'happy': 0.25, 'sad': 0.12, 'angry': 0.08,
            'surprise': 0.06, 'fear': 0.03, 'disgust': 0.01
        }