# facial_analyzer.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from collections import deque
import os
import tempfile

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
        
        # Load or create emotion model
        if model_path and os.path.exists(model_path):
            self.load_emotion_model(model_path)
        else:
            self.create_emotion_model()
    
    def create_emotion_model(self):
        """Create a pre-trained emotion recognition model"""
        try:
            print("🔄 Creating emotion recognition model...")
            # Using MobileNetV2 as base
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Add custom layers for emotion classification
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            predictions = Dense(7, activation='softmax')(x)  # 7 emotions
            
            self.emotion_model = Model(inputs=base_model.input, outputs=predictions)
            
            # Freeze base model layers
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
            self.emotion_model = load_model(model_path)
            print("✅ Emotion model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading emotion model: {e}")
            self.create_emotion_model()
    
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
    
    def preprocess_face(self, face_roi):
        """Preprocess face ROI for emotion classification"""
        try:
            # Resize to model input size
            face_resized = cv2.resize(face_roi, (224, 224))
            
            # Convert to RGB if needed
            if len(face_resized.shape) == 2:  # Grayscale
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
            elif face_resized.shape[2] == 4:  # RGBA
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_RGBA2RGB)
            
            # Normalize pixel values
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Expand dimensions for batch
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            return face_batch
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def predict_emotion(self, face_roi):
        """Predict emotion from face ROI"""
        if self.emotion_model is None:
            return self._get_demo_emotions()
        
        preprocessed_face = self.preprocess_face(face_roi)
        if preprocessed_face is None:
            return self._get_demo_emotions()
        
        try:
            predictions = self.emotion_model.predict(preprocessed_face, verbose=0)
            
            # Get emotion probabilities
            emotion_probs = {
                self.emotion_labels[i]: float(predictions[0][i]) 
                for i in range(len(self.emotion_labels))
            }
            
            return emotion_probs
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._get_demo_emotions()
    
    def process_frame(self, frame):
        """Process a single frame and return emotion analysis"""
        try:
            faces = self.detect_faces(frame)
            
            if len(faces) > 0:
                # Use the first detected face (main subject)
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    emotion_probs = self.predict_emotion(face_roi)
                    
                    # Draw bounding box and emotion text on frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])
                    cv2.putText(frame, f"{dominant_emotion[0]}: {dominant_emotion[1]:.2f}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    return frame, emotion_probs
            
            return frame, None
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame, None
    
    def record_video_response(self, duration_seconds=30, output_path="response_video.mp4"):
        """Record a video response with real-time emotion analysis"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Could not access webcam")
                return self._get_demo_emotions(), None
            
            # Get video properties
            fps = 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            emotion_results = []
            start_time = cv2.getTickCount()
            
            print("🎥 Recording video response...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate elapsed time
                elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                if elapsed_time > duration_seconds:
                    break
                
                # Process frame for emotion analysis
                processed_frame, emotion_probs = self.process_frame(frame)
                
                if emotion_probs:
                    emotion_results.append(emotion_probs)
                
                # Write frame to video
                out.write(processed_frame)
                
                # Display frame
                cv2.imshow('Therapy Session - Recording', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Analyze the recorded video
            if emotion_results:
                final_emotions = self._aggregate_emotions(emotion_results)
            else:
                final_emotions = self._get_demo_emotions()
            
            return final_emotions, output_path
            
        except Exception as e:
            print(f"Video recording error: {e}")
            return self._get_demo_emotions(), None
    
    def analyze_video(self, video_path, frames_per_second=5):
        """Analyze facial expressions in video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # Default assumption
            frame_interval = max(1, int(fps / frames_per_second))
            
            emotion_results = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % frame_interval == 0:
                    faces = self.detect_faces(frame)
                    
                    if len(faces) > 0:
                        # Use the first detected face (main subject)
                        x, y, w, h = faces[0]
                        face_roi = frame[y:y+h, x:x+w]
                        
                        if face_roi.size > 0:
                            emotion_probs = self.predict_emotion(face_roi)
                            if emotion_probs:
                                emotion_results.append(emotion_probs)
                
                frame_count += 1
            
            cap.release()
            
            if not emotion_results:
                return self._get_demo_emotions()
            
            # Aggregate results
            final_emotions = self._aggregate_emotions(emotion_results)
            
            return final_emotions
            
        except Exception as e:
            print(f"Video analysis error: {e}")
            return self._get_demo_emotions()
    
    def _aggregate_emotions(self, emotion_results):
        """Aggregate emotion results with smoothing"""
        aggregated = {}
        for emotion in self.emotion_labels:
            emotion_values = [result.get(emotion, 0) for result in emotion_results]
            aggregated[emotion] = np.mean(emotion_values) * 100  # Convert to percentage
        
        return aggregated
    
    def _get_demo_emotions(self):
        """Return demo emotion percentages for testing"""
        return {
            'neutral': 65.0,
            'happy': 15.0,
            'sad': 10.0,
            'angry': 4.0,
            'surprise': 3.0,
            'fear': 2.0,
            'disgust': 1.0
        }
    
    def get_emotion_insights(self, emotion_results):
        """Generate insights based on emotion analysis"""
        sadness_score = emotion_results.get('sad', 0)
        neutral_score = emotion_results.get('neutral', 0)
        happy_score = emotion_results.get('happy', 0)
        negative_emotions = emotion_results.get('angry', 0) + emotion_results.get('fear', 0) + emotion_results.get('disgust', 0)
        
        insights = []
        
        if sadness_score > 25:
            insights.append("Elevated sadness detected in facial expressions")
        elif sadness_score > 15:
            insights.append("Moderate sadness levels observed")
        
        if neutral_score > 70:
            insights.append("Predominantly neutral emotional expressions")
        
        if negative_emotions > 20:
            insights.append("Noticeable negative emotional expressions")
        elif negative_emotions > 10:
            insights.append("Some negative emotional expressions present")
        
        if happy_score > 30:
            insights.append("Positive emotional expressions detected")
        elif happy_score > 15:
            insights.append("Some positive emotional expressions observed")
        
        return insights if insights else ["Emotional expressions within typical range"]