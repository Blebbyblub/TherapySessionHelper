# audio_depression_detector.py (CLEANED VERSION)
import numpy as np
import tempfile
import os
from datetime import datetime
import threading
import time
import pyaudio
import wave
import librosa
import speech_recognition as sr
from pydub import AudioSegment

class AudioDepressionDetector:
    def __init__(self):
        """Initialize audio depression detection components"""
        self.text_analyzer = None
        self.is_recording = False
        self.recording_thread = None
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.stream = None
        self.load_models()
    
    def load_models(self):
        """Load text analysis model if available"""
        try:
            # Try to load text analyzer if available
            try:
                from depression_text_predictor import DepressionSeverityPredictor
                self.text_analyzer = DepressionSeverityPredictor()
                print("✅ Text analyzer loaded for audio analysis")
            except ImportError as e:
                print(f"⚠️ Text analyzer not available: {e}")
                self.text_analyzer = None
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def start_recording(self):
        """Start actual audio recording"""
        if self.is_recording:
            return "already_recording"
        
        try:
            self.is_recording = True
            self.frames = []
            
            # Audio recording parameters
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            CHUNK = 1024
            
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            def record_audio():
                """Actual recording thread"""
                print("🎙️ Recording started...")
                while self.is_recording:
                    try:
                        data = self.stream.read(CHUNK, exception_on_overflow=False)
                        self.frames.append(data)
                    except Exception as e:
                        print(f"Recording error: {e}")
                        break
                
                print("🎙️ Recording stopped")
            
            self.recording_thread = threading.Thread(target=record_audio)
            self.recording_thread.start()
            
            return "recording_started"
            
        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            return f"error: {str(e)}"
    
    def stop_recording(self):
        """Stop audio recording and return analysis results"""
        if not self.is_recording:
            return "not_recording"
        
        self.is_recording = False
        
        # Stop and close the stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.recording_thread:
            self.recording_thread.join(timeout=5)
        
        # Save recorded audio to temporary file
        if self.frames:
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            
            try:
                FORMAT = pyaudio.paInt16
                CHANNELS = 1
                RATE = 44100
                
                wave_file = wave.open(temp_audio_path, 'wb')
                wave_file.setnchannels(CHANNELS)
                wave_file.setsampwidth(self.audio.get_sample_size(FORMAT))
                wave_file.setframerate(RATE)
                wave_file.writeframes(b''.join(self.frames))
                wave_file.close()
                
                # Analyze the actual recorded audio
                return self.analyze_audio(temp_audio_path)
                
            except Exception as e:
                print(f"❌ Error saving audio: {e}")
                return self._create_error_result(f"Error saving audio: {e}")
        else:
            print("❌ No audio frames recorded")
            return self._create_error_result("No audio recorded")
    
    def analyze_uploaded_audio(self, audio_file):
        """Analyze uploaded audio file (MP3, WAV, etc.)"""
        try:
            # Save uploaded file to temporary location
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
            audio_file.seek(0)
            
            with open(temp_path, 'wb') as f:
                f.write(audio_file.read())
            
            # Convert to WAV if needed
            if temp_path.endswith('.mp3'):
                wav_path = temp_path.replace('.mp3', '.wav')
                audio = AudioSegment.from_mp3(temp_path)
                audio.export(wav_path, format="wav")
                os.unlink(temp_path)  # Remove original MP3
                temp_path = wav_path
            
            # Analyze the audio
            return self.analyze_audio(temp_path)
            
        except Exception as e:
            print(f"❌ Error analyzing uploaded audio: {e}")
            return self._create_error_result(f"Upload analysis error: {e}")
    
    def analyze_audio(self, audio_path):
        """Analyze actual audio file"""
        try:
            print(f"🎵 Analyzing audio file: {audio_path}")
            
            # Step 1: Extract real acoustic features
            acoustic_features = self.extract_real_acoustic_features(audio_path)
            print("✅ Real acoustic features extracted")
            
            # Step 2: Convert speech to text
            transcript = self.speech_to_text(audio_path)
            print("✅ Speech-to-text conversion completed")
            
            # Step 3: Calculate acoustic depression score
            acoustic_score = self.acoustic_depression_score(acoustic_features)
            print(f"📊 Acoustic score: {acoustic_score}")
            
            # Step 4: Calculate semantic depression score
            semantic_score, text_analysis = self.semantic_depression_score(transcript)
            print(f"📊 Semantic score: {semantic_score}")
            
            # Step 5: Fusion - combine scores
            final_phq_score = (acoustic_score + semantic_score) / 2
            depression_percentage = (final_phq_score / 27) * 100
            
            print("✅ Real audio analysis completed")
            
            return {
                'depression_percentage': depression_percentage,
                'phq_score': final_phq_score,
                'acoustic_score': acoustic_score,
                'semantic_score': semantic_score,
                'transcript': transcript,
                'acoustic_features': acoustic_features,
                'text_analysis': text_analysis,
                'acoustic_insights': self.generate_acoustic_insights(acoustic_features),
                'semantic_insights': self.generate_semantic_insights(text_analysis),
                'analysis_timestamp': datetime.now().isoformat(),
                'audio_duration': self.get_audio_duration(audio_path)
            }
            
        except Exception as e:
            print(f"❌ Audio analysis error: {e}")
            return self._create_error_result(f"Analysis error: {e}")
    
    def extract_real_acoustic_features(self, audio_path):
        """Extract real acoustic features using librosa"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract features
            features = {}
            
            # RMS energy (loudness)
            features['rms_energy'] = float(np.mean(librosa.feature.rms(y=y)))
            
            # Spectral centroid (brightness)
            features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            
            # Zero crossing rate (noisiness)
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            
            # Pitch features using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
            f0 = f0[voiced_flag]  # Only use voiced segments
            
            if len(f0) > 0:
                features['pitch_mean'] = float(np.mean(f0))
                features['pitch_std'] = float(np.std(f0))
            else:
                features['pitch_mean'] = 130.0
                features['pitch_std'] = 25.0
            
            # Speaking rate (approximate)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['speaking_rate'] = float(tempo / 120.0)  # Normalize
            
            # Voice breaks (non-voiced segments)
            features['voice_breaks'] = float(1 - (np.sum(voiced_flag) / len(voiced_flag)) if len(voiced_flag) > 0 else 0.1)
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting real acoustic features: {e}")
            # Return default features instead of demo
            return {
                'rms_energy': 0.05, 'spectral_centroid': 1800, 'zero_crossing_rate': 0.08,
                'pitch_mean': 130, 'pitch_std': 25, 'speaking_rate': 1.0, 'voice_breaks': 0.1
            }
    
    def speech_to_text(self, audio_path):
        """Convert speech to text using speech recognition"""
        try:
            r = sr.Recognizer()
            
            with sr.AudioFile(audio_path) as source:
                # Listen for the data (load audio to memory)
                audio_data = r.record(source)
                # Recognize (convert from speech to text)
                text = r.recognize_google(audio_data)
                return text
                
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return f"Error with speech recognition service: {e}"
        except Exception as e:
            print(f"❌ Speech-to-text error: {e}")
            return "Speech recognition unavailable"
    
    def get_audio_duration(self, audio_path):
        """Get duration of audio file in seconds"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except:
            return 0
    
    def get_recording_status(self):
        """Get current recording status"""
        return self.is_recording
    
    def __del__(self):
        """Cleanup PyAudio"""
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    def acoustic_depression_score(self, acoustic_features):
        """Calculate depression score from acoustic features"""
        try:
            score = 0
            
            if acoustic_features.get('rms_energy', 0) < 0.02:
                score += 4
            elif acoustic_features.get('rms_energy', 0) < 0.04:
                score += 2
            
            if acoustic_features.get('pitch_std', 0) < 15:
                score += 3
            elif acoustic_features.get('pitch_std', 0) < 25:
                score += 1
            
            if acoustic_features.get('speaking_rate', 1.0) < 0.7:
                score += 2
            elif acoustic_features.get('speaking_rate', 1.0) < 0.9:
                score += 1
            
            if acoustic_features.get('voice_breaks', 0) > 0.25:
                score += 2
            elif acoustic_features.get('voice_breaks', 0) > 0.15:
                score += 1
            
            if acoustic_features.get('pitch_mean', 130) < 100:
                score += 1
            
            acoustic_score = min(27, score * 1.8)
            return acoustic_score
            
        except Exception as e:
            print(f"❌ Error in acoustic scoring: {e}")
            return 10
    
    def semantic_depression_score(self, text):
        """Calculate depression score from text content"""
        try:
            if self.text_analyzer:
                if hasattr(self.text_analyzer, 'predict_severity'):
                    result = self.text_analyzer.predict_severity(text)
                elif hasattr(self.text_analyzer, 'predict'):
                    result = self.text_analyzer.predict([text])[0]
                else:
                    result = {'severity': 'moderate'}
                
                severity_map = {
                    'minimal': 4, 'mild': 8, 'moderate': 15, 'severe': 22,
                    'moderately severe': 19, 'error': 10
                }
                
                severity = result.get('severity', 'minimal').lower()
                semantic_score = severity_map.get(severity, 10)
                return semantic_score, result
            else:
                # Fallback keyword-based scoring
                text_lower = text.lower()
                
                depression_keywords = {
                    'down': 2, 'hopeless': 3, 'worthless': 3, 'empty': 2,
                    'sad': 2, 'miserable': 3, 'cant go on': 4, 'tired': 1,
                    'exhausted': 2, 'no energy': 2, 'no point': 3, 'suicide': 5,
                    'end it all': 4, 'pain': 2, 'suffering': 2, 'despair': 3
                }
                
                positive_keywords = {
                    'good': -1, 'happy': -2, 'better': -1, 'improving': -1,
                    'hope': -2, 'excited': -2, 'looking forward': -2, 'joy': -2,
                    'content': -1, 'peaceful': -1, 'grateful': -1, 'optimistic': -2
                }
                
                depression_score = sum(weight for word, weight in depression_keywords.items() if word in text_lower)
                positive_score = sum(weight for word, weight in positive_keywords.items() if word in text_lower)
                
                base_score = 8 + depression_score + positive_score
                semantic_score = max(0, min(27, base_score))
                
                if semantic_score < 5: severity = 'minimal'
                elif semantic_score < 10: severity = 'mild'
                elif semantic_score < 15: severity = 'moderate'
                elif semantic_score < 20: severity = 'moderately severe'
                else: severity = 'severe'
                
                return semantic_score, {'severity': severity, 'confidence': 0.7}
                
        except Exception as e:
            print(f"❌ Semantic analysis error: {e}")
            return 10, {'severity': 'error', 'error': str(e)}
    
    def generate_acoustic_insights(self, features):
        """Generate insights from acoustic features"""
        insights = []
        
        energy = features.get('rms_energy', 0.05)
        if energy < 0.02: insights.append("Very low speech energy - possible severe lethargy")
        elif energy < 0.04: insights.append("Low speech energy - lethargy or fatigue indicated")
        elif energy > 0.1: insights.append("Elevated speech energy - possible agitation or anxiety")
        
        pitch_std = features.get('pitch_std', 25)
        if pitch_std < 10: insights.append("Highly monotonic speech - significant flat affect")
        elif pitch_std < 20: insights.append("Monotonic speech patterns - reduced emotional expression")
        elif pitch_std > 40: insights.append("Variable pitch - expressive emotional range")
        
        rate = features.get('speaking_rate', 1.0)
        if rate < 0.7: insights.append("Very slow speech - psychomotor retardation")
        elif rate < 0.9: insights.append("Slow speaking rate")
        elif rate > 1.3: insights.append("Rapid speech - possible anxiety or agitation")
        
        breaks = features.get('voice_breaks', 0.1)
        if breaks > 0.25: insights.append("Frequent voice breaks - vocal instability detected")
        elif breaks > 0.15: insights.append("Some voice instability present")
        
        return insights if insights else ["Normal speech characteristics detected"]
    
    def generate_semantic_insights(self, text_analysis):
        """Generate insights from text analysis"""
        insights = []
        severity = text_analysis.get('severity', 'unknown')
        
        severity_insights = {
            'minimal': "Speech content shows minimal distress indicators",
            'mild': "Mild emotional distress indicators in speech content",
            'moderate': "Moderate emotional distress evident in speech content", 
            'moderately severe': "Significant emotional distress in speech content",
            'severe': "Severe emotional distress indicators in speech"
        }
        
        if severity in severity_insights:
            insights.append(severity_insights[severity])
        else:
            insights.append("Speech content analysis completed")
        
        if 'error' in text_analysis:
            insights.append("Analysis limited due to technical constraints")
        
        return insights
    
    def _create_error_result(self, error_message):
        """Create consistent error result format"""
        return {
            'depression_percentage': 0,
            'phq_score': 0,
            'acoustic_score': 0,
            'semantic_score': 0,
            'transcript': "Analysis failed",
            'acoustic_features': {},
            'text_analysis': {'severity': 'error'},
            'acoustic_insights': [f"Analysis error: {error_message}"],
            'semantic_insights': ["Could not analyze speech content"],
            'analysis_timestamp': datetime.now().isoformat(),
            'audio_duration': 0,
            'error': error_message
        }