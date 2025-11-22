# audio_depression_detector.py
import numpy as np
import tempfile
import os
from datetime import datetime
import threading
import time

class AudioDepressionDetector:
    def __init__(self):
        """Initialize audio depression detection components"""
        self.text_analyzer = None
        self.is_recording = False
        self.recording_thread = None
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
        """Start audio recording"""
        if self.is_recording:
            return "already_recording"
        
        self.is_recording = True
        self.audio_data = []
        
        def record_audio():
            """Simulated recording thread"""
            start_time = time.time()
            while self.is_recording:
                # Simulate collecting audio data
                time.sleep(0.1)
                # In real implementation, you would capture audio chunks here
            
            recording_duration = time.time() - start_time
            print(f"🎙️ Recording stopped. Duration: {recording_duration:.1f} seconds")
        
        self.recording_thread = threading.Thread(target=record_audio)
        self.recording_thread.start()
        
        return "recording_started"
    
    def stop_recording(self):
        """Stop audio recording and return analysis results"""
        if not self.is_recording:
            return "not_recording"
        
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        # Simulate processing time
        time.sleep(2)
        
        # Return analysis results
        return self.analyze_audio_demo()
    
    def get_recording_status(self):
        """Get current recording status"""
        return self.is_recording
    
    def extract_acoustic_features(self, audio_path=None):
        """Extract acoustic features from audio - demo version"""
        try:
            # For demo purposes, return simulated acoustic features
            # In production, you would use librosa to analyze real audio
            
            # Simulate different emotional states
            emotional_states = [
                # Depressed state features
                {
                    'rms_energy': 0.008,
                    'spectral_centroid': 1200,
                    'zero_crossing_rate': 0.06,
                    'pitch_mean': 110,
                    'pitch_std': 8,
                    'speaking_rate': 0.8,
                    'voice_breaks': 0.3
                },
                # Neutral state features  
                {
                    'rms_energy': 0.05,
                    'spectral_centroid': 1800,
                    'zero_crossing_rate': 0.08,
                    'pitch_mean': 130,
                    'pitch_std': 25,
                    'speaking_rate': 1.0,
                    'voice_breaks': 0.1
                },
                # Anxious state features
                {
                    'rms_energy': 0.12,
                    'spectral_centroid': 2200,
                    'zero_crossing_rate': 0.15,
                    'pitch_mean': 150,
                    'pitch_std': 40,
                    'speaking_rate': 1.4,
                    'voice_breaks': 0.2
                }
            ]
            
            # Return a random emotional state for demo
            import random
            features = random.choice(emotional_states)
            
            # Add some random variation
            for key in features:
                if isinstance(features[key], float):
                    features[key] *= random.uniform(0.8, 1.2)
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting acoustic features: {e}")
            # Return default neutral features
            return {
                'rms_energy': 0.05,
                'spectral_centroid': 1800,
                'zero_crossing_rate': 0.08,
                'pitch_mean': 130,
                'pitch_std': 25,
                'speaking_rate': 1.0,
                'voice_breaks': 0.1
            }
    
    def acoustic_depression_score(self, acoustic_features):
        """Calculate depression score from acoustic features"""
        try:
            score = 0
            
            # Depression indicators (based on research)
            # Low energy - lethargy
            if acoustic_features.get('rms_energy', 0) < 0.02:
                score += 4
            elif acoustic_features.get('rms_energy', 0) < 0.04:
                score += 2
            
            # Monotonic speech - flat affect
            if acoustic_features.get('pitch_std', 0) < 15:
                score += 3
            elif acoustic_features.get('pitch_std', 0) < 25:
                score += 1
            
            # Slow speaking rate
            if acoustic_features.get('speaking_rate', 1.0) < 0.7:
                score += 2
            elif acoustic_features.get('speaking_rate', 1.0) < 0.9:
                score += 1
            
            # Voice breaks and instability
            if acoustic_features.get('voice_breaks', 0) > 0.25:
                score += 2
            elif acoustic_features.get('voice_breaks', 0) > 0.15:
                score += 1
            
            # Low pitch
            if acoustic_features.get('pitch_mean', 130) < 100:
                score += 1
            
            # Normalize to 0-27 scale (PHQ-9 like)
            acoustic_score = min(27, score * 1.8)
            
            return acoustic_score
            
        except Exception as e:
            print(f"❌ Error in acoustic scoring: {e}")
            return 10  # Default middle score
    
    def speech_to_text_demo(self):
        """Generate demo speech transcripts"""
        demo_transcripts = [
            # Depressed speech patterns
            "I've been feeling really down lately. Nothing seems to bring me joy anymore and I'm just going through the motions. It's hard to find the energy to do anything.",
            
            # Anxious speech patterns  
            "I've been so worried about everything lately. My mind is constantly racing and I can't seem to relax. I feel like something bad is going to happen.",
            
            # Neutral/mild distress
            "I've been a bit stressed with work and everything, but I'm managing. Some days are better than others, you know?",
            
            # Positive/healthy
            "I've been feeling pretty good recently. I've been exercising and spending time with friends, which helps my mood a lot.",
            
            # Mixed emotions
            "It's been ups and downs. Some days I feel okay, other days it's a struggle. I'm trying to take things one day at a time."
        ]
        
        import random
        return random.choice(demo_transcripts)
    
    def semantic_depression_score(self, text):
        """Calculate depression score from text content"""
        try:
            if self.text_analyzer:
                # Use the text analyzer's predict method
                if hasattr(self.text_analyzer, 'predict_severity'):
                    result = self.text_analyzer.predict_severity(text)
                elif hasattr(self.text_analyzer, 'predict'):
                    result = self.text_analyzer.predict([text])[0]
                else:
                    result = {'severity': 'moderate'}
                
                # Map severity to PHQ-like score (0-27)
                severity_map = {
                    'minimal': 4,
                    'mild': 8,
                    'moderate': 15,
                    'severe': 22,
                    'moderately severe': 19,
                    'error': 10
                }
                
                severity = result.get('severity', 'minimal').lower()
                semantic_score = severity_map.get(severity, 10)
                
                return semantic_score, result
            else:
                # Fallback: keyword-based scoring
                text_lower = text.lower()
                
                # Depression indicators
                depression_keywords = {
                    'down': 2, 'hopeless': 3, 'worthless': 3, 'empty': 2,
                    'sad': 2, 'miserable': 3, 'cant go on': 4, 'tired': 1,
                    'exhausted': 2, 'no energy': 2, 'no point': 3, 'suicide': 5,
                    'end it all': 4, 'pain': 2, 'suffering': 2, 'despair': 3
                }
                
                # Positive indicators (reduce score)
                positive_keywords = {
                    'good': -1, 'happy': -2, 'better': -1, 'improving': -1,
                    'hope': -2, 'excited': -2, 'looking forward': -2, 'joy': -2,
                    'content': -1, 'peaceful': -1, 'grateful': -1, 'optimistic': -2
                }
                
                depression_score = 0
                for word, weight in depression_keywords.items():
                    if word in text_lower:
                        depression_score += weight
                
                positive_score = 0
                for word, weight in positive_keywords.items():
                    if word in text_lower:
                        positive_score += weight
                
                base_score = 8 + depression_score + positive_score
                semantic_score = max(0, min(27, base_score))
                
                # Determine severity based on score
                if semantic_score < 5:
                    severity = 'minimal'
                elif semantic_score < 10:
                    severity = 'mild'
                elif semantic_score < 15:
                    severity = 'moderate'
                elif semantic_score < 20:
                    severity = 'moderately severe'
                else:
                    severity = 'severe'
                
                return semantic_score, {'severity': severity, 'confidence': 0.7}
                
        except Exception as e:
            print(f"❌ Semantic analysis error: {e}")
            return 10, {'severity': 'error', 'error': str(e)}
    
    def analyze_audio_demo(self):
        """Provide complete demo audio analysis"""
        try:
            print("🎵 Starting demo audio analysis...")
            
            # Step 1: Extract acoustic features (demo)
            acoustic_features = self.extract_acoustic_features()
            print("✅ Acoustic features extracted")
            
            # Step 2: Calculate acoustic depression score
            acoustic_score = self.acoustic_depression_score(acoustic_features)
            print(f"📊 Acoustic score: {acoustic_score}")
            
            # Step 3: Generate demo transcript
            transcript = self.speech_to_text_demo()
            print("✅ Transcript generated")
            
            # Step 4: Calculate semantic depression score
            semantic_score, text_analysis = self.semantic_depression_score(transcript)
            print(f"📊 Semantic score: {semantic_score}")
            
            # Step 5: Fusion - combine scores
            final_phq_score = (acoustic_score + semantic_score) / 2
            depression_percentage = (final_phq_score / 27) * 100
            
            print("✅ Audio analysis completed")
            
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
                'demo_mode': True,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Audio analysis error: {e}")
            # Return safe demo data
            return {
                'depression_percentage': 35.2,
                'phq_score': 9.5,
                'acoustic_score': 8,
                'semantic_score': 11,
                'transcript': "I've been feeling a bit stressed lately but I'm managing okay.",
                'acoustic_features': {},
                'text_analysis': {'severity': 'mild'},
                'acoustic_insights': ["Demo analysis - normal speech patterns"],
                'semantic_insights': ["Demo analysis - mild distress indicators"],
                'demo_mode': True,
                'error': str(e)
            }
    
    def generate_acoustic_insights(self, features):
        """Generate insights from acoustic features"""
        insights = []
        
        # Energy analysis
        energy = features.get('rms_energy', 0.05)
        if energy < 0.02:
            insights.append("Very low speech energy - possible severe lethargy")
        elif energy < 0.04:
            insights.append("Low speech energy - lethargy or fatigue indicated")
        elif energy > 0.1:
            insights.append("Elevated speech energy - possible agitation or anxiety")
        
        # Pitch analysis
        pitch_std = features.get('pitch_std', 25)
        if pitch_std < 10:
            insights.append("Highly monotonic speech - significant flat affect")
        elif pitch_std < 20:
            insights.append("Monotonic speech patterns - reduced emotional expression")
        elif pitch_std > 40:
            insights.append("Variable pitch - expressive emotional range")
        
        # Speaking rate
        rate = features.get('speaking_rate', 1.0)
        if rate < 0.7:
            insights.append("Very slow speech - psychomotor retardation")
        elif rate < 0.9:
            insights.append("Slow speaking rate")
        elif rate > 1.3:
            insights.append("Rapid speech - possible anxiety or agitation")
        
        # Voice quality
        breaks = features.get('voice_breaks', 0.1)
        if breaks > 0.25:
            insights.append("Frequent voice breaks - vocal instability detected")
        elif breaks > 0.15:
            insights.append("Some voice instability present")
        
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
        
        # Add additional insights based on common themes
        if 'error' in text_analysis:
            insights.append("Analysis limited due to technical constraints")
        
        return insights