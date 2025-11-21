# depression_predictor.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os
from googletrans import Translator
import langdetect

class DepressionSeverityPredictor:
    def __init__(self, model_path='depression_model.h5', tokenizer_path='tokenizer.pkl', 
                 label_encoder_path='label_encoder.pkl'):
        """Initialize the predictor with pre-trained model components"""
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.translator = Translator()
        self.max_sequence_length = 200
        
        # Load pre-trained components
        self.load_pretrained_model(model_path, tokenizer_path, label_encoder_path)
    
    def load_pretrained_model(self, model_path, tokenizer_path, label_encoder_path):
        """Load pre-trained model, tokenizer, and label encoder"""
        try:
            # Check if model files exist
            if not all(os.path.exists(path) for path in [model_path, tokenizer_path, label_encoder_path]):
                missing_files = [path for path in [model_path, tokenizer_path, label_encoder_path] if not os.path.exists(path)]
                raise FileNotFoundError(f"Missing model files: {missing_files}")
            
            # Load model
            self.model = load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
            
            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Tokenizer loaded from {tokenizer_path}")
            
            # Load label encoder
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Label encoder loaded from {label_encoder_path}")
            
            print("🎯 Pre-trained model ready for predictions!")
            
        except Exception as e:
            print(f"❌ Error loading pre-trained model: {e}")
            raise e
    
    def detect_language(self, text):
        """Detect the language of the text"""
        try:
            # Simple keyword-based detection for Indonesian
            indonesian_keywords = ['saya', 'aku', 'kamu', 'dia', 'mereka', 'ini', 'itu', 
                                 'dan', 'atau', 'tapi', 'sedang', 'sudah', 'akan',
                                 'depresi', 'cemas', 'stress', 'putus asa', 'sedih',
                                 'hati', 'perasaan', 'pikiran', 'jiwa', 'mental']
            
            text_lower = text.lower()
            indonesian_count = sum(1 for keyword in indonesian_keywords if keyword in text_lower)
            
            if indonesian_count > 2:  # If contains multiple Indonesian keywords
                return 'id'
            else:
                # Use langdetect for other languages
                return langdetect.detect(text)
        except:
            return 'en'  # Default to English if detection fails
    
    def translate_to_english(self, text, source_lang='id'):
        """Translate Indonesian text to English"""
        try:
            if source_lang == 'en':
                return text, 'en'  # No translation needed
            
            translation = self.translator.translate(text, src=source_lang, dest='en')
            return translation.text, source_lang
        except Exception as e:
            print(f"Translation error: {e}")
            return text, 'en'  # Return original text if translation fails
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove user mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            # Remove punctuation and extra spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return ""
    
    def predict_severity(self, text):
        """Predict depression severity for new text with language detection and translation"""
        try:
            # Detect language
            detected_lang = self.detect_language(text)
            
            # Translate if Indonesian
            if detected_lang == 'id':
                translated_text, original_lang = self.translate_to_english(text, 'id')
                original_text = text
            else:
                translated_text = text
                original_lang = 'en'
                original_text = text
            
            # Clean the translated text
            cleaned_text = self.clean_text(translated_text)
            
            # Convert to sequence
            sequence = self.tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)
            
            # Make prediction
            prediction = self.model.predict(padded_sequence, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            # Get the label
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Get all probabilities
            probabilities = {
                label: float(prob) for label, prob in 
                zip(self.label_encoder.classes_, prediction[0])
            }
            
            return {
                'original_text': original_text,
                'translated_text': translated_text,
                'cleaned_text': cleaned_text,
                'detected_language': original_lang,
                'severity': predicted_label,
                'confidence': float(confidence),
                'probabilities': probabilities,
                'was_translated': original_lang == 'id'
            }
        except Exception as e:
            return {
                'original_text': text,
                'severity': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_batch(self, texts):
        """Predict depression severity for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_severity(text)
            results.append(result)
        return results
    
    def predict_dataframe(self, df, text_column='text'):
        """Predict for an entire DataFrame"""
        predictions = []
        for text in df[text_column]:
            prediction = self.predict_single(text)
            predictions.append(prediction)
        
        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        return results_df