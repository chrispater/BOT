"""
Machine Learning Predictor for Crypto Breakouts and Breakdowns
Learns from historical patterns and technical indicators to predict price movements
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
import pickle


class MLPredictor:
    """
    ML-based predictor that learns from historical technical patterns
    to predict breakouts and breakdowns before they happen
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = "ml_model.pkl"
        self.scaler_path = "ml_scaler.pkl"
        self.training_data_path = "ml_training_data.json"
        self.min_training_samples = 20  # Minimum samples needed to train
        
        # Load existing model if available
        self._load_model()
    
    def predict_breakout_probability(self, features: Dict) -> Dict:
        """
        Predict probability of breakout, breakdown, or neutral movement.
        
        Args:
            features: Dict with technical indicators and patterns
            
        Returns:
            Dict with predictions and probabilities
        """
        if not self.is_trained:
            # Not enough data yet, return neutral prediction
            return {
                'prediction': 'NEUTRAL',
                'breakout_probability': 0.33,
                'breakdown_probability': 0.33,
                'neutral_probability': 0.34,
                'confidence': 0.0,
                'ml_enabled': False,
                'reason': 'ML model not trained yet - needs more historical data'
            }
        
        try:
            # Extract and scale features
            feature_vector = self._extract_feature_vector(features)
            scaled_features = self.scaler.transform([feature_vector])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(scaled_features)[0]
            prediction = self.model.predict(scaled_features)[0]
            
            # Map to readable labels
            class_names = ['BREAKDOWN', 'NEUTRAL', 'BREAKOUT']
            breakdown_prob, neutral_prob, breakout_prob = probabilities
            
            # Determine confidence (how certain the model is)
            max_prob = max(probabilities)
            confidence = (max_prob - 0.33) / 0.67 * 100  # Scale to 0-100
            
            return {
                'prediction': class_names[prediction],
                'breakout_probability': float(breakout_prob),
                'breakdown_probability': float(breakdown_prob),
                'neutral_probability': float(neutral_prob),
                'confidence': float(confidence),
                'ml_enabled': True,
                'reason': f'ML model predicts {class_names[prediction]} with {confidence:.0f}% confidence'
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return {
                'prediction': 'NEUTRAL',
                'breakout_probability': 0.33,
                'breakdown_probability': 0.33,
                'neutral_probability': 0.34,
                'confidence': 0.0,
                'ml_enabled': False,
                'reason': f'ML prediction failed: {str(e)}'
            }
    
    def learn_from_outcome(self, features: Dict, outcome: str):
        """
        Learn from a new signal outcome.
        
        Args:
            features: Technical indicators and patterns when signal was generated
            outcome: 'BREAKOUT', 'BREAKDOWN', or 'NEUTRAL'
        """
        # Save training sample
        training_sample = {
            'features': features,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_training_sample(training_sample)
        
        # Check if we have enough data to retrain
        training_data = self._load_training_data()
        if len(training_data) >= self.min_training_samples:
            print(f"ML: Retraining model with {len(training_data)} samples...")
            self.train_model(training_data)
    
    def train_model(self, training_data: List[Dict] = None):
        """
        Train the ML model on historical data.
        
        Args:
            training_data: List of training samples, or None to load from file
        """
        if training_data is None:
            training_data = self._load_training_data()
        
        if len(training_data) < self.min_training_samples:
            print(f"ML: Not enough training data ({len(training_data)}/{self.min_training_samples})")
            return False
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for sample in training_data:
                features = sample['features']
                outcome = sample['outcome']
                
                feature_vector = self._extract_feature_vector(features)
                X.append(feature_vector)
                
                # Map outcome to class (0=BREAKDOWN, 1=NEUTRAL, 2=BREAKOUT)
                if outcome == 'BREAKOUT':
                    y.append(2)
                elif outcome == 'BREAKDOWN':
                    y.append(0)
                else:
                    y.append(1)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train Random Forest model (good for pattern recognition)
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'  # Handle imbalanced data
            )
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Calculate accuracy
            train_accuracy = self.model.score(X_scaled, y)
            print(f"ML: Model trained! Accuracy: {train_accuracy*100:.1f}% on {len(training_data)} samples")
            
            # Save model
            self._save_model()
            
            return True
            
        except Exception as e:
            print(f"ML training error: {e}")
            return False
    
    def _extract_feature_vector(self, features: Dict) -> List[float]:
        """
        Extract numerical feature vector from feature dict.
        
        Features used:
        - RSI
        - MACD signal
        - Bollinger Band position
        - Volume ratio
        - Price momentum (1h, 24h, 7d)
        - Volatility
        - Trend strength
        """
        feature_vector = [
            features.get('rsi', 50.0),
            features.get('macd', 0.0),
            features.get('bb_position', 0.5),  # 0-1, where in BB range
            features.get('volume_ratio', 1.0),  # Current vs average volume
            features.get('price_change_1h', 0.0),
            features.get('price_change_24h', 0.0),
            features.get('price_change_7d', 0.0),
            features.get('volatility', 0.0),  # Price volatility
            features.get('trend_strength', 0.0),  # -1 to 1
            features.get('volume_trend', 0.0),  # Volume momentum
        ]
        
        # Handle missing values
        feature_vector = [f if f is not None and not np.isnan(f) else 0.0 for f in feature_vector]
        
        return feature_vector
    
    def _save_training_sample(self, sample: Dict):
        """Save a new training sample to file"""
        training_data = self._load_training_data()
        training_data.append(sample)
        
        # Keep only last 1000 samples to avoid file getting too large
        if len(training_data) > 1000:
            training_data = training_data[-1000:]
        
        with open(self.training_data_path, 'w') as f:
            json.dump(training_data, f, indent=2)
    
    def _load_training_data(self) -> List[Dict]:
        """Load training data from file"""
        if not os.path.exists(self.training_data_path):
            return []
        
        try:
            with open(self.training_data_path, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def _save_model(self):
        """Save trained model and scaler to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"ML: Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load trained model and scaler from disk"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                print("ML: Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_trained = False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        Shows which indicators are most predictive of breakouts/breakdowns.
        """
        if not self.is_trained:
            return {}
        
        feature_names = [
            'RSI', 'MACD', 'BB Position', 'Volume Ratio',
            'Price 1h', 'Price 24h', 'Price 7d',
            'Volatility', 'Trend Strength', 'Volume Trend'
        ]
        
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))
    
    def create_synthetic_training_data(self):
        """
        Create synthetic training data for initial model training.
        This gives the model a head start before real data accumulates.
        """
        synthetic_samples = []
        
        # Breakout patterns
        for _ in range(10):
            synthetic_samples.append({
                'features': {
                    'rsi': np.random.uniform(60, 80),
                    'macd': np.random.uniform(0.5, 2.0),
                    'bb_position': np.random.uniform(0.7, 1.0),
                    'volume_ratio': np.random.uniform(2.0, 5.0),
                    'price_change_1h': np.random.uniform(2, 10),
                    'price_change_24h': np.random.uniform(5, 20),
                    'price_change_7d': np.random.uniform(10, 50),
                    'volatility': np.random.uniform(0.02, 0.1),
                    'trend_strength': np.random.uniform(0.5, 1.0),
                    'volume_trend': np.random.uniform(0.3, 1.0),
                },
                'outcome': 'BREAKOUT',
                'timestamp': datetime.now().isoformat()
            })
        
        # Breakdown patterns
        for _ in range(10):
            synthetic_samples.append({
                'features': {
                    'rsi': np.random.uniform(20, 40),
                    'macd': np.random.uniform(-2.0, -0.5),
                    'bb_position': np.random.uniform(0.0, 0.3),
                    'volume_ratio': np.random.uniform(2.0, 5.0),
                    'price_change_1h': np.random.uniform(-10, -2),
                    'price_change_24h': np.random.uniform(-20, -5),
                    'price_change_7d': np.random.uniform(-50, -10),
                    'volatility': np.random.uniform(0.02, 0.1),
                    'trend_strength': np.random.uniform(-1.0, -0.5),
                    'volume_trend': np.random.uniform(-1.0, -0.3),
                },
                'outcome': 'BREAKDOWN',
                'timestamp': datetime.now().isoformat()
            })
        
        # Neutral patterns
        for _ in range(10):
            synthetic_samples.append({
                'features': {
                    'rsi': np.random.uniform(40, 60),
                    'macd': np.random.uniform(-0.5, 0.5),
                    'bb_position': np.random.uniform(0.3, 0.7),
                    'volume_ratio': np.random.uniform(0.5, 2.0),
                    'price_change_1h': np.random.uniform(-2, 2),
                    'price_change_24h': np.random.uniform(-5, 5),
                    'price_change_7d': np.random.uniform(-10, 10),
                    'volatility': np.random.uniform(0.01, 0.05),
                    'trend_strength': np.random.uniform(-0.3, 0.3),
                    'volume_trend': np.random.uniform(-0.3, 0.3),
                },
                'outcome': 'NEUTRAL',
                'timestamp': datetime.now().isoformat()
            })
        
        # Save synthetic data
        for sample in synthetic_samples:
            self._save_training_sample(sample)
        
        print(f"ML: Created {len(synthetic_samples)} synthetic training samples")
        
        # Train initial model
        self.train_model()
