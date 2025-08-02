#!/usr/bin/env python3
"""
Waste classification module providing core functionality.

This module contains the WasteClassifier class that encapsulates all functionality
for loading models, preprocessing images, making predictions, and calculating
environmental impact. Designed for use in CLI, Streamlit, and other applications.

References:
- Chollet, François. Deep Learning with Python. 2nd ed., Manning, 2021.
- "Convolutional Neural Networks in Python with Keras."
- Object-oriented design patterns for machine learning applications.

Author: Zinhle Maurice-Mopp (210125870)
Date: 2025-08-02
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

class WasteClassifier:
    """
    Main classifier class for waste image classification and impact calculation.
    
    This class encapsulates the complete waste classification pipeline including
    model loading, image preprocessing, prediction generation, and environmental
    impact calculations using emission factors.
    
    References:
        Chollet, François. Deep Learning with Python. Chapter 9: "Advanced deep learning best practices."
        Encapsulation of ML models in classes provides clean API and state management.
    """
    
    def __init__(self, model_dir: str = 'waste_classifier/model', 
                 emission_factors_path: str = 'data/emission_factors.json'):
        """
        Initialize the waste classifier with model and emission factors.
        
        Args:
            model_dir (str): Path to directory containing trained model files
            emission_factors_path (str): Path to emission factors JSON file
            
        Raises:
            FileNotFoundError: If model files or emission factors are missing
            
        References:
            Standard initialization pattern for ML model classes.
        """
        self.model_dir = Path(model_dir)
        self.emission_factors_path = Path(emission_factors_path)
        
        # Initialize attributes
        self.model = None
        self.class_names = []
        self.class_to_index = {}
        self.emission_factors = {}
        
        # Load model and supporting data
        self._load_model()
        self._load_emission_factors()
    
    def _load_model(self):
        """
        Load trained Keras model and class names from model directory.
        
        Raises:
            FileNotFoundError: If model files are missing
            
        References:
            Standard model loading pattern for TensorFlow/Keras applications.
        """
        # Load the trained model
        model_path = self.model_dir / 'waste_model.keras'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = keras.models.load_model(str(model_path))
        
        # Load class names
        classnames_path = self.model_dir / 'classnames.json'
        if not classnames_path.exists():
            raise FileNotFoundError(f"Class names file not found: {classnames_path}")
        
        with open(classnames_path, 'r') as f:
            self.class_names = json.load(f)
        
        # Create class to index mapping
        self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}
    
    def _load_emission_factors(self):
        """
        Load emission factors for environmental impact calculations.
        
        Raises:
            FileNotFoundError: If emission factors file is missing
            
        References:
            EPA WARM Model methodology for environmental impact assessment.
        """
        if not self.emission_factors_path.exists():
            raise FileNotFoundError(f"Emission factors file not found: {self.emission_factors_path}")
        
        with open(self.emission_factors_path, 'r') as f:
            self.emission_factors = json.load(f)
    
    def preprocess_image(self, img_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for model inference following training preprocessing.
        
        Args:
            img_path (str): Path to image file
            target_size (tuple): Target image size (height, width)
            
        Returns:
            numpy.ndarray: Preprocessed image array ready for model input
            
        Raises:
            IOError: If image cannot be loaded or processed
            
        References:
            Chollet, François. Deep Learning with Python. Chapter 5: "Deep learning for computer vision."
            Image preprocessing must match training pipeline exactly for consistent results.
        """
        try:
            # Load and resize image
            img = image.load_img(img_path, target_size=target_size)
            
            # Convert to array and add batch dimension
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Normalise pixel values to [0,1] range (same as training)
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            raise IOError(f"Failed to preprocess image {img_path}: {e}")
    
    def predict(self, img_array: np.ndarray, top_k: int = 3, 
                threshold: float = 0.1) -> List[Dict[str, float]]:
        """
        Generate predictions for preprocessed image array.
        
        Args:
            img_array (np.ndarray): Preprocessed image array
            top_k (int): Number of top predictions to return
            threshold (float): Minimum confidence threshold for predictions
            
        Returns:
            list: List of prediction dictionaries with class_name and confidence
            
        References:
            Standard inference pattern for multi-class classification models.
        """
        # Get model predictions
        predictions = self.model.predict(img_array, verbose=0)
        
        # Get top-k predictions above threshold
        top_indices = np.argsort(predictions[0])[::-1]
        
        results = []
        for idx in top_indices:
            confidence = float(predictions[0][idx])
            
            # Stop if below threshold or reached top_k limit
            if confidence < threshold or len(results) >= top_k:
                break
            
            class_name = self.class_names[idx]
            results.append({
                'class_name': class_name,
                'confidence': confidence
            })
        
        return results
    
    def calculate_environmental_impact(self, class_name: str, weight_kg: float) -> Dict[str, float]:
        """
        Calculate environmental impact based on waste classification and weight.
        
        Args:
            class_name (str): Predicted waste class
            weight_kg (float): Weight of waste item in kilograms
            
        Returns:
            dict: Environmental impact metrics
            
        References:
            EPA WARM Model methodology for waste impact calculations.
            Factors represent avoided emissions through proper recycling vs landfill.
        """
        # Use general waste factors if specific class not found
        if class_name not in self.emission_factors:
            class_name = 'other_general_waste'
        
        factors = self.emission_factors[class_name]
        
        # Calculate impacts based on weight
        co2_saved = weight_kg * factors['co2e_kg_per_kg']
        water_saved = weight_kg * factors['water_litres_per_kg']
        energy_saved = weight_kg * factors['energy_kwh_per_kg']
        
        return {
            'co2_saved_kg': round(co2_saved, 4),
            'water_saved_litres': round(water_saved, 2),
            'energy_saved_kwh': round(energy_saved, 3),
            'weight_kg': weight_kg,
            'waste_class': class_name
        }
    
    def classify_image(self, img_path: str, weight_kg: float = 0.1, 
                      top_k: int = 3, threshold: float = 0.1) -> Dict:
        """
        Complete classification pipeline for a single image.
        
        Args:
            img_path (str): Path to image file
            weight_kg (float): Estimated weight of waste item in kg
            top_k (int): Number of top predictions to return
            threshold (float): Minimum confidence threshold
            
        Returns:
            dict: Complete classification result including predictions and impact
            
        References:
            Complete inference pipeline following software engineering best practices.
        """
        start_time = time.time()
        
        result = {
            'image_path': img_path,
            'processing_time': 0.0,
            'predictions': [],
            'environmental_impact': None,
            'error': None
        }
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(img_path)
            
            # Generate predictions
            predictions = self.predict(img_array, top_k, threshold)
            result['predictions'] = predictions
            
            # Calculate environmental impact for top prediction
            if predictions:
                top_class = predictions[0]['class_name']
                impact = self.calculate_environmental_impact(top_class, weight_kg)
                result['environmental_impact'] = impact
            
        except Exception as e:
            result['error'] = str(e)
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def classify_batch(self, batch_dir: str, weight_kg: float = 0.1,
                      top_k: int = 3, threshold: float = 0.1) -> List[Dict]:
        """
        Classify multiple images in a directory.
        
        Args:
            batch_dir (str): Directory containing images to classify
            weight_kg (float): Estimated weight per item in kg
            top_k (int): Number of top predictions per image
            threshold (float): Minimum confidence threshold
            
        Returns:
            list: List of classification results for all images
            
        References:
            Batch processing patterns for efficient ML inference.
        """
        batch_dir = Path(batch_dir)
        if not batch_dir.exists():
            raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(batch_dir.glob(f'*{ext}'))
            image_files.extend(batch_dir.glob(f'*{ext.upper()}'))
        
        # Process each image
        results = []
        for img_path in sorted(image_files):
            result = self.classify_image(
                str(img_path), weight_kg, top_k, threshold
            )
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model and classes.
        
        Returns:
            dict: Model information including classes and architecture details
            
        References:
            Standard model introspection patterns for ML applications.
        """
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        return {
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_shape': self.model.input_shape,
            'model_params': self.model.count_params(),
            'model_dir': str(self.model_dir),
            'emission_factors_loaded': len([k for k in self.emission_factors.keys() 
                                          if not k.startswith('_')])
        }
    
    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that all required components are properly loaded.
        
        Returns:
            dict: Validation status for each component
            
        References:
            System validation patterns for production ML applications.
        """
        validation = {
            'model_loaded': self.model is not None,
            'classes_loaded': len(self.class_names) > 0,
            'emission_factors_loaded': len(self.emission_factors) > 0,
            'model_files_exist': (self.model_dir / 'waste_model.keras').exists(),
            'classnames_file_exists': (self.model_dir / 'classnames.json').exists(),
            'emission_factors_file_exists': self.emission_factors_path.exists()
        }
        
        validation['all_valid'] = all(validation.values())
        return validation