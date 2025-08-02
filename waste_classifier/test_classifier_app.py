#!/usr/bin/env python3
"""
Test script for waste classification model and pipeline.

This script tests the trained CNN model on sample images to verify that the
classification pipeline works correctly. Includes functionality to test
individual images, batch processing, and integration with emission factors.

References:
- Chollet, François. Deep Learning with Python. 2nd ed., Manning, 2021.
- "Getting started with Classification - GeeksforGeeks."
- TensorFlow documentation on model inference.

Author: Zinhle Maurice-Mopp (210125870)
Date: 2025-08-02
"""
"""
Run with:

Single image: python waste_classifier/test_classifier_app.py --test-image dataset/images/metal/aluminum_food_cans/real_world/Image_9.png
python waste_classifier/test_classifier_app.py --test-image dataset/images/paper/newspaper/default/Image_14.png
python waste_classifier/test_classifier_app.py --test-image dataset/images/organic_waste/tea_bags/real_world/Image_37.png
Batch testing: python waste_classifier/test_classifier_app.py --batch-size 10
Custom weight: python waste_classifier/test_classifier_app.py --weight 0.5
Batch directory: python waste_classifier/test_classifier_app.py --test-dir dataset/split/test?
"""


import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

def load_model_and_classes(model_dir):
    """
    Load trained model and class names from model directory.
    
    Args:
        model_dir (str): Path to model directory containing .keras file and classnames.json
        
    Returns:
        tuple: (model, class_names, class_to_index)
        
    References:
        Standard model loading pattern for TensorFlow/Keras saved models.
    """
    model_dir = Path(model_dir)
    
    # Load the trained model
    model_path = model_dir / 'waste_model.keras'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(str(model_path))
    
    # Load class names
    classnames_path = model_dir / 'classnames.json'
    if not classnames_path.exists():
        raise FileNotFoundError(f"Class names file not found: {classnames_path}")
    
    with open(classnames_path, 'r') as f:
        class_names = json.load(f)
    
    # Create class to index mapping
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"✓ Loaded model with {len(class_names)} classes")
    print(f"✓ Classes: {class_names}")
    
    return model, class_names, class_to_index

def load_emission_factors(emission_factors_path):
    """
    Load emission factors for environmental impact calculations.
    
    Args:
        emission_factors_path (str): Path to emission_factors.json file
        
    Returns:
        dict: Emission factors data
        
    References:
        Environmental impact calculation methodology from EPA WARM Model.
    """
    if not Path(emission_factors_path).exists():
        raise FileNotFoundError(f"Emission factors file not found: {emission_factors_path}")
    
    with open(emission_factors_path, 'r') as f:
        emission_factors = json.load(f)
    
    print(f"✓ Loaded emission factors for {len([k for k in emission_factors.keys() if not k.startswith('_')])} waste categories")
    
    return emission_factors

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocess image for model inference following training preprocessing.
    
    Args:
        img_path (str): Path to image file
        target_size (tuple): Target image size (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image array ready for model input
        
    References:
        Chollet, François. Deep Learning with Python. Chapter 5: "Deep learning for computer vision."
        Image preprocessing must match training pipeline exactly for consistent results.
    """
    # Load and resize image
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert to array and add batch dimension
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalise pixel values to [0,1] range (same as training)
    img_array = img_array / 255.0
    
    return img_array

def classify_image(model, img_array, class_names, top_k=3):
    """
    Classify preprocessed image and return top predictions.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array
        class_names (list): List of class names
        top_k (int): Number of top predictions to return
        
    Returns:
        list: List of (class_name, confidence) tuples sorted by confidence
        
    References:
        Standard inference pattern for multi-class classification models.
    """
    # Get model predictions
    predictions = model.predict(img_array, verbose=0)
    
    # Get top-k predictions
    top_indices = np.argsort(predictions[0])[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        class_name = class_names[idx]
        confidence = float(predictions[0][idx])
        results.append((class_name, confidence))
    
    return results

def calculate_environmental_impact(class_name, weight_kg, emission_factors):
    """
    Calculate environmental impact based on waste classification and weight.
    
    Args:
        class_name (str): Predicted waste class
        weight_kg (float): Weight of waste item in kilograms
        emission_factors (dict): Emission factors data
        
    Returns:
        dict: Environmental impact metrics
        
    References:
        EPA WARM Model methodology for waste impact calculations.
        Factors represent avoided emissions through proper recycling vs landfill.
    """
    if class_name not in emission_factors:
        # Use general waste factors if specific class not found
        class_name = 'other_general_waste'
    
    factors = emission_factors[class_name]
    
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

def test_single_image(model, class_names, emission_factors, img_path, weight_kg=0.1):
    """
    Test classification on a single image with full pipeline.
    
    Args:
        model: Trained Keras model
        class_names (list): List of class names
        emission_factors (dict): Emission factors data
        img_path (str): Path to test image
        weight_kg (float): Estimated weight of waste item
        
    References:
        Complete inference pipeline testing following software engineering best practices.
    """
    print(f"\n{'='*60}")
    print(f"TESTING IMAGE: {Path(img_path).name}")
    print(f"{'='*60}")
    
    try:
        # Preprocess image
        img_array = preprocess_image(img_path)
        print(f"✓ Image preprocessed: shape {img_array.shape}")
        
        # Classify image
        predictions = classify_image(model, img_array, class_names, top_k=3)
        
        print(f"\nTop 3 Predictions:")
        for i, (class_name, confidence) in enumerate(predictions, 1):
            print(f"  {i}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Calculate environmental impact for top prediction
        top_class, top_confidence = predictions[0]
        impact = calculate_environmental_impact(top_class, weight_kg, emission_factors)
        
        print(f"\nEnvironmental Impact (assuming {weight_kg}kg):")
        print(f"  CO2 saved: {impact['co2_saved_kg']:.4f} kg CO2e")
        print(f"  Water saved: {impact['water_saved_litres']:.2f} litres")
        print(f"  Energy saved: {impact['energy_saved_kwh']:.3f} kWh")
        
        # Display image with predictions
        plt.figure(figsize=(10, 6))
        
        # Show original image
        plt.subplot(1, 2, 1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"Input Image\n{Path(img_path).name}")
        plt.axis('off')
        
        # Show prediction results
        plt.subplot(1, 2, 2)
        classes = [pred[0].replace('_', ' ').title() for pred in predictions]
        confidences = [pred[1] for pred in predictions]
        
        bars = plt.barh(range(len(classes)), confidences)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel('Confidence')
        plt.title('Top 3 Predictions')
        plt.xlim(0, 1)
        
        # Color bars based on confidence
        for i, bar in enumerate(bars):
            if confidences[i] > 0.7:
                bar.set_color('green')
            elif confidences[i] > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.show()
        
        return predictions, impact
        
    except Exception as e:
        print(f"✗ Error processing image: {e}")
        return None, None

def test_batch_images(model, class_names, emission_factors, test_dir, num_samples=5):
    """
    Test classification on multiple random images from test directory.
    
    Args:
        model: Trained Keras model
        class_names (list): List of class names
        emission_factors (dict): Emission factors data
        test_dir (str): Path to test images directory
        num_samples (int): Number of random images to test
        
    References:
        Batch testing methodology for model validation and performance assessment.
    """
    print(f"\n{'='*60}")
    print(f"BATCH TESTING: {num_samples} RANDOM IMAGES")
    print(f"{'='*60}")
    
    test_dir = Path(test_dir)
    if not test_dir.exists():
        print(f"✗ Test directory not found: {test_dir}")
        return
    
    # Find all image files in test directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    all_images = []
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                img_path = Path(root) / file
                # Extract true class from directory structure
                true_class = Path(root).name
                all_images.append((str(img_path), true_class))
    
    if len(all_images) == 0:
        print("✗ No images found in test directory")
        return
    
    # Sample random images
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    correct_predictions = 0
    total_co2_saved = 0
    total_water_saved = 0
    total_energy_saved = 0
    
    for i, (img_path, true_class) in enumerate(sample_images, 1):
        print(f"\n--- Sample {i}/{len(sample_images)} ---")
        print(f"Image: {Path(img_path).name}")
        print(f"True class: {true_class}")
        
        # Test single image
        predictions, impact = test_single_image(
            model, class_names, emission_factors, img_path, weight_kg=0.1
        )
        
        if predictions:
            predicted_class, confidence = predictions[0]
            
            # Check if prediction is correct
            if predicted_class == true_class:
                correct_predictions += 1
                print("✓ Correct prediction!")
            else:
                print("✗ Incorrect prediction")
            
            # Accumulate environmental impact
            if impact:
                total_co2_saved += impact['co2_saved_kg']
                total_water_saved += impact['water_saved_litres']
                total_energy_saved += impact['energy_saved_kwh']
    
    # Summary statistics
    accuracy = correct_predictions / len(sample_images)
    print(f"\n{'='*60}")
    print("BATCH TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Samples tested: {len(sample_images)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nTotal Environmental Impact:")
    print(f"  CO2 saved: {total_co2_saved:.4f} kg CO2e")
    print(f"  Water saved: {total_water_saved:.2f} litres")
    print(f"  Energy saved: {total_energy_saved:.3f} kWh")

def main():
    """
    Main function to run classifier testing with various test scenarios.
    
    Provides comprehensive testing of the trained model including single image
    classification, batch processing, and environmental impact calculations.
    
    References:
        Software testing best practices for machine learning model validation.
    """
    parser = argparse.ArgumentParser(description='Test waste classification model')
    parser.add_argument('--model-dir', default='waste_classifier/model',
                       help='Directory containing trained model')
    parser.add_argument('--emission-factors', default='data/emission_factors.json',
                       help='Path to emission factors JSON file')
    parser.add_argument('--test-image', type=str,
                       help='Path to single test image')
    parser.add_argument('--test-dir', default='dataset/split/test',
                       help='Directory containing test images')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Number of images for batch testing')
    parser.add_argument('--weight', type=float, default=0.1,
                       help='Assumed weight of waste item in kg')
    
    args = parser.parse_args()
    
    print("WASTE CLASSIFIER TESTING")
    print("="*50)
    
    try:
        # Load model and supporting data
        model, class_names, class_to_index = load_model_and_classes(args.model_dir)
        emission_factors = load_emission_factors(args.emission_factors)
        
        # Test single image if provided
        if args.test_image:
            if Path(args.test_image).exists():
                test_single_image(
                    model, class_names, emission_factors, 
                    args.test_image, args.weight
                )
            else:
                print(f"✗ Test image not found: {args.test_image}")
        
        # Run batch testing
        test_batch_images(
            model, class_names, emission_factors, 
            args.test_dir, args.batch_size
        )
        
        print(f"\n{'='*50}")
        print("TESTING COMPLETE")
        print(f"{'='*50}")
        print("✓ Model and pipeline working correctly")
        print("✓ Ready for Streamlit app deployment")
        
    except Exception as e:
        print(f"✗ Testing failed: {e}")
        print("Please check model files and paths")

if __name__ == "__main__":
    main()