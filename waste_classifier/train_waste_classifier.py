#!/usr/bin/env python3
"""
Waste Classification CNN Training Script

This script trains a convolutional neural network for waste classification using
TensorFlow/Keras. Implements data augmentation, transfer learning, and proper
validation techniques following deep learning best practices.

References:
- Chollet, François. Deep Learning with Python. 2nd ed., Manning, 2021.
- "Convolutional Neural Network (CNN) in Machine Learning - GeeksforGeeks."
- "Introduction to Convolution Neural Network - GeeksforGeeks."
- "Epoch in Machine Learning - GeeksforGeeks."

Author: Zinhle Maurice-Mopp (210125870)
Date: 2025-08-02
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    """
    Create data generators with augmentation for training and validation.
    
    Args:
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory  
        test_dir (str): Path to test data directory
        img_size (tuple): Target image size (height, width)
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (train_generator, val_generator, test_generator, class_names)
        
    References:
        Chollet, François. Deep Learning with Python. Chapter 5: "Deep learning for computer vision."
        Data augmentation prevents overfitting by artificially expanding training dataset.
    """
    # Data augmentation for training set only
    # Horizontal flip, rotation, zoom, and shift help model generalise
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalise pixel values to [0,1] range
        rotation_range=20,           # Random rotation up to 20 degrees
        width_shift_range=0.2,       # Random horizontal shift up to 20%
        height_shift_range=0.2,      # Random vertical shift up to 20%
        shear_range=0.2,            # Shear transformation for geometric variation
        zoom_range=0.2,             # Random zoom in/out up to 20%
        horizontal_flip=True,        # Random horizontal flip (waste items can be flipped)
        fill_mode='nearest'          # Fill pixels after transformation
    )
    
    # Validation and test sets only need rescaling (no augmentation)
    # This ensures consistent evaluation metrics
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators from directory structure
    # Assumes directory structure: split_dir/class_name/image_files
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',    # Multi-class classification
        shuffle=True,               # Shuffle training data each epoch
        seed=42                     # Reproducible shuffling
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,              # Don't shuffle validation data
        seed=42
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,              # Don't shuffle test data
        seed=42
    )
    
    # Extract class names from generator
    class_names = list(train_generator.class_indices.keys())
    
    print(f"✓ Found {len(class_names)} classes: {class_names}")
    print(f"✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Test samples: {test_generator.samples}")
    
    return train_generator, val_generator, test_generator, class_names

def create_cnn_model(num_classes, img_size=(224, 224), use_transfer_learning=True):
    """
    Create CNN model architecture for waste classification.
    
    Args:
        num_classes (int): Number of waste categories to classify
        img_size (tuple): Input image dimensions (height, width)
        use_transfer_learning (bool): Whether to use pre-trained MobileNetV2
        
    Returns:
        keras.Model: Compiled CNN model ready for training
        
    References:
        Chollet, François. Deep Learning with Python. Chapter 8: "Introduction to deep learning for computer vision."
        MobileNetV2 provides efficient feature extraction pre-trained on ImageNet.
        Transfer learning reduces training time and improves performance on small datasets.
    """
    if use_transfer_learning:
        # Use MobileNetV2 as base model (efficient for mobile deployment)
        # Pre-trained on ImageNet provides good feature representations
        base_model = MobileNetV2(
            weights='imagenet',          # Use ImageNet pre-trained weights
            include_top=False,           # Exclude final classification layer
            input_shape=(*img_size, 3)   # RGB images
        )
        
        # Freeze base model weights initially
        # This preserves learned ImageNet features during early training
        base_model.trainable = False
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),  # Reduce spatial dimensions
            layers.Dropout(0.2),              # Prevent overfitting
            layers.Dense(128, activation='relu'),  # Feature combination layer
            layers.Dropout(0.2),              # Additional regularisation
            layers.Dense(num_classes, activation='softmax')  # Multi-class output
        ])
        
        print("✓ Created transfer learning model with MobileNetV2 base")
        
    else:
        # Build CNN from scratch following standard architecture patterns
        # Alternating convolution-pooling blocks with increasing filter counts
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
            layers.BatchNormalization(),      # Normalise activations for stable training
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block  
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),              # Higher dropout for scratch training
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        print("✓ Created CNN model from scratch")
    
    # Compile model with appropriate loss function and metrics
    # Categorical crossentropy for multi-class classification
    # Adam optimiser with default learning rate (0.001)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']  # Track top-3 accuracy for multi-class
    )
    
    return model

def create_callbacks(model_save_path):
    """
    Create training callbacks for monitoring and model improvement.
    
    Args:
        model_save_path (str): Path to save best model weights
        
    Returns:
        list: List of Keras callbacks
        
    References:
        Chollet, François. Deep Learning with Python. Chapter 7: "Working with Keras."
        Callbacks provide automated training control and prevent overfitting.
    """
    callbacks = [
        # Early stopping prevents overfitting by monitoring validation loss
        # Stops training when validation loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=10,                # Wait 10 epochs before stopping
            restore_best_weights=True,  # Restore weights from best epoch
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        # Helps model converge to better local minimum
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,                 # Reduce LR by factor of 5
            patience=5,                 # Wait 5 epochs before reducing
            min_lr=1e-7,               # Minimum learning rate
            verbose=1
        ),
        
        # Save best model weights based on validation accuracy
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,    # Save entire model
            verbose=1
        )
    ]
    
    return callbacks

def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Keras training history object
        save_path (str): Optional path to save plot
        
    References:
        Standard practice for visualising training progress and detecting overfitting.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {save_path}")
    
    plt.show()

def evaluate_model(model, test_generator, class_names, save_dir=None):
    """
    Evaluate trained model on test set and generate classification report.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        class_names (list): List of class names
        save_dir (str): Directory to save evaluation results
        
    References:
        Standard evaluation metrics for multi-class classification problems.
    """
    print("\n" + "="*50)
    print("EVALUATING MODEL ON TEST SET")
    print("="*50)
    
    # Get predictions on test set
    test_generator.reset()  # Reset generator to start
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    
    # Calculate test accuracy
    test_accuracy = np.mean(predicted_classes == true_classes)
    print(f"\n✓ Test Accuracy: {test_accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_names,
        digits=4
    )
    print("\nClassification Report:")
    print(report)
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_dir:
        # Save classification report
        report_path = Path(save_dir) / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
            f.write(report)
        print(f"✓ Classification report saved to: {report_path}")
        
        # Save confusion matrix plot
        cm_path = Path(save_dir) / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {cm_path}")
    
    plt.show()
    
    return test_accuracy, report

def main():
    """
    Main training function implementing complete CNN training pipeline.
    
    Follows standard machine learning workflow: data loading, model creation,
    training with validation, and final evaluation on test set.
    
    References:
        "Introduction to Machine Learning - GeeksforGeeks."
        "What is Machine Learning Pipeline? - GeeksforGeeks."
    """
    parser = argparse.ArgumentParser(description='Train waste classification CNN')
    parser.add_argument('--data-dir', default='dataset/split',
                       help='Path to split dataset directory')
    parser.add_argument('--model-dir', default='waste_classifier/model',
                       help='Directory to save trained model')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size (square)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum training epochs')
    parser.add_argument('--transfer-learning', action='store_true', default=True,
                       help='Use transfer learning with MobileNetV2')
    parser.add_argument('--fine-tune', action='store_true', default=False,
                       help='Fine-tune pre-trained layers after initial training')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    # Validate data directories exist
    for dir_path in [train_dir, val_dir, test_dir]:
        if not dir_path.exists():
            raise FileNotFoundError(f"Data directory not found: {dir_path}")
    
    print("WASTE CLASSIFICATION CNN TRAINING")
    print("="*50)
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.epochs}")
    print(f"Transfer learning: {args.transfer_learning}")
    print(f"Fine-tuning: {args.fine_tune}")
    
    # Create data generators
    img_size = (args.img_size, args.img_size)
    train_gen, val_gen, test_gen, class_names = create_data_generators(
        train_dir, val_dir, test_dir, img_size, args.batch_size
    )
    
    # Save class names for inference
    classnames_path = model_dir / 'classnames.json'
    with open(classnames_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"✓ Class names saved to: {classnames_path}")
    
    # Create model
    num_classes = len(class_names)
    model = create_cnn_model(num_classes, img_size, args.transfer_learning)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Calculate steps per epoch
    steps_per_epoch = train_gen.samples // args.batch_size
    validation_steps = val_gen.samples // args.batch_size
    
    # Setup callbacks
    model_path = model_dir / 'waste_model.keras'
    callbacks = create_callbacks(str(model_path))
    
    # Train model
    print(f"\n{'='*50}")
    print("STARTING TRAINING")
    print(f"{'='*50}")
    
    start_time = datetime.now()
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.now() - start_time
    print(f"\n✓ Training completed in: {training_time}")
    
    # Fine-tuning phase (optional)
    if args.fine_tune and args.transfer_learning:
        print(f"\n{'='*50}")
        print("STARTING FINE-TUNING")
        print(f"{'='*50}")
        
        # Unfreeze base model for fine-tuning
        model.layers[0].trainable = True
        
        # Use lower learning rate for fine-tuning
        # This prevents destroying pre-trained features
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),  # 10x lower learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Fine-tune for fewer epochs
        fine_tune_epochs = args.epochs // 4
        
        history_fine = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=fine_tune_epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine training histories
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])
    
    # Plot training history
    plot_path = model_dir / 'training_history.png'
    plot_training_history(history, str(plot_path))
    
    # Load best model for evaluation
    print(f"\n✓ Loading best model from: {model_path}")
    best_model = keras.models.load_model(str(model_path))
    
    # Evaluate on test set
    test_accuracy, report = evaluate_model(
        best_model, test_gen, class_names, str(model_dir)
    )
    
    # Save training summary
    summary_path = model_dir / 'training_summary.json'
    summary = {
        'model_path': str(model_path),
        'classnames_path': str(classnames_path),
        'num_classes': num_classes,
        'class_names': class_names,
        'img_size': img_size,
        'batch_size': args.batch_size,
        'epochs_trained': len(history.history['loss']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'test_accuracy': float(test_accuracy),
        'training_time': str(training_time),
        'transfer_learning': args.transfer_learning,
        'fine_tuning': args.fine_tune,
        'created_at': datetime.now().isoformat()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"✓ Best model saved to: {model_path}")
    print(f"✓ Training summary saved to: {summary_path}")
    print(f"✓ Final test accuracy: {test_accuracy:.4f}")
    print(f"✓ Total training time: {training_time}")

if __name__ == "__main__":
    main()
