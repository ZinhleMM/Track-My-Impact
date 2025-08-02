#!/usr/bin/env python3
"""
Command-line interface for waste classification.

This script provides a simple CLI for classifying waste images using the trained
CNN model. Supports single image classification and batch processing with
environmental impact calculations.

References:
- Chollet, François. Deep Learning with Python. 2nd ed., Manning, 2021.
- "Getting started with Classification - GeeksforGeeks."
- Standard CLI design patterns for machine learning applications.

Author: Zinhle Maurice-Mopp (210125870)
Date: 2025-08-02
"""

import argparse
import sys
from pathlib import Path
import json

from classifier import WasteClassifier

def main():
    """
    Main CLI function for waste classification.
    
    Provides command-line interface for classifying waste images with options
    for single images, batch processing, and environmental impact reporting.
    
    References:
        Standard CLI design patterns following Unix philosophy of simple tools.
    """
    parser = argparse.ArgumentParser(
        description='Classify waste images using trained CNN model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python waste_classifier/classify.py --image photo.jpg
  python waste_classifier/classify.py --batch-dir test_images/ --output results.json
  python waste_classifier/classify.py --image bottle.jpg --weight 0.5 --verbose
        """
    )
    
    # Input options
    parser.add_argument('--image', type=str,
                       help='Path to single image file to classify')
    parser.add_argument('--batch-dir', type=str,
                       help='Directory containing images for batch classification')
    
    # Model and data paths
    parser.add_argument('--model-dir', default='waste_classifier/model',
                       help='Directory containing trained model (default: waste_classifier/model)')
    parser.add_argument('--emission-factors', default='data/emission_factors.json',
                       help='Path to emission factors JSON file (default: data/emission_factors.json)')
    
    # Classification options
    parser.add_argument('--weight', type=float, default=0.1,
                       help='Estimated weight of waste item in kg (default: 0.1)')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top predictions to show (default: 3)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Minimum confidence threshold for predictions (default: 0.1)')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output including environmental impact')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress all output except errors')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.batch_dir:
        parser.error("Must specify either --image or --batch-dir")
    
    if args.image and args.batch_dir:
        parser.error("Cannot specify both --image and --batch-dir")
    
    if args.quiet and args.verbose:
        parser.error("Cannot specify both --quiet and --verbose")
    
    try:
        # Initialize classifier
        if not args.quiet:
            print("Loading waste classifier...")
        
        classifier = WasteClassifier(
            model_dir=args.model_dir,
            emission_factors_path=args.emission_factors
        )
        
        if not args.quiet:
            print(f"✓ Loaded model with {len(classifier.class_names)} classes")
        
        results = []
        
        # Single image classification
        if args.image:
            img_path = Path(args.image)
            if not img_path.exists():
                print(f"Error: Image file not found: {img_path}", file=sys.stderr)
                sys.exit(1)
            
            result = classifier.classify_image(
                str(img_path),
                weight_kg=args.weight,
                top_k=args.top_k,
                threshold=args.threshold
            )
            
            results.append(result)
            
            if not args.quiet:
                print_single_result(result, args.verbose)
        
        # Batch classification
        elif args.batch_dir:
            batch_dir = Path(args.batch_dir)
            if not batch_dir.exists():
                print(f"Error: Batch directory not found: {batch_dir}", file=sys.stderr)
                sys.exit(1)
            
            results = classifier.classify_batch(
                str(batch_dir),
                weight_kg=args.weight,
                top_k=args.top_k,
                threshold=args.threshold
            )
            
            if not args.quiet:
                print_batch_results(results, args.verbose)
        
        # Save results to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            if not args.quiet:
                print(f"\n✓ Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def print_single_result(result, verbose=False):
    """
    Print results for single image classification.
    
    Args:
        result (dict): Classification result from WasteClassifier
        verbose (bool): Whether to show detailed environmental impact
        
    References:
        Standard output formatting for CLI applications.
    """
    print(f"\nImage: {result['image_path']}")
    print(f"Processing time: {result['processing_time']:.3f}s")
    
    if result['error']:
        print(f"Error: {result['error']}")
        return
    
    print(f"\nTop {len(result['predictions'])} Predictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"  {i}. {pred['class_name'].replace('_', ' ').title()}: "
              f"{pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
    
    if verbose and result['environmental_impact']:
        impact = result['environmental_impact']
        print(f"\nEnvironmental Impact (assuming {impact['weight_kg']}kg):")
        print(f"  CO2 saved: {impact['co2_saved_kg']:.4f} kg CO2e")
        print(f"  Water saved: {impact['water_saved_litres']:.2f} litres")
        print(f"  Energy saved: {impact['energy_saved_kwh']:.3f} kWh")

def print_batch_results(results, verbose=False):
    """
    Print summary results for batch classification.
    
    Args:
        results (list): List of classification results
        verbose (bool): Whether to show detailed statistics
        
    References:
        Batch processing output patterns for CLI tools.
    """
    total_images = len(results)
    successful = len([r for r in results if not r['error']])
    failed = total_images - successful
    
    print(f"\nBatch Classification Summary:")
    print(f"  Total images: {total_images}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if successful > 0:
        avg_time = sum(r['processing_time'] for r in results if not r['error']) / successful
        print(f"  Average processing time: {avg_time:.3f}s")
        
        # Calculate total environmental impact
        total_co2 = sum(r['environmental_impact']['co2_saved_kg'] 
                       for r in results if r['environmental_impact'])
        total_water = sum(r['environmental_impact']['water_saved_litres'] 
                         for r in results if r['environmental_impact'])
        total_energy = sum(r['environmental_impact']['energy_saved_kwh'] 
                          for r in results if r['environmental_impact'])
        
        print(f"\nTotal Environmental Impact:")
        print(f"  CO2 saved: {total_co2:.4f} kg CO2e")
        print(f"  Water saved: {total_water:.2f} litres")
        print(f"  Energy saved: {total_energy:.3f} kWh")
    
    if verbose:
        print(f"\nDetailed Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {Path(result['image_path']).name}")
            if result['error']:
                print(f"   Error: {result['error']}")
            else:
                top_pred = result['predictions'][0]
                print(f"   Prediction: {top_pred['class_name'].replace('_', ' ').title()} "
                      f"({top_pred['confidence']*100:.1f}%)")

if __name__ == "__main__":
    main()