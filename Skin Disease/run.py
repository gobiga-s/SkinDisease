"""
Runner script for SkinX Application
Provides easy startup and configuration options
"""

import os
import sys
import argparse
import logging
from app import app

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('skinx.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'tensorflow', 'torch', 'transformers', 'flask', 
        'opencv-python', 'scipy', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_model_files():
    """Check if model files exist (will create dummy models if not)"""
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 Created models directory: {model_dir}")
    
    # Check for model weights (these would need to be trained/downloaded)
    efficientnet_model = os.path.join(model_dir, "efficientnet_skin_classifier.h5")
    biobert_model = os.path.join(model_dir, "biobert_skin_classifier")
    
    if not os.path.exists(efficientnet_model):
        print(f"⚠️  EfficientNet model not found at {efficientnet_model}")
        print("   The application will run with dummy predictions")
    
    if not os.path.exists(biobert_model):
        print(f"⚠️  BioBERT model not found at {biobert_model}")
        print("   The application will run with dummy predictions")

def create_directories():
    """Create necessary directories"""
    directories = [
        "static/uploads",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def main():
    """Main function to run the SkinX application"""
    parser = argparse.ArgumentParser(description='SkinX Skin Disease Prediction System')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies and exit')
    parser.add_argument('--no-check', action='store_true', help='Skip dependency checks')
    
    args = parser.parse_args()
    
    print("🩺 SkinX: AI-Powered Skin Disease Prediction System")
    print("=" * 60)
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create necessary directories
    create_directories()
    
    # Check dependencies
    if not args.no_check:
        if not check_dependencies():
            sys.exit(1)
        
        check_model_files()
        
        if args.check_deps:
            print("✅ Dependency check completed")
            sys.exit(0)
    
    # Print startup information
    print(f"🚀 Starting SkinX server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Debug: {args.debug}")
    print(f"   Log Level: {args.log_level}")
    print("=" * 60)
    print("🌐 Open your browser and navigate to:")
    if args.host == '0.0.0.0':
        print(f"   http://localhost:{args.port}")
        print(f"   http://127.0.0.1:{args.port}")
    else:
        print(f"   http://{args.host}:{args.port}")
    print("=" * 60)
    print("⚠️  Medical Disclaimer: This system is for educational purposes only")
    print("   and should not replace professional medical advice.")
    print("=" * 60)
    
    try:
        # Run the Flask application
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down SkinX server...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
