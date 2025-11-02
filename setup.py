import os
import subprocess
import sys

def setup_project():
    print("🚀 Setting up Crop Yield Prediction Project...")
    
    # Create directories
    directories = ['model', 'assets', 'utils', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}/")
    
    # Install requirements
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("🎉 Setup complete! Now you can:")
    print("  1. Export your model from the notebook")
    print("  2. Run: streamlit run app.py")
    print("  3. Open http://localhost:8501 in your browser")

if __name__ == "__main__":
    setup_project()