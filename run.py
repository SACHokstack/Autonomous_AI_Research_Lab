"""
Local development entry point for the Flask app.
For production deployment, use: gunicorn src.app:app
"""
import sys
import os
print(f"🚀 Starting Prometheus Lab Flask App")
print(f"📁 Working directory: {os.getcwd()}")
print(f"🐍 Python version: {sys.version}")
print(f"🌐 Will be available at: http://localhost:5000")
print("=" * 50)

from src.app import app

if __name__ == '__main__':
    print("✅ Flask app imported successfully")
    print("🔧 Starting development server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
