from flask import Flask, request, render_template, jsonify, send_file
import os
import logging
import traceback
from pathlib import Path
import zipfile
from datetime import datetime

from .auto_lab import run_auto_lab
from .analyze_results import load_all_runs

# Configure logging for debug
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

logger.info("🚀 Starting Prometheus Lab Flask App")
logger.info(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
logger.info(f"📁 Results directory: {RESULTS_DIR.absolute()}")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
logger.info("✅ Directories created successfully")

# Check environment variables
env_vars = ['GROQ_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']
for var in env_vars:
    value = os.getenv(var)
    if value:
        logger.info(f"✅ {var} is set ({value[:20]}...)")
    else:
        logger.warning(f"⚠️  {var} is not set")

logger.info("🎯 Flask app initialization complete")

@app.route('/', methods=['GET'])
def index():
    logger.info("🏠 GET / - Serving homepage")
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>Autonomous Fairness Agent</title></head>
    <body>
        <h1>🚀 Autonomous Fairness Agent</h1>
        <p>Upload ANY CSV → Auto-detect groups → Optimize worst_group_accuracy</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="csv_file" accept=".csv" required>
            <button type="submit">Analyze & Optimize</button>
        </form>
    </body>
    </html>
    '''

@app.route('/', methods=['POST'])
def upload_csv():
    logger.info("📤 POST / - File upload request received")

    if 'csv_file' not in request.files:
        logger.error("❌ No file uploaded in request")
        return "No file uploaded", 400

    file = request.files['csv_file']
    if file.filename == '':
        logger.error("❌ No file selected")
        return "No file selected", 400

    logger.info(f"📄 File received: {file.filename}")

    # Save uploaded CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"
    logger.info(f"💾 Saving file to: {csv_path}")

    try:
        file.save(csv_path)
        logger.info(f"✅ File saved successfully: {csv_path.stat().st_size} bytes")
    except Exception as e:
        logger.error(f"❌ Failed to save file: {e}")
        return f"Failed to save file: {str(e)}", 500

    try:
        # Run autonomous agent
        logger.info(f"🤖 Starting auto_lab analysis on {csv_path}")
        run_auto_lab(str(csv_path))
        logger.info("✅ auto_lab completed successfully")

        # Collect results
        logger.info("📊 Loading experiment results")
        runs = load_all_runs()
        logger.info(f"📈 Found {len(runs)} experiment runs")

        if not runs:
            logger.error("❌ No experiment runs found")
            return "No experiment results found", 500

        best_run = max(runs, key=lambda r: r["ood"]["worst_group_accuracy"])
        logger.info(f"🏆 Best run: worst_group_accuracy = {best_run['ood']['worst_group_accuracy']:.4f}")

        return f'''
        <!DOCTYPE html>
        <html>
        <head><title>Results</title></head>
        <body>
            <h1>✅ Analysis Complete!</h1>
            <h2>Dataset: {csv_path.name}</h2>

            <h3>Best Strategy</h3>
            <pre>{best_run["config"]}</pre>

            <h3>Performance</h3>
            <p>ID Accuracy: {best_run["id"]["accuracy"]:.4f}</p>
            <p>OOD Accuracy: {best_run["ood"]["accuracy"]:.4f}</p>
            <p><b>Worst Group Accuracy: {best_run["ood"]["worst_group_accuracy"]:.4f}</b></p>

            <h3>Group Breakdown</h3>
            <pre>{best_run["ood"]["group_accuracy"]}</pre>

            <a href="/download/{timestamp}" class="btn">📥 Download Results</a>
        </body>
        </html>
        '''
    except Exception as e:
        logger.error(f"❌ Error during processing: {str(e)}")
        logger.error(f"📋 Full traceback:\n{traceback.format_exc()}")
        return f"Error: {str(e)}", 500

@app.route('/download/<timestamp>')
def download_results(timestamp):
    logger.info(f"📥 Download request for timestamp: {timestamp}")
    zip_path = RESULTS_DIR / f"{timestamp}_results.zip"

    try:
        with zipfile.ZipFile(zip_path, 'w') as zf:
            # Add result files
            result_files = list(RESULTS_DIR.glob("run_*.json"))
            logger.info(f"📋 Adding {len(result_files)} result files to zip")
            for run_file in result_files:
                zf.write(run_file, run_file.name)
                logger.debug(f"   Added: {run_file.name}")

            # Add input CSV
            csv_files = list(UPLOAD_DIR.glob(f"{timestamp}_*.csv"))
            logger.info(f"📄 Adding {len(csv_files)} CSV files to zip")
            for csv_file in csv_files:
                zf.write(csv_file, f"{timestamp}_input.csv")
                logger.debug(f"   Added: {csv_file.name}")

        logger.info(f"✅ Zip created successfully: {zip_path}")
        return send_file(zip_path, as_attachment=True)

    except Exception as e:
        logger.error(f"❌ Failed to create download zip: {e}")
        logger.error(f"📋 Full traceback:\n{traceback.format_exc()}")
        return f"Download failed: {str(e)}", 500

# For local development, use: python run.py from the project root
# For production, use: gunicorn src.app:app
