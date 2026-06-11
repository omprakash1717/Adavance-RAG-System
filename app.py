import sys
print(">>> [1/5] APP STARTING...", flush=True)

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print(">>> [2/5] SQLITE PATCH APPLIED SUCCESSFULLY", flush=True)
except ImportError:
    print(">>> [2/5] SQLITE PATCH SKIPPED", flush=True)
    pass

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
print(">>> [3/5] FLASK IMPORTED", flush=True)

import rag_engine
print(">>> [4/5] RAG ENGINE IMPORTED", flush=True)

import csv_engine
print(">>> [5/5] CSV ENGINE IMPORTED", flush=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 50MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error occurred."}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'document' not in request.files:
        return jsonify({"error": "No document provided"}), 400
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            chunks = rag_engine.process_pdf(filepath)
            return jsonify({
                "message": "File uploaded and indexed successfully", 
                "chunks_processed": chunks
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
        
    user_query = data['query']
    
    try:
        answer = rag_engine.query_pdf(user_query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    if 'document' not in request.files:
        return jsonify({"error": "No document provided"}), 400
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            stats = csv_engine.process_csv(filepath)
            return jsonify({
                "message": "CSV uploaded and analyzed successfully", 
                "stats": stats,
                "filename": filename
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file format. Please upload a .csv or .xlsx file."}), 400

@app.route('/api/chat-csv', methods=['POST'])
def chat_csv():
    data = request.json
    if not data or 'query' not in data or 'filename' not in data:
        return jsonify({"error": "Missing query or filename"}), 400
        
    user_query = data['query']
    filename = secure_filename(data['filename'])
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
         return jsonify({"error": "File not found on server. Please re-upload."}), 404
    
    try:
        answer = csv_engine.query_csv(user_query, filepath)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f">>> [SUCCESS] BINDING TO PORT {port} NOW!", flush=True)
    app.run(host='0.0.0.0', port=port, debug=False)
