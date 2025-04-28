from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
from docx import Document
import PyPDF2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def extract_text(file_path, file_extension):
    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_extension == '.docx':
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif file_extension == '.pdf':
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
    else:
        return ''

@app.route('/')
def welcome():
    # Показываем шаблон index.html — главную страницу
    return render_template('welcome.html')

@app.route('/about')
def about():
    # Показываем шаблон about.html
    return render_template('about.html')

@app.route('/audit')
def audit():
    # Показываем шаблон about.html
    return render_template('audit.html')

@app.route('/audit_hand')
def audit_hand():
    # Показываем шаблон about.html
    return render_template('upload_file.html')

@app.route('/upload_file')
def upload_file():
    # Показываем шаблон about.html
    return render_template('upload_file.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Проверяем, есть ли файл в запросе
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Защищаем имя файла и получаем расширение
    filename = secure_filename(file.filename)
    file_extension = os.path.splitext(filename)[1].lower()

    # Проверяем, что расширение допустимо
    if file_extension not in ['.txt', '.docx', '.pdf']:
        return jsonify({'error': 'Only .txt, .docx, and .pdf files are allowed'}), 400

    # Сохраняем файл
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Извлекаем текст
    content = extract_text(file_path, file_extension)

    # Формируем краткое содержимое (первые 100 символов)
    summary = content[:100] + '...' if len(content) > 100 else content

    # Возвращаем ответ
    return jsonify({'filename': filename, 'summary': summary})


@app.route('/view/<filename>')
def view_file(filename):
    # Формируем путь к файлу
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        return 'Файл не найден', 404

    # Извлекаем текст из .docx файла
    try:
        doc = Document(file_path)
        content = '\n'.join([para.text for para in doc.paragraphs])
        return content
    except Exception as e:
        return f'Ошибка при обработке файла: {str(e)}', 500

if __name__ == "__main__":
    app.run(debug=True)
