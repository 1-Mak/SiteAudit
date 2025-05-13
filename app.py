from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
import os
from docx import Document
import PyPDF2
import mammoth


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'KonstantaSosat'

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
    return render_template('1.welcome.html')

@app.route('/aiaudit')
def aiaudit():
    # Показываем шаблон about.html
    return render_template('2.ai-audit.html')

@app.route('/manaudit')
def manaudit():
    # Показываем шаблон about.html
    return render_template('5.manual-audit.html')

@app.route('/manaudit_check')
def manaudit_check():
    filename = session.get('uploaded_file')
    if not filename:
        return 'Файл не был загружен.', 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        with open(file_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html_content = result.value  # Преобразованный HTML
        return render_template('6.manual-check.html', content=html_content)
    except Exception as e:
        return f'Ошибка при обработке файла: {str(e)}', 500


@app.route('/analyze', methods=['POST'])
def analyze():
    # Проверяем, есть ли файл в запросе
    file = request.files['file']

    # Защищаем имя файла и получаем расширение
    filename = secure_filename(file.filename)
    file_extension = os.path.splitext(filename)[1].lower()

    

    # Сохраняем файл
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    session['uploaded_file'] = filename
    # # Извлекаем текст
    # content = extract_text(file_path, file_extension)

    # # Формируем краткое содержимое (первые 100 символов)
    # summary = content[:100] + '...' if len(content) > 100 else content

    # # Возвращаем ответ
    # return jsonify({'filename': filename, 'summary': summary})
    return '', 204

@app.route('/save_audit', methods=['POST'])
def save_audit():
    data = request.get_json()
    criteria = data.get('criteria', [])
    general_comment = data.get('general_comment', '')

    try:
        # Пример: сохраняем как JSON файл
        import json
        with open('audit_result.json', 'w', encoding='utf-8') as f:
            json.dump({
                'criteria': criteria,
                'general_comment': general_comment,
                'file_name' : session.get('uploaded_file')
            }, f, ensure_ascii=False, indent=2)

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
