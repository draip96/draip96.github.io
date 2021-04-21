import os

from flask import Flask, render_template, request, redirect, send_from_directory
from inference import get_prediction
from commons import format_class_name


UPLOAD_FOLDER = 'static/uploads/'



app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print(request.form)
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        
        
        filename = file.filename
        img_bytes = file.read()
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        filename = 'http://127.0.0.1:5000/uploads/' + filename

        if not file:
            return redirect(request.url)

        model = request.form.get('model')
        
        
        
        class_id, class_name = get_prediction(model, image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        return render_template('result.html', class_id=class_id,
                               class_name=class_name,
                               model=model,
                               filename=filename)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run()
