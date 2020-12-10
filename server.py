from flask import Flask,render_template, request, flash, request, redirect, url_for, send_from_directory
import os

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# @app.route("/homepage")
# def homepage():
#     return render_template('homepage.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    global image_ids
    # If users post an image onto the server
    print(request)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # if user submit a file and that file is valid
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # construct the input path
            input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], INPUT_FILE_NAME)
            #  save that file to the folder
            file.save(input_file_path)
        
        # Transform that image into vectors user 3 pre-initialized moduel
        image_vector = inference("static/images/input.jpg",tflite_interpreter_obj,scaler_obj,pca_obj)
        
        # Get the NUM_NEAREST_NEIGHBORS neighbors of that vector to a pretrained indexing module
        # This will return a top nearest image_ids that closest to the vector
        image_ids = approx_search_obj.get_nns_by_vector(image_vector,NUM_NEAREST_NEIGHBORS)

        # Perform ranking by comparing the query vector to nearest neighbor
        image_ids = ranking(image_vector,images_features_scaled_pca,image_ids)
        
        # This is for rendering the table in result page
        page, per_page, offset = get_page_args(page_parameter='page',
                                            per_page_parameter='per_page')
        total = len(image_ids)
        pagination_image_ids = get_image_ids(offset=offset, per_page=per_page)
        pagination = Pagination(page=page, per_page=per_page, total=total,
                                css_framework='bootstrap4')
        return redirect(url_for('result'),)
        
    else:
        return render_template('homepage.html')

@app.route("/result")
def shop():
    return render_template('all_images.html', links=abc)

@app.route('/sample/<idx>', methods=['GET', 'POST'])
def sample(idx):
    if idx == '0':
        path = './static/images/'
        imgs = []
        for img in os.listdir(path):
            imgs.append(img)
        return render_template('searchBySample.html', imgs=imgs)
    else:
        

if __name__ == '__main__':
    app.run(port='80', debug=True)

    