# importing necessary libraries
import os
from ultralytics import YOLO
from flask import Flask, flash, url_for, request, render_template,  redirect, session, g

# making basic settings
UPLOAD_FOLDER = ''     # folder to upload images, here - images will be uploaded to project's folder

app = Flask(__name__)  # app - new Flask application

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # app wil upload images in our folder - project's folder


# describing what our program should do when user uploads the image
@app.route('/upload', methods=['POST']) # address of this method will be like site_address/upload
def upload():                                # method's name should be the same as link name, here - "upload"
    if request.method == 'POST':             # working with the form
        file = request.files['file']         # getting image file
        try:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename)) # saving file to UPLOAD_FOLDER folder
            model = YOLO('best.pt')             # setting the model
            results2 = model([file.filename])   # giving an image to our model to recognize
            for result in results2:             # working with the recognized result
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                #result.show()  # display to screen
                result.save(filename='static/images/result.jpg')  # save to disk

        except FileNotFoundError as e:  # if there are some problems with file
            return 'File not found'
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return redirect(url_for('results')) # redirection to results page
    return redirect(url_for('index'))  # redirection to main page


@app.route('/')                        # main page
def index():
    return render_template('index.html')

@app.route('/hiw')                    # how it works page
def hiw():
    return render_template('hiw.html')

@app.route('/test')                    # how it works page
def test():
    return 'Test from Flask'

@app.route('/results')                # results page
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)








