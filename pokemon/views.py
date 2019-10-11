from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from django.shortcuts import render
from keras.preprocessing import image

from . import forms


def predict(test_image):
    path = Path('static/Images/')
    files = path.glob('*.jpg')

    all_files = [str(file) for file in files]

    df = pd.read_csv('static/train.csv')
    data = df.values

    X = data[:, 0]
    Y = data[:, 1]
    image_data = []
    labels_dict = {'Pikachu': 0, 'Bulbasaur': 1, 'Charmander': 2}

    labels = []

    """
        If we don't scale the image, it will take longer time to process.
        11:00 PM 10/11/19 by Arjun Adhikari
    """
    for d in all_files:
        img = image.load_img(d, target_size=(32, 32))
        img_array = image.img_to_array(img)
        image_data.append(img_array)

    for i in range(X.shape[0]):
        label = Y[i]
        labels.append(labels_dict[label])

    labels = np.array(labels)

    Y = np.array(labels)
    X = np.array(image_data)

    """
        Reshaping 4D numpy array into 2D array.
        11:13 PM 10/11/19 by Arjun Adhikari
    """
    X = X.reshape((X.shape[0], -1))

    """
        Training the data at runtime.
        11:04 PM 10/11/19 by Arjun Adhikari
    """

    def distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def KNN(x, y, query_point, k=7):
        m = x.shape[0]
        vals = []

        for i in range(m):
            d = distance(x[i], query_point)
            vals.append((d, y[i]))

        vals = sorted(vals)
        vals = vals[:k]
        vals = np.array(vals)

        new_val = np.unique(vals[:, 1], return_counts=True)
        index = new_val[1].argmax()
        pred = new_val[0][index]

        return pred

    """
        Prediction.
        11:09 PM 10/11/19 by Arjun Adhikari
    """

    val = int(KNN(X, Y, test_image))

    for key, value in labels_dict.items():
        if value == val:
            return key


def prepare(img_):
    """
        Converting the high quality picture into 32*32 sized image.
        12:13 AM 10/12/19 by Arjun Adhikari
    """
    img = image.img_to_array(img_.resize((32, 32), Image.ANTIALIAS))
    # Converting to 1D Array.
    img = img.ravel()
    return predict(img)


def classify(request):
    if request.method == 'POST':

        url = request.POST['url']
        img = request.FILES['img']

        if url:
            response = requests.get(url)
            img_data = Image.open(BytesIO(response.content))
            """
                Converting transparency to solid white.
                1:15 AM 10/12/19 by Arjun Adhikari
            """
            if img_data.mode == 'RGBA':
                # Create a blank background image
                bg = Image.new('RGB', img_data.size, (255, 255, 255))
                # Paste image to background image
                bg.paste(img_data, (0, 0), img_data)
                data = prepare(bg)
            else:
                data = prepare(img_data)

            return render(
                request,
                'index.html',
                {
                    'urlresult': data,
                    'urlimage': url
                }
            )

        if img:
            img_data = image.load_img(img, target_size=(32,32))

            if img_data.mode == 'RGBA':
                # Create a blank background image
                bg = Image.new('RGB', img_data.size, (255, 255, 255))
                # Paste image to background image
                bg.paste(img_data, (0, 0), img_data)
                data = prepare(bg)
            else:
                data = prepare(img_data)

            if img_data.mode == 'RGBA':
                # Create a blank background image
                bg = Image.new('RGB', img_data.size, (255, 255, 255))
                # Paste image to background image
                bg.paste(img_data, (0, 0), img_data)
                data = prepare(bg)
            else:
                data = prepare(img_data)

            return render(
                request,
                'index.html',
                {
                    'imageresult': data,
                }
            )


    else:
        form_page = forms.ClassifyForm()
        context = form_page

        return render(
            request,
            'index.html',
            {
                'context': context
            }
        )
