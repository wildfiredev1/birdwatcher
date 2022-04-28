import yaml
import sys
from io import BytesIO
from torch.autograd import Variable
from typing import List, Dict, Union, ByteString, Any
import os
import flask
from flask import Flask, render_template
from PIL import Image
import requests
import torch
from torchvision import transforms
import json

with open("config.yaml", 'r') as stream:
    APP_CONFIG = yaml.full_load(stream)

app = Flask(__name__)

classes = ['American Coot',
 'Barn Owl',
 'Blue Jay',
 'Cattle Egret',
 'Eastern Screech Owl',
 'Great Blue Heron',
 'Neotropic Cormorant',
 'Northern Mockingbird',
 'Red Headed Woodpecker',
 'Redtailed Hawk']


def load_model(path="."):
    model = torch.load(path,map_location=torch.device('cpu'))
    return model

def predict(img):

    loader = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    image = loader(img).float().unsqueeze(0)
    image = Variable(image, requires_grad=True)

    model_ft = load_model('./birdwatcher.pth')
    model_ft.eval()
    output = model_ft(image)
    return classes[output.argmax().item()]

@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':
        url = flask.request.args.get("url")
        img = Image.open(requests.get(url, stream=True).raw)

    else:
        bytes = flask.request.files['file'].read()
        img = Image.open(BytesIO(bytes))
    res = predict(img)
    return res


@app.route('/api/classes', methods=['GET'])
def classlist():
    classes = sorted(classes)
    return flask.jsonify(classes)


@app.route('/config')
def config():
    return flask.jsonify(APP_CONFIG)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"

    response.cache_control.max_age = 0
    return response


@app.route('/<path:path>')
def static_file(path):
    if ".js" in path or ".css" in path:
        return app.send_static_file(path)
    else:
        return render_template('index.html')


@app.route('/')
def root():
    return render_template('index.html')


def before_request():
    app.jinja_env.cache = {}


if __name__ == '__main__':
    app.run(debug=False)
