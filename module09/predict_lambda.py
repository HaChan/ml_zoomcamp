import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

interpreter = tflite.Interpreter(model_path="model_2024_hairstyle_v2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def rescale_img(img):
    img_arr = np.array(img)
    return img_arr / 255.0

def predict(url):
    img = download_image(url)
    prep_img = prepare_image(img, (200,200))
    rescaled_img_arr = rescale_img(prep_img)
    X = np.expand_dims(rescaled_img_arr, axis=0).astype('float32')
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return preds[0].tolist()

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
