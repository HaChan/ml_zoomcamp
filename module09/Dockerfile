FROM agrigorev/model-2024-hairstyle:v3

RUN pip install numpy==1.23.2
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install pillow

COPY predict_lambda.py .

CMD [ "predict_lambda.lambda_handler" ]
