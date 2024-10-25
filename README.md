# CalcPal---Flask-Application-CNN

## Build the Docker Image
docker build -t dyscalculia-flask-app-cnn .

## Run the Docker Container
docker run --name dyscalculia-flask-cnn-container -p 5002:5002 dyscalculia-flask-app-cnn

## Sending a POST request to the CNN number model prediction endpoint
curl -X POST http://localhost:5002/predict-number -H "Content-Type: application/json" -d '{"image": "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"}'

## Sending a POST request to the CNN symbol model prediction endpoint
curl -X POST http://127.0.0.1:5002/predict-symbol -H "Content-Type: application/json" -d '{"image": "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"}'
