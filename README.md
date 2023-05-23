# Object Detection API with Faster R-CNN

This repository hosts an application that utilizes the Faster R-CNN model to perform object detection on the [PASCAL Visual Object Classes (VOC)] (http://host.robots.ox.ac.uk/pascal/VOC/) dataset. The model is then deployed as a Flask API, enabling a user-friendly interaction with the trained model.

## Directory Structure

The directory structure of this repository is organized as follows:

```.
├── src              
│   ├── train.py      # Training script
│   ├── test.py       # Testing script
│   └── eval.py       # Evaluation script
├── Dockerfile        # Dockerfile to containerize the application
├── main.py           # Flask API service
├── Makefile          # Makefile for automating tasks
└── requirements.txt  # Project dependencies
```

## Getting Started

To get the project up and running, follow these steps:

1. Clone the repository.
2. Install the necessary Python packages using `pip install -r requirements.txt`.
3. Download the dataset from [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/) and add to root directory.
4. Run the Makefile using `make all` to load data, train,test/eval the model, and build the Docker image.

## Running the Flask API

After creating the Docker image, you can run the Flask API. The API consists of two main endpoints:

1. **`GET /`**: A simple endpoint returning a welcome message.

2. **`POST /detect`**: This endpoint expects an image file in the `file` field of the request body. It processes the image through the object detection model and returns a list of detected objects, each represented as a JSON object with `category_id`, `bbox` (bounding box coordinates), and `score` (confidence score).

To interact with the API, can use any HTTP client like curl, Postman, or even your web browser (for the GET request).

Start the Flask API by running the Docker container:
```
docker run -p 8000:8000 <docker-image-name>:<tag>
``` 

Then, you can access the API at `http://localhost:8000`.

For more information on how to interact with the API, check out the [Flask Documentation](https://flask.palletsprojects.com/)


## Additional Resources

- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)

## License

This project is licensed under the MIT License. See the [LICENSE] file for details.