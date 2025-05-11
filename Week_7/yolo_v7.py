from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('yolov8n.pt')  # load an official pre-trained model

    # Predict with the model and show the results
    results = model.predict(source='2.jpg', show=True, conf=0.5, save=True)  # predict on an image folder

if __name__ == "__main__":
    main()    
