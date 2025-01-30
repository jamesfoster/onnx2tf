import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="./yolov4-16.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess input image


def preprocess_image(image_path, input_shape=(416, 416)):
    original_image = cv2.imread(image_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image, input_shape)
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(
        image_normalized,
        axis=0
    ).astype(np.float32)
    # Return original image as well for later resizing
    return original_image, image_expanded

# Inference on image


def run_inference(image_path):
    original_image, input_data = preprocess_image(image_path, (320, 576))

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor(s)
    output_data = [interpreter.get_tensor(
        output_detail['index']) for output_detail in output_details]

    return output_data, original_image

# Check the output data


def check_output(output_data):
    print(f"Number of output tensors: {len(output_data)}")
    for i, data in enumerate(output_data):
        print(f"Output data {i} shape: {data.shape}")
        print(f"Output data {i}: {data}")

# Post-process the output to extract bounding boxes (Optional)


def post_process(output_data, conf_threshold=0.5):
    # Inspect output data shape
    # check_output(output_data)

    # Extract bounding boxes (assuming the first output tensor corresponds to bounding boxes)
    # Flatten to (2535, 4) for easier processing
    boxes = output_data[0].reshape(-1, 4)

    # Extract class probabilities (second output tensor)
    num_classes = 8
    class_probs = output_data[1].reshape(-1, num_classes)

    # Get the class with the highest probability for each box
    # Get the class with the max probability
    class_ids = np.argmax(class_probs, axis=1)
    scores = np.max(class_probs, axis=1)  # Get the max probability as score

    # Filter out low-confidence boxes
    valid_boxes = {}
    for i in range(len(scores)):
        if scores[i] > conf_threshold:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            if class_id not in valid_boxes or score > valid_boxes[class_id]["score"]:
                # Include class ID for display
                valid_boxes[class_id] = {"box": box, "score": score}
                print(
                    f"Found {class_id}, {round(score * 100000) / 1000}% {box}")

    return valid_boxes

# Display results (bounding boxes on the image)


def display_results(original_image, valid_boxes, class_names):
    # Get original image dimensions
    h, w, _ = original_image.shape

    image = original_image.copy()
    for class_id, dict in valid_boxes.items():
        x1, y1, x2, y2 = dict["box"]
        score = dict["score"]

        # Rescale the bounding box coordinates back to the original image size
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        # Draw the bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"Conf: {score:.2f} {class_names[class_id]}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    cv2.imwrite("./data/output.png", image)

# Load class names (80 classes in YOLOv4-tiny)


def load_class_names():
    class_names = []
    with open("./data/classes.names", "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names


# Example usage
image_path = "./data/test.png"
class_names = load_class_names()  # Load the class names

output_data, original_image = run_inference(image_path)
# Lowered confidence threshold for testing
valid_boxes = post_process(output_data, conf_threshold=0.60)
display_results(original_image, valid_boxes, class_names)
