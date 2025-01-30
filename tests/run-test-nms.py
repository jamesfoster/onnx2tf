import numpy as np
import tensorflow as tf
import cv2

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="./yolov4-32.tflite")
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


def post_process(output_data, original_image, conf_threshold=0.5, iou_threshold=0.4):
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

    image_shape = original_image.shape

    # Filter out low-confidence boxes
    detected_boxes = []
    detected_scores = []
    detected_classes = []
    for i in range(len(scores)):
        if scores[i] > conf_threshold:
            box = boxes[i]
            
            # 🔥 Correct Scaling 🔥
            x1 = int(boxes[i][0] * image_shape[1])
            y1 = int(boxes[i][1] * image_shape[0])
            x2 = int(boxes[i][2] * image_shape[1])
            y2 = int(boxes[i][3] * image_shape[0])

            # Ensure coordinates stay within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_shape[1] - 1, x2), min(image_shape[0] - 1, y2)

            score = scores[i]
            class_id = class_ids[i]
            detected_boxes.append([x1, y1, x2, y2])
            detected_scores.append(float(score))
            detected_classes.append(class_id)
            print(
                f"Found {class_id}, {round(score * 100000) / 1000}% {box}")

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(detected_boxes, detected_scores, conf_threshold, iou_threshold)

    final_boxes, final_scores, final_classes = [], [], []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(detected_boxes[i])
            final_scores.append(detected_scores[i])
            final_classes.append(detected_classes[i])
            print(f"supp.{i}: {detected_classes[i]}, {round(detected_scores[i] * 100000) / 1000}% {detected_boxes[i]}")

    return final_boxes, final_scores, final_classes

# Display results (bounding boxes on the image)


def display_results(original_image, boxes, scores, class_names):
    # Get original image dimensions
    h, w, _ = original_image.shape

    image = original_image.copy()
    for i in range(len(boxes)):
    
        x1, y1, x2, y2 = boxes[i]
        score = scores[i]
        class_name = class_names[i]

        # Draw the bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"Conf: {score:.2f} {class_name}",
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
print(class_names)

output_data, original_image = run_inference(image_path)

# Process output and get final detections
boxes, scores, class_ids = post_process(output_data, original_image, conf_threshold=0.60)

classes=[class_names[class_id] for class_id in class_ids]

display_results(original_image, boxes, scores, classes)
