import cv2 as cv
import sys


class Model:
    def __init__(self, model_path, config_path, confidence_th=0.2):
        self.net = None
        self.model_path = model_path
        self.config_path = config_path
        self.confidence_th = confidence_th
        self.load_model()

    def load_model(self):
        self.net = cv.dnn.readNetFromCaffe(self.config_path, self.model_path)

    def _preprocess(self, frame):
        return cv.resize(frame, (300, 300))

    def predict(self, frame):
        frame_copy = frame.copy()
        frame_copy = self._preprocess(frame_copy)
        blob = cv.dnn.blobFromImage(frame_copy, 0.007843, (300, 300), 127.5)

        # Run a model
        self.net.setInput(blob)
        out = self.net.forward()

        return self._postprocess(out, frame.shape)

    def _postprocess(self, detections, original_shape):

        bboxes = []
        score = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of prediction
            if confidence > self.confidence_th:  # Filter prediction
                class_id = int(detections[0, 0, i, 1])  # Class label
                # only look for people
                if class_id != 15:
                    continue
                score.append(confidence)

                # Object location
                xLeftBottom = detections[0, 0, i, 3]
                yLeftBottom = detections[0, 0, i, 4]
                xRightTop = detections[0, 0, i, 5]
                yRightTop = detections[0, 0, i, 6]

                height = original_shape[0]
                width = original_shape[1]
                # Scale object detection to frame
                xLeftBottom = int(width * xLeftBottom)
                yLeftBottom = int(height * yLeftBottom)
                xRightTop = int(width * xRightTop)
                yRightTop = int(height * yRightTop)
                # Draw location of object
                bboxes.append((xLeftBottom, yLeftBottom, xRightTop, yRightTop))

        return bboxes, score

    def draw_bboxes(self, frame):
        bboxes, score = self.predict(frame)

        for bbox in bboxes:
            xLeftBottom, yLeftBottom, xRightTop, yRightTop = bbox
            cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                         (0, 255, 0))

        return frame

# model_path = "/Users/Pavel/programs/EORA/test-task-EORA/model_data/mobilenet_iter_73000.caffemodel"
# config_path = "/Users/Pavel/programs/EORA/test-task-EORA/model_data/deploy.prototxt"
#
# m = Model(model_path, config_path)
# im = cv.imread('/Users/Pavel/programs/EORA/test-task-EORA/model_data/img.png')
# detections = m.draw_bboxes(im)
#
# cv.imshow("d", detections)
# cv.waitKey(0)
#https://www.ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/