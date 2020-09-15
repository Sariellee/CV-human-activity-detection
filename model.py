import cv2 as cv
import numpy as np


class Model:
    def predict(self, frame):
        return

    def draw_bboxes(self, frame):
        bboxes, score = self.predict(frame)

        for bbox in bboxes:
            xLeftBottom, yLeftBottom, xRightTop, yRightTop = bbox
            cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                         (0, 255, 0), thickness=2)

        return frame, len(bboxes)


class YOLOv4(Model):
    def __init__(self, model_path='model_data/yolov4.weights',
                 config_path='model_data/yolov4.cfg', confidence_th=0.8, nms_threshold=0.4):
        self.net = None
        self.model_path = model_path
        self.config_path = config_path
        self.confidence_th = confidence_th
        self.nms_threshold = nms_threshold
        self.load_model()

    def load_model(self):
        self.net = cv.dnn.readNet(self.config_path, self.model_path)

    def _preprocess(self, frame):
        return cv.resize(frame, (320, 320))

    def predict(self, frame):
        frame_copy = frame.copy()
        frame_copy = self._preprocess(frame_copy)
        blob = cv.dnn.blobFromImage(frame_copy, 1 / 255.0, (320, 320), 0)

        # Run a model
        self.net.setInput(blob)
        out = self.net.forward(self.get_output_layers())

        return self._postprocess(out, frame.shape)

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        return [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def _postprocess(self, outs, original_shape):
        frame_height = original_shape[0]
        frame_width = original_shape[1]

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                if classId != 0:
                    continue
                confidence = scores[classId]
                if confidence > self.confidence_th:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_th,
            self.nms_threshold
        )
        coords = []
        confidence_res = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            coords.append((left, top, left+width, top+height))
            confidence_res.append(confidences[i])

        return coords, confidence_res


class MobileNet(Model):
    def __init__(self, model_path='model_data/mobilenet_iter_73000.caffemodel',
                 config_path='model_data/deploy.prototxt', confidence_th=0.2):
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

        return frame, len(bboxes)

# import matplotlib.pyplot as plt
# im = cv.imread('/Users/Pavel/programs/EORA/test-task-EORA/model_data/img.png')
# # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
# m = YOLOv4()
# detections = m.draw_bboxes(im)
# plt.imshow(im)
# plt.show()
# [(1459, 21, 1790, 995), (100, 80, 484, 1059), (1076, 48, 1332, 992), (640, 128, 954, 1065)]
# [(1469, 28, 1782, 1076), (103, 83, 467, 1068), (633, 107, 919, 1083), (1086, 52, 1321, 1072), (1614, 198, 1652, 324)]
# model_path = "/Users/Pavel/programs/EORA/test-task-EORA/model_data/mobilenet_iter_73000.caffemodel"
# config_path = "/Users/Pavel/programs/EORA/test-task-EORA/model_data/deploy.prototxt"
#
# m = Model(model_path, config_path)
# im = cv.imread('/Users/Pavel/programs/EORA/test-task-EORA/model_data/img.png')
# detections = m.draw_bboxes(im)
#
# cv.imshow("d", detections)
# cv.waitKey(0)
# https://www.ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/
