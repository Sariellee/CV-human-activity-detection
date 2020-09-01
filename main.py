import os
import threading

import cv2
import imutils
import pafy as pafy
from flask import Flask, Response
from flask import render_template

from src.detector import detect_human_bodies

youtube_url = os.environ.get("YOUTUBE_URL", "https://www.youtube.com/watch?v=VweY4kbkk5g")
video = pafy.new(youtube_url)
best = video.getbest()

stream_url = os.environ.get("STREAM_URL")
video_capture = cv2.VideoCapture(stream_url or best.url)

app = Flask(__name__)

outputFrame = None
lock = threading.Lock()


def detect_position():
    global video_capture, outputFrame, lock
    # watch ip camera stream
    while True:
        try:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            if frame is None:
                continue
            frame = imutils.resize(frame, width=400)

            detect_human_bodies(frame)

            # Display the resulting frame (optional)
            # cv2.imshow('Video', frame)

            with lock:
                outputFrame = frame.copy()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except KeyError:
            continue
        except KeyboardInterrupt:
            print("program stopped")
            exit(0)

    video_capture.release()
    cv2.destroyAllWindows()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_position)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=80, debug=True, threaded=True, use_reloader=False)
