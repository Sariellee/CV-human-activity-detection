import time

import requests


class TelegramNotification:
    def __init__(self):
        self.url = "https://api.telegram.org/bot1342890350:AAHWqDyNXYpUe6oe_keMwvV64cIh9O9nS84/sendmessage?chat_id=-1001300235442&text="
        self.delay = 10
        self.last_time = time.time()

    def update(self, text):
        if time.time() - self.last_time > self.delay:
            requests.get(self.url + text)
            self.last_time = time.time()
