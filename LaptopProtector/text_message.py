import sys
from CONFIG import CONFIG
from twilio.rest import Client
import time


def send_message(text):
    client = Client(CONFIG["SID"], CONFIG["TOKEN"])

    to = '+8613818988768'

    msg = client.messages.create(to=to, from_=CONFIG["MY_NUMBER"], body=text)

    time.sleep(5)

    print("STATUS: {}".format(msg.status))

