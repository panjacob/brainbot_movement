import socket
import time

SERVER_ADDRESS_PORT = ("127.0.0.1", 20001)
BUFFER_SIZE = 1024
UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)


def send_message(message):
    bytes_to_send = str.encode(message)
    UDP_CLIENT_SOCKET.sendto(bytes_to_send, SERVER_ADDRESS_PORT)


def receive_message():
    message = UDP_CLIENT_SOCKET.recvfrom(BUFFER_SIZE)
    return message[0], message[1]


i = 0
while True:
    i += 1
    send_message(str(i))
    received_message, address = receive_message()
    print(received_message, address)
    time.sleep(1)
