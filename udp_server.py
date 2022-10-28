import socket

SERVER_ADDRESS_PORT = ("127.0.0.1", 20001)
BUFFER_SIZE = 1024
UDP_SERVER_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
UDP_SERVER_SOCKET.bind(SERVER_ADDRESS_PORT)


def receive_message():
    message = UDP_SERVER_SOCKET.recvfrom(BUFFER_SIZE)
    return message[0], message[1]


msgFromServer = "Hello UDP Client"
bytesToSend = str.encode(msgFromServer)
print("UDP server up and listening")


def send_message(message, address):
    bytes_to_send = str.encode(message)
    UDP_SERVER_SOCKET.sendto(bytes_to_send, address)


while True:
    message, address = receive_message()
    print(message, address)
    UDP_SERVER_SOCKET.sendto(bytesToSend, address)
