# import socket
# import time
#
# SERVER_ADDRESS_PORT = ("192.0.0.240", 20001)
# BUFFER_SIZE = 1024
# UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
#
#
# def send_message(message):
#     bytes_to_send = str.encode(message)
#     UDP_CLIENT_SOCKET.sendto(bytes_to_send, SERVER_ADDRESS_PORT)
#
#
# def receive_message():
#     message = UDP_CLIENT_SOCKET.recvfrom(BUFFER_SIZE)
#     return message[0], message[1]
#
#
# i = 0
# while True:
#     i += 1
#     print("send message")
#     send_message(str(i))
#     print("sended message")
#     received_message, address = receive_message()
#     print(received_message, address)
#     time.sleep(1)

import socket
#The type of communications between the two endpoints, typically SOCK_STREAM for connection-oriented protocols and SOCK_DGRAM for connectionless protocols.
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
host = "192.0.0.45"
port = 8000
addr = (host, port)
#s.connect(addr) #---This method actively initiates TCP server connection.


def send(user_input):
    s.sendto(user_input.encode(), addr)
    data = s.recvfrom(1024).decode()
    print('Server tell that: ', data)


while 1:
    # if input('Exit? (y/n)') == 'y':
    #     break
    # i = input('>> ')
    i = "123"
    send(i)

s.close()