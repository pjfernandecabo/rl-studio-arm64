import socket


#AF_INET = ipv4
#SOCK_STREAM = TCP


#HOST=''
#HOST=socket.gethostbyname(socket.gethostname())
#HOST=socket.gethostbyname_ex(socket.gethostname())
HOST=socket.gethostname()
#HOST=socket.getsockname()
PORT=74
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("socket is created")
print(f"socket is bound to address {socket.gethostname()} and port {PORT} ")
print(f"socket is bound to address {socket.gethostbyname_ex(socket.gethostname())} and port {PORT} ")
print(f"socket is bound to address {socket.getfqdn()} ")
print(f"socket is bound to address {socket.getaddrinfo(HOST, PORT)} ")
print(f"socket is bound to address {socket.if_nameindex()} ")


#s.bind((socket.gethostname(), PORT))
s.bind((HOST, PORT))
#s.getaddrinfo()


s.listen(5)
print("listening for incoming connection ...")

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = s.accept()
    print(f"Connection from clientsocket: {clientsocket} and address:  {address} has been established.")
    clientsocket.send(bytes("Hey there!!!","utf-8"))





'''
# Echo server program
import socket
import sys

HOST = None               # Symbolic name meaning all available interfaces
PORT = 74              # Arbitrary non-privileged port
s = None
for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC,
                              socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
    print(res)                          
    af, socktype, proto, canonname, sa = res
    try:
        s = socket.socket(af, socktype, proto)
    except OSError as msg:
        s = None
        continue
    try:
        s.bind(sa)
        s.listen(1)
    except OSError as msg:
        s.close()
        s = None
        continue
    break
if s is None:
    print('could not open socket')
    sys.exit(1)
conn, addr = s.accept()
with conn:
    print('Connected by', addr)
    while True:
        data = conn.recv(1024)
        if not data: break
        conn.send(data)

'''
