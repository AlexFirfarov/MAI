all: client server

client:
	gcc -std=c99 -pthread -D_POSIX_SOURCE=700 -D_XOPEN_SOURCE=700 client.c -o client.exe -L/usr/local/lib -lzmq

server:
	gcc -std=c99 -pthread -D_POSIX_SOURCE=700 -D_XOPEN_SOURCE=700 server.c -o server.exe -L/usr/local/lib -lzmq
