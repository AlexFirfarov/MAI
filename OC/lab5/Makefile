CC = gcc
FLAGS = -std=c99 -pthread -w -lm

all: run clear

run: libvector.so mainStat.o mainDin.o
	$(CC) $(FLAGS) -o mainstat.exe mainStat.o -L. -lvector -Wl,-rpath,.
	$(CC) $(FLAGS) -o maindyn.exe mainDin.o -ldl

mainStat.o: mainStat.c
	$(CC) -c $(FLAGS) mainStat.c

mainDin.o: mainDin.c
	$(CC) -c $(FLAGS) mainDin.c

vector.o: vector.c
	$(CC) -c -fPIC $(FLAGS) vector.c

libvector.so: vector.o
	$(CC) $(FLAGS) -shared -o libvector.so vector.o

clear:
	rm -f *.o

