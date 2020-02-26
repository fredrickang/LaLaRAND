CC=g++
CFLAGS=-g -Wall
OBJS=lalarand.o lalarand_fn.o
TARGET=lalarand

$(TARGET) : $(OBJS)
	$(CC) -o $@ $(OBJS)

lalarand.o: lalarand.h lalarand_fn.h lalarand.c
lalarand_fn.o: lalarand_fn.h lalarand.h lalarand_fn.c

clean:
	rm -f *.o
	rm -f $(TARGET)
