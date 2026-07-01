CC = gcc
CFLAGS = -O2 -Wall -lm

host_test_bin: main.c model_reg.c model_clf.c
	$(CC) $(CFLAGS) -o host_test_bin main.c model_reg.c model_clf.c

clean:
	rm -f host_test_bin
