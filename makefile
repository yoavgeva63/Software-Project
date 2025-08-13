CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
OBJS = symnmf.o

all: symnmf

symnmf: $(OBJS)
	$(CC) $(CFLAGS) -o symnmf $(OBJS) -lm

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

clean:
	rm -f symnmf $(OBJS)
