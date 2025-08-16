# Compiler and flags
CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

# Rule to build the executable 'symnmf'
all: symnmf

# The 'symnmf' executable depends on the C source file.
# -lm for the math library
symnmf: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -o $@ $< -lm

# A 'clean' rule to remove generated files
clean:
	rm -f symnmf *.o

.PHONY: all clean