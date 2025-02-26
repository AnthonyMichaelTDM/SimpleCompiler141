# compile every file in this directory, the binary will be named the same as the source file without the extension (e.g. test1.c -> test1.out)
# the binary will be placed in the same directory as the source file

# get all the source files in the current directory
SOURCES := $(wildcard *.c)

# get all the binary files in the current directory
BINS := $(SOURCES:.c=)

# compile every source file into a binary
all: $(BINS)

# remove all the binaries
clean:
	rm $(BINS)

# compile a source file into a binary
%: %.c
	gcc -o $@ $<