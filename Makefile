build/ane-compile: tool/ane_compile.m
	mkdir -p build
	xcrun -sdk macosx clang -O2 -Wall -Wextra -Werror -fobjc-arc -framework Foundation $< -o $@

clean:
	rm -rf build
