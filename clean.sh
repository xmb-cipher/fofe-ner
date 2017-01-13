#!/cs/local/bin/bash

Color_Off="\033[0m"
Red="\033[0;31m"

function INFO() {
	printf ${Red}
	printf "[%s] %s\n" "`date`" "$@"
	printf ${Color_Off}
}


rm -rf *.pyc
rm -rf build
rm -rf checkpoint
rm -rf *.so
rm -f `find ./ -maxdepth 1 -type l`
