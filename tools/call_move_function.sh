#!/bin/bash

args=(name source dest)
opts=("[files]")
usage="usage: $(basename ${0}) ${args[@]} ${opts[@]}"
[ ${#} -lt ${#args[@]} ] && { echo "${usage}" >&2; exit 1; }
name="${1}"
source="${2}"
dest="${3}"
shift 3
files=(${@})

[ ${#files[@]} -eq 0 ] && files=($(\find src tests -name '*.py' -o -name '*.pxd' -o -name '*.pyx'))

\sed -i "s/\(# :call: [>v] \)${source}::${name}$/\1${dest}::${name}/" ${files[@]}
