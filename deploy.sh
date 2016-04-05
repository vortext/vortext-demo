#!/bin/bash

echo "copying to temporary directory"
TMP=`mktemp -d`
cp -a . $TMP

echo "adding to tar file"
tar --exclude "*.log" --exclude ".git/*" --exclude ".git" --exclude "target" --exclude "node_modules" -z -c -v -f deploy.tar.gz $TMP

echo "removing temporary directory"
rm -rf $TMP

#echo "building image"
#docker build .
