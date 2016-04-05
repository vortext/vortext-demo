#!/bin/sh

cd /var/lib/deploy/src/
/var/lib/deploy/bin/lein with-profile production trampoline $1
