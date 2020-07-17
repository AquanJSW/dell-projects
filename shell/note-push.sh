#!/bin/bash

if [[ -z "$*" ]];then
    cat >&1 << EOF
*--------------------------------*
|   Please type your message ~   |
*--------------------------------*
EOF
exit 1
fi

cd ~/Notes
git add *.md
git commit -m "$*"
git push origin master

exit 0
