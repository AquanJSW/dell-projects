#!/bin/bash

LBLs=`ls ./*.LBL`
for LBL in $LBLs
do
    m_file_path="`echo ${LBL} | awk -F '[./]' '{print $3}'`.m"
    > $m_file_path
    cat $LBL | grep 'MODEL_COMPONENT_[1-9]' | while read line
    do
        echo $line | awk -F '[ (),]' '{\
            if ($1=="MODEL_COMPONENT_1")\
                printf "C = [%s, %s, %s];\n", $4, $5, $6;\
            else if ($1=="MODEL_COMPONENT_2")\
                printf "A = [%s, %s, %s];\n", $4, $5, $6;\
            else if ($1=="MODEL_COMPONENT_3")\
                printf "H = [%s, %s, %s];\n", $4, $5, $6;\
            else if ($1=="MODEL_COMPONENT_4")\
                printf "V = [%s, %s, %s];\n", $4, $5, $6;\
            else if ($1=="MODEL_COMPONENT_5")\
                printf "O = [%s, %s, %s];\n", $4, $5, $6;\
            else if ($1=="MODEL_COMPONENT_6")\
                printf "R = [%s, %s, %s];\n", $4, $5, $6;\
            else if ($1=="MODEL_COMPONENT_7")\
                printf "E = [%s, %s, %s];\n", $4, $5, $6;\
            else if ($1=="MODEL_COMPONENT_8")\
                printf "T = %d;\n", $3;\
            else if ($1=="MODEL_COMPONENT_9")\
                printf "P = %d;\n", $3;\
            }' >> $m_file_path
    done
done

