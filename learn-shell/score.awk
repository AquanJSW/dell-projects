#!/bin/awk -f
# score.txt成绩处理

BEGIN{
    math = 0
    eng = 0
    computer = 0

    printf "%6s %5s %6s %8s %9s %7s\n", "Name", "No.", "Math", "English", "Computer", "Total"
    printf "----------------------------------------------------------------------\n"
}

{
    math += $3
    eng += $4
    computer += $5
    printf "%6s %5d %6d %8d %9d %7d\n", $1, $2, $3, $4, $5, $3+$4+$5
}

END{
    printf "----------------------------------------------------------------------\n"
    printf "%9s %9d %8d %9d\n","TOTAL:", math, eng, computer
    printf "%9s %9.2f %8.2f %9.2f\n", "AVERAGE:", math/NR, eng/NR, computer/NR
}
