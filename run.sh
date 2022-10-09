#!/bin/bash

for ((j=$6;j<=$7;j+=1)); do 
    for ((i=$3;i<=$4;i+=$5)); do
        name="k=$i times=$j"
        timer_start=`date "+%Y-%m-%d %H:%M:%S"`
        echo "$name begin: $timer_start"
        python get_coreset.py $i $j $1 $2 $8
        timer_end=`date "+%Y-%m-%d %H:%M:%S"`
        echo "$name end: $timer_end"
        duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
        echo "$name use: $duration"
    done
    wait
done

python merge_corset.py $2 $3 $4 $5 $6 $7 $8 $1

echo "END"
