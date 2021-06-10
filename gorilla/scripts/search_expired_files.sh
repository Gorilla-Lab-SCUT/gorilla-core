# Copyright (c) Gorilla-Lab. All rights reserved.
# Author: zhang.haojian
# usage: sh search_expired_files.sh user "/data" 60 "*.pth *.pt"
output_file="./expired_files_$(date '+%Y-%m-%d').txt"
echo "Search result will be saved in $output_file."

user=$1
base_dir=$2
days=$3  # remove file modified x days earlier 
file_fmts=$4

# clear old file content
echo -n "" > $output_file
echo Search time: `date`, days=$days, file_fmts=${file_fmts[*]}, user=$user > $output_file

for fmt in ${file_fmts[*]}
do
    echo "searching $fmt..."
    for filepath in $(locate $base_dir/$fmt)
    do
        # ${filepath%/*} get directory, and ${filepath##*/} get file name
        echo find ${filepath%/*} -name ${filepath##*/} -user $user -mtime +$days
        for res in $(find ${filepath%/*} -name ${filepath##*/} -user $user -mtime +$days)
        do
            echo $res >> $output_file
        done
    done
done
