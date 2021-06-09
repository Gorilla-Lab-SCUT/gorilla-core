# Author: zhang.haojian
# usage: sh search_expired_files.sh username 60 "*.pth *.pt"
output_file="./expired_files_$(date '+%Y-%m-%d').txt"
echo "Search result will be saved in $output_file."

username=$1
days=$2  # remove file modified x days earlier 
file_fmts=$3

# clear old file content
echo -n "" > $output_file
echo Search time: `date`, days=$days, file_fmts=${file_fmts[*]}, username=$username > $output_file

for fmt in ${file_fmts[*]}
do
    echo "searching $fmt..."
    for filepath in $(locate $fmt)
    do
        # ${filepath%/*} get directory, and ${filepath##*/} get file name
        echo find ${filepath%/*} -name ${filepath##*/} -user $username -mtime +$days
        for res in $(find ${filepath%/*} -name ${filepath##*/} -user $username -mtime +$days)
        do
            echo $res >> $output_file
        done
    done
done
