# for d in */; 
# do 
	# echo "put ${d}";
	# cd ${d};
	# for d2 in */;
		# cd ${d2};
		# for file in *; do 
			# if [ -f "$file" ]; then 
				# if [[ $file == *.vl ]]; then 
					# echo "$file"
					# #rm $file
				# fi
			# fi 
		# done
		# cd ..
	# cd ..
	
# done


# for file in *; do 
    # if [ -f "$file" ]; then 
        # echo "$file" 
    # fi 
# done 

find . -print0 | while IFS= read -r -d '' file
do 
    if [ -f "$file" ]; then 
		if [[ $file == *.vl ]]; then 
			#echo "$file"
			rm -v $file
		fi
	fi 
done

