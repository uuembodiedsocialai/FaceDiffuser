# for d in */; 
# do 
	# echo "put ${d}";
	# cd ${d};
	# echo "starting extract!";
	# for file in *; do 
		# if [ -f "$file" ]; then 
			# if [[ $file == *.tar ]]; then 
				# echo "$file"
				# tar -xf $file --one-top-level
			# fi
		# fi 
		# done
	# cd ..
	
# done


for d in */; 
do 
	echo "put ${d}";
	cd ${d};
	echo "starting extract!";
	for file in *; do 
		if [ -f "$file" ]; then 
			if [[ $file == *.tar ]]; then 
				echo "$file"
				tar -xf $file --one-top-level
			fi
		fi 
		done
	cd ..
	
done