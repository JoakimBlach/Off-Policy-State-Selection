for filename in config/example3_tests/*.yaml; do
    yaml_filename="$(basename "$filename")"
    pkl_filename="$(basename "$filename" .yaml).pkl"

    # Check if the output file exists
    if [ ! -f "output/example3_tests/$pkl_filename" ]; then
        echo "Processing file $filename"

        python3 run.py --num_episodes=500 --config_file="config/example3_tests/$yaml_filename" --output_file="output/example3_tests/$pkl_filename" --save_results="True" --plot="False"

        echo "File $filename done"
        echo
    else
        echo "File output/example3_tests/$pkl_filename exists."
    fi
done