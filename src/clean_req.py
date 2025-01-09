def remove_lines_with_at_symbol(input_file='requirements.txt', output_file='clean_requirements.txt'):
    """
    Removes lines containing '@' from the input file and writes the cleaned content to the output file.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if '@' not in line:
                outfile.write(line)

if __name__=="__main__":
    remove_lines_with_at_symbol('./requirements.txt','./clean_requirements.txt')