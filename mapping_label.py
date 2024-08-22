def map_label(old_file_path, new_file_path, keywords):
    lines = []
    # Read the content of the file
    with open(old_file_path, 'r') as file:
        for line in file:
            if any(keyword in line for keyword in keywords):
                lines.append(line)

    # Extract unique labels from the new file
    unique_labels_new = set(int(line.split()[-1]) for line in lines)
    sorted_labels_new = sorted(unique_labels_new)

    # Mapping old labels to new labels starting from 0 using the previously created mapping
    label_mapping_new = {label: idx for idx, label in enumerate(sorted_labels_new)}

    # Apply the mapping to the new file and save the results to another new file
    mapped_lines_new = [line.rsplit(' ', 1)[0] + ' ' + str(label_mapping_new[int(line.split()[-1])]) + '\n' for line in lines]

    # Write the modified data to a new file
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(mapped_lines_new)

if __name__ == "__main__":
    keywords_list = ['Thu-dog']
    map_label('datasets/Jittor4/classes.txt', 'datasets/Jittor4/classes_dog.txt', keywords_list)
    map_label('datasets/Jittor4/train_label.txt', 'datasets/Jittor4/train_label_dog.txt', keywords_list)
    map_label('datasets/Jittor4/val.txt', 'datasets/Jittor4/val_dog.txt', keywords_list)
