def generate_filename_list(folder_name="rgb", 
                           input_file=r"D:\Desktop\EECS568\Project\DenseFusionMap\data\slam\times.txt", 
                           output_file=r"D:\Desktop\EECS568\Project\DenseFusionMap\data\slam\rgb.txt"):
    # Read timestamps from the input file, ignoring empty lines
    with open(input_file, "r") as f:
        timestamps = [line.strip() for line in f if line.strip()]

    # Open the output file to write the new list with the folder name prepended to each filename
    with open(output_file, "w") as f:
        for i, timestamp in enumerate(timestamps, start=1):
            # Generate filename with a 6-digit number padded with zeros and .png extension.
            # Prepend folder_name followed by a forward slash.
            filename = f"{folder_name}/{i:06d}.png"
            # Write the timestamp and the generated filename to the output file
            f.write(f"{timestamp} {filename}\n")

    print(f"Output written to {output_file}")


if __name__ == "__main__":
    generate_filename_list(folder_name="rgb")
    generate_filename_list(folder_name="depth", 
                           output_file=r"D:\Desktop\EECS568\Project\DenseFusionMap\data\slam\depth.txt")
