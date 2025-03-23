import os

directory = "./"

for filename in os.listdir(directory):
    if filename.endswith(".png.png"):
        old_path = os.path.join(directory, filename)
        new_filename = filename[:-4]
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)

        print(f"Renamed: {filename} -> {new_filename}")
