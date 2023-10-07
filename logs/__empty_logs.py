# delete all files with .out extension
import os

for file in os.listdir("logs"):
    if file.endswith(".out"):
        os.remove(f"logs/{file}")
