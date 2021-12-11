import os
import sys
with open('implemented.txt', 'w') as file:
    for filename in os.listdir("./"):
        if filename.endswith(".txt") and filename != 'implemented.txt': 
            file.write(filename)
            file.write('\n')