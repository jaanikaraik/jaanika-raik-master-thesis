#!/usr/bin/python3
# Open the input file
with open("printOutHD.txt", "r") as f:
    lines = f.readlines()

# Initialize variables to store the shortie and its index
shortie = None
shortie_index = None

# Initialize a list to store the results
results = []

# Loop over the lines in the file
for i, line in enumerate(lines):
    # Check if the line contains "_500"
    if "_500" in line:
        # Store the shortie and its index
        shortie = list(map(float, lines[i+1][1:-2].split()))
        shortie_index = i
    # Check if the line contains "_2000" and the shortie has been found
    elif shortie is not None and "_2000" in line:
        # Extract the longie
        longie = list(map(float, line[line.find("[")+1:line.find("]")].split()))

        # Compute the ratios between the longie and the shortie
        ratios = [a/b for a,b in zip(longie, shortie)]

        # Add the result to the list
        results.append((shortie, longie, ratios))

        # Reset the shortie and its index
        shortie = None
        shortie_index = None

# Print the results
for shortie, longie, ratios in results:
    print(f"Shortie: {shortie}")
    print(f"Longie: {longie}")
    print(f"Ratios: {ratios}")
    print()
