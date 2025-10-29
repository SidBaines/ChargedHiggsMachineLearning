# %%
# Define the path to your input file
input_file = 'tmp.txt'
# input_file = 'tmpqqbb.txt'

# Initialize sums for each column
sum_col1 = 0
sum_col2 = 0
sum_col3 = 0

# Open and read the file
with open(input_file, 'r') as f:
    for line in f:
        # Split the line by commas and convert to floats
        values = line.strip().split(',')
        if len(values) == 3:
            sum_col1 += float(values[0])
            sum_col2 += float(values[1])
            sum_col3 += float(values[2])

# Print the results
print(f"Sum of Column 1: {sum_col1}")
print(f"Sum of Column 2: {sum_col2}")
print(f"Sum of Column 3: {sum_col3}")

# %%
