import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import seaborn as sns  # Import Seaborn for KDE plots

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to calculate Euclidean distance between two points in millimeters
def calculate_distance_mm(point1, point2, scale_mm_per_pixel):
    distance_pixels = calculate_distance(point1, point2)
    distance_mm = distance_pixels * scale_mm_per_pixel
    return distance_mm

# Global variables to store the nucleus positions for control and experimental groups
control_nucleus_positions = []
experimental_nucleus_positions = []

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global control_nucleus_positions, experimental_nucleus_positions
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the mouse click is for the control image or the experimental image
        if param == 'control':
            control_nucleus_positions.append((x, y))
            print("Control Nucleus positions:", control_nucleus_positions)
        elif param == 'experimental':
            experimental_nucleus_positions.append((x, y))
            print("Experimental Nucleus positions:", experimental_nucleus_positions)

# Paths to the image files for control and experimental groups
control_file_path = r'D:\LP\RANDOMFOREST\AD1C1.jpg'
experimental_file_path = r'D:\LP\RANDOMFOREST\AD1C1Exper.jpg'

# Load the control image
control_image = cv2.imread(control_file_path)
if control_image is None:
    print("Error: Unable to load the control image.")
    exit()
print("Control Image loaded successfully.")

# Load the experimental image
experimental_image = cv2.imread(experimental_file_path)
if experimental_image is None:
    print("Error: Unable to load the experimental image.")
    exit()
print("Experimental Image loaded successfully.")

# Display the control image for annotation
cv2.imshow('Control Image for Annotation', control_image)
cv2.setMouseCallback('Control Image for Annotation', mouse_callback, param='control')
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the experimental image for annotation
cv2.imshow('Experimental Image for Annotation', experimental_image)
cv2.setMouseCallback('Experimental Image for Annotation', mouse_callback, param='experimental')
cv2.waitKey(0)
cv2.destroyAllWindows()



# Prompt user for run number
run_number = input("Enter the run number: ")

# Create a DataFrame from the control nucleus positions
df_control_positions = pd.DataFrame(control_nucleus_positions, columns=['X', 'Y'])
# Create a DataFrame from the experimental nucleus positions
df_experimental_positions = pd.DataFrame(experimental_nucleus_positions, columns=['X', 'Y'])

# Save nucleus positions to Excel file with a unique name based on the run number
excel_control_file_name = f'control_nucleus_positions_run{run_number}.xlsx'
excel_experimental_file_name = f'experimental_nucleus_positions_run{run_number}.xlsx'

df_control_positions.to_excel(excel_control_file_name, index=False)
df_experimental_positions.to_excel(excel_experimental_file_name, index=False)

print(f"Control nucleus positions saved to '{excel_control_file_name}'.")
print(f"Experimental nucleus positions saved to '{excel_experimental_file_name}'.")



# Merge nucleus positions into a single DataFrame
merged_positions_df = pd.concat([df_control_positions, df_experimental_positions], ignore_index=True)
# Add a column indicating the group (Control or Experimental)
merged_positions_df['Group'] = ['Control'] * len(df_control_positions) + ['Experimental'] * len(df_experimental_positions)

# Save merged nucleus positions to Excel file with a unique name based on the run number
excel_merged_file_name = f'merged_nucleus_positions_run{run_number}.xlsx'
merged_positions_df.to_excel(excel_merged_file_name, index=False)

print(f"Merged nucleus positions saved to '{excel_merged_file_name}'.")




# Calculate pairwise distances between nucleus positions for the control group
control_distances = squareform(pdist(control_nucleus_positions))

# Calculate pairwise distances between nucleus positions for the experimental group
experimental_distances = squareform(pdist(experimental_nucleus_positions))

# Save distances to Excel file
control_df_distances = pd.DataFrame(control_distances, columns=[f'Nucleus_{i}' for i in range(len(control_distances))])
experimental_df_distances = pd.DataFrame(experimental_distances, columns=[f'Nucleus_{i}' for i in range(len(experimental_distances))])

control_df_distances.to_excel('control_distances.xlsx', index=False)
experimental_df_distances.to_excel('experimental_distances.xlsx', index=False)

# Convert the images from BGR to RGB color space
control_image_rgb = cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB)
experimental_image_rgb = cv2.cvtColor(experimental_image, cv2.COLOR_BGR2RGB)

# Create a figure to display images and plots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Display the control image with annotated nuclei positions
axs[0, 0].imshow(control_image_rgb)
axs[0, 0].set_title('Control Image')
axs[0, 0].axis('off')
for position in control_nucleus_positions:
    axs[0, 0].plot(position[0], position[1], marker='o', markersize=6, color='green')

# Display the experimental image with annotated nuclei positions
axs[0, 1].imshow(experimental_image_rgb)
axs[0, 1].set_title('Experimental Image')
axs[0, 1].axis('off')
for position in experimental_nucleus_positions:
    axs[0, 1].plot(position[0], position[1], marker='o', markersize=6, color='yellow')

# Scatter plot of distances for both groups
axs[0, 2].scatter(range(len(control_distances.flatten())), control_distances.flatten(), color='green', label='Control', alpha=0.5)
axs[0, 2].scatter(range(len(experimental_distances.flatten())), experimental_distances.flatten(), color='blue', label='Experimental', alpha=0.5)
axs[0, 2].set_xlabel('Pair Index')
axs[0, 2].set_ylabel('Distance (mm)')
axs[0, 2].set_title('Scatter Plot of Nucleus Distances')
axs[0, 2].legend()

# Histogram of distances for control group
axs[1, 0].hist(control_distances.flatten(), bins=20, color='green', alpha=0.5, label='Control')
axs[1, 0].set_xlabel('Distance (mm)')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].set_title('Histogram of Nucleus Distances (Control)')
axs[1, 0].legend()

# Histogram of distances for experimental group
axs[1, 1].hist(experimental_distances.flatten(), bins=20, color='blue', alpha=0.5, label='Experimental')
axs[1, 1].set_xlabel('Distance (mm)')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].set_title('Histogram of Nucleus Distances (Experimental)')
axs[1, 1].legend()

# Combined histogram for both groups
axs[1, 2].hist(control_distances.flatten(), bins=20, color='green', alpha=0.5, label='Control')
axs[1, 2].hist(experimental_distances.flatten(), bins=20, color='blue', alpha=0.5, label='Experimental')
axs[1, 2].set_xlabel('Distance (mm)')
axs[1, 2].set_ylabel('Frequency')
axs[1, 2].set_title('Combined Histogram of Nucleus Distances')
axs[1, 2].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Plot histograms of distances for both groups
plt.figure(figsize=(12, 6))

# Histogram for control group
plt.hist(control_distances.flatten(), bins=20, color='green', alpha=0.5, label='Control')

# Histogram for experimental group
plt.hist(experimental_distances.flatten(), bins=20, color='blue', alpha=0.5, label='Experimental')

plt.xlabel('Distance (mm)')
plt.ylabel('Frequency')
plt.title('Histogram of Nucleus Distances')
plt.legend()
plt.show()

# Plot KDE (frequency curve) for both groups
plt.figure(figsize=(12, 6))

# KDE for control group
sns.kdeplot(control_distances.flatten(), color='green', label='Control', fill=True, alpha=0.5)

# KDE for experimental group
sns.kdeplot(experimental_distances.flatten(), color='blue', label='Experimental', fill=True, alpha=0.5)

plt.xlabel('Distance (mm)')
plt.ylabel('Density')
plt.title('Frequency Curve of Nucleus Distances')
plt.legend()
plt.show()




# Create a figure to display images and plots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display scatter plot of control group nucleus positions
control_nucleus_positions = np.array(control_nucleus_positions)
axs[0].scatter(control_nucleus_positions[:, 0], control_nucleus_positions[:, 1], color='green')
axs[0].set_title('Control Group Nucleus Positions')
axs[0].set_xlabel('X-coordinate')
axs[0].set_ylabel('Y-coordinate')
axs[0].invert_yaxis()  # Invert y-axis to match image coordinates
axs[0].set_aspect('equal')

# Display scatter plot of experimental group nucleus positions
experimental_nucleus_positions = np.array(experimental_nucleus_positions)
axs[1].scatter(experimental_nucleus_positions[:, 0], experimental_nucleus_positions[:, 1], color='blue')
axs[1].set_title('Experimental Group Nucleus Positions')
axs[1].set_xlabel('X-coordinate')
axs[1].set_ylabel('Y-coordinate')
axs[1].invert_yaxis()  # Invert y-axis to match image coordinates
axs[1].set_aspect('equal')

# Show the plot
plt.tight_layout()
plt.show()


# Merge nucleus positions only if both control and experimental groups have the same number of nucleus positions
if len(control_nucleus_positions) == len(experimental_nucleus_positions):
    merged_positions = control_nucleus_positions + experimental_nucleus_positions
    merged_positions = np.array(merged_positions)

    # Create a figure to display images and plots
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display control group nucleus positions
    ax.scatter(control_nucleus_positions[:, 0], control_nucleus_positions[:, 1], color='green', label='Control')
    # Display experimental group nucleus positions
    ax.scatter(experimental_nucleus_positions[:, 0], experimental_nucleus_positions[:, 1], color='blue', label='Experimental')

    ax.set_title('Merged Nucleus Positions')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
else:
    print("The number of nucleus positions in the control and experimental groups is different. Skipping merging and visualization.")


# Perform ANOVA
from scipy.stats import f_oneway
try:
    f_statistic, p_value = f_oneway(control_distances.flatten(), experimental_distances.flatten())
    print("ANOVA p-value:", p_value)
except ValueError:
    print("Error occurred during ANOVA. Skipping to the next test...")

# Perform pairwise comparisons (Tukey HSD test)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
try:
    data = np.concatenate([control_distances.flatten(), experimental_distances.flatten()])
    group_labels = ['Control'] * len(control_distances.flatten()) + ['Experimental'] * len(experimental_distances.flatten())
    tukey_results = pairwise_tukeyhsd(data, group_labels)
    print(tukey_results)
except ValueError:
    print("Error occurred during Tukey HSD test. Skipping to the next test...")

# Calculate effect size
try:
    from scipy.stats import ttest_ind
    effect_size = abs(np.mean(control_distances) - np.mean(experimental_distances)) / np.std(data)
    print("Effect size (Cohen's d):", effect_size)
except ValueError:
    print("Error occurred during effect size calculation.")

# Mann-Whitney U Test (for two groups):
try:
    from scipy.stats import mannwhitneyu
    statistic, p_value = mannwhitneyu(control_distances.flatten(), experimental_distances.flatten())
    print("Mann-Whitney U test p-value:", p_value)
except ValueError:
    print("Error occurred during Mann-Whitney U test.")

# Kruskal-Wallis Test (for multiple groups):
try:
    from scipy.stats import kruskal
    statistic, p_value = kruskal(control_distances.flatten(), experimental_distances.flatten())
    print("Kruskal-Wallis test p-value:", p_value)
except ValueError:
    print("Error occurred during Kruskal-Wallis test.")



# Results of statistical tests
test_results = {
    'ANOVA': p_value,
    'Tukey HSD': tukey_results.pvalues[0],
    'Mann-Whitney U': p_value,
    'Kruskal-Wallis': p_value
}

# Plotting
plt.figure(figsize=(8, 6))

# Plot p-values
plt.barh(range(len(test_results)), test_results.values(), color='skyblue')
plt.yticks(range(len(test_results)), test_results.keys())
plt.xlabel('p-value')
plt.title('Statistical Tests')

plt.tight_layout()
plt.show()



import pandas as pd

# Number of nuclei for each group
num_control_nuclei = len(control_nucleus_positions)
num_experimental_nuclei = len(experimental_nucleus_positions)

# Summary table
summary_table = pd.DataFrame({
    'Group': ['Control', 'Experimental'],
    'Nuclei Count': [num_control_nuclei, num_experimental_nuclei],
    'ANOVA p-value': [p_value, None],
    'Tukey HSD p-value': [tukey_results.pvalues[0], None],
    'Mann-Whitney U p-value': [p_value, None],
    'Kruskal-Wallis p-value': [p_value, None]
})

# Display summary table
print("Summary Table:")
print(summary_table)
