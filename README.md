# Customer Segmentation Using K-Means Clustering

## Overview
This repository showcases a customer segmentation analysis using K-means clustering on E-commerce customer data. The project aims to identify distinct customer groups based on their behaviors and characteristics, enabling businesses to tailor marketing strategies and enhance customer engagement.

## Project Details
The analysis includes:
- **Data Cleaning:** Handling missing values and removing duplicate entries to ensure data integrity.
- **Feature Selection:** Utilizing key features such as Time on App, Time on Website, Length of Membership, and Yearly Amount Spent to capture customer behavior.
- **Data Standardization:** Standardizing numerical features using StandardScaler to prepare the data for clustering algorithms.
- **Clustering Methodology:** Implementing the Elbow Method to determine the optimal number of clusters (k) and applying K-means clustering to segment customers into meaningful groups.
- **Visualization:** Creating insightful visualizations using seaborn and matplotlib to illustrate the identified customer segments.

## What I Have Achieved
- **Robust Data Processing:** Developed a comprehensive data preprocessing pipeline to ensure high-quality input for clustering analysis.
- **Effective Segmentation:** Successfully applied unsupervised learning techniques to uncover distinct customer segments based on behavioral patterns.
- **Clear Presentation:** Generated clear and informative visualizations to present the clustering results, facilitating easy interpretation and actionable insights.

## Interpretation of Segments
Based on the cluster analysis, we have identified three distinct customer segments:

### Cluster 0
- **Time on App:** 11.72 units
- **Time on Website:** 37.37 units
- **Length of Membership:** 2.58 years
- **Yearly Amount Spent:** $422.47
- **Total Price:** $1113.02

**Characteristics:**
- Customers in this segment spend relatively less time on the app but a significant amount of time on the website.
- They have the shortest membership duration compared to other segments.
- They have the lowest yearly spending and total price, indicating they may be newer or less engaged customers.

**Marketing Strategy:**
- Encourage app usage by offering app-exclusive discounts and promotions.
- Provide personalized onboarding and engagement strategies to increase their loyalty and spending.
- Target them with introductory offers to increase their interaction and spending over time.

### Cluster 1
- **Time on App:** 12.83 units
- **Time on Website:** 37.57 units
- **Length of Membership:** 4.14 years
- **Yearly Amount Spent:** $571.45
- **Total Price:** $2396.38

**Characteristics:**
- These customers spend the most time on both the app and the website.
- They have the longest membership duration, indicating strong loyalty.
- They have the highest yearly spending and total price, making them the most valuable segment.

**Marketing Strategy:**
- Focus on retention strategies to keep these high-value customers engaged.
- Offer exclusive deals, early access to new products, and loyalty rewards.
- Personalize communication and product recommendations based on their usage patterns and preferences.

### Cluster 2
- **Time on App:** 11.67 units
- **Time on Website:** 36.30 units
- **Length of Membership:** 3.90 years
- **Yearly Amount Spent:** $508.86
- **Total Price:** $2010.34

**Characteristics:**
- Customers in this segment spend less time on the app compared to Cluster 1 but still have substantial engagement on the website.
- They have a long membership duration, indicating they are relatively loyal.
- Their yearly spending and total price are moderate, positioning them between Clusters 0 and 1 in terms of value.

**Marketing Strategy:**
- Increase app engagement by highlighting the benefits and features of the app.
- Provide tailored offers and recommendations to boost their spending.
- Develop loyalty programs that cater to their preferences and enhance their overall experience.

## Visualizations
1. **Pairplot:** Visual representation of relationships between key features:
   ![image](https://github.com/user-attachments/assets/9a7c437f-4419-4808-b5e4-5db3155f6260)

2. **Elbow Method:** Determining the optimal number of clusters (k):
![image](https://github.com/user-attachments/assets/ecb55886-4f82-49a9-80cd-a1ab02f34a87)
The optimal number of clusters is 3, as seen in the above figure.

4. **Cluster Analysis:** Mean values of features across identified clusters:
![image](https://github.com/user-attachments/assets/374340d6-56e6-4d54-8536-9ff3f68a0bc4)
As seen, Cluster 1 spends the most amount of time engaging in the website.

6. **Customer Segments:** Scatter plot showing customer segments based on Time on App and Yearly Amount Spent:
![image](https://github.com/user-attachments/assets/2a6718a3-3475-4088-8d5b-ba560ca8358c)
Similar results are also shown in this diagram.


## Usage
Explore the detailed implementation and analysis steps in the Jupyter Notebook (`Customer_Segmentation_KMeans.ipynb`). The notebook provides a step-by-step guide to replicating the analysis and understanding the methodology behind customer segmentation using K-means clustering.

## Feedback and Contributions
Feedback, suggestions, and contributions are welcome! Feel free to open issues for discussion or submit pull requests with improvements to the code or analysis techniques.

#Code
```bash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Load the data
data = pd.read_csv('/E-commerce Customer Data.csv')

# Drop rows with missing email addresses
data = data.dropna(subset=['\tEmail'])

# Remove duplicate rows if any
data.drop_duplicates(inplace=True)

# Define features for clustering
features = ['Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']
X = data[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pairplot to visualize relationships between features
sns.pairplot(data[features])
plt.title('Pairplot of E-commerce Customer Data')
plt.savefig('images/pairplot.png')  # Save the plot as an image
plt.show()

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.savefig('images/elbow_method.png')  # Save the plot as an image
plt.show()

# Fit K-Means clustering model with the optimal number of clusters (k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
data['Cluster'] = clusters

# Analyze the clusters, excluding non-numeric columns
cluster_analysis = data.groupby('Cluster').mean(numeric_only=True)
print(cluster_analysis)

# Create the 'images' directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Visualize cluster centers using a bar plot
cluster_centers = kmeans.cluster_centers_
cluster_center_df = pd.DataFrame(cluster_centers, columns=features)
cluster_center_df = cluster_center_df.T.reset_index()
cluster_center_df.columns = ['Feature', 'Cluster 0', 'Cluster 1', 'Cluster 2']  # Adjust cluster labels as needed

plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='value', hue='variable', data=pd.melt(cluster_center_df, id_vars=['Feature']), palette='viridis')
plt.title('Cluster Analysis: Mean Values of Features across Clusters')
plt.xlabel('Features')
plt.ylabel('Mean Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/cluster_analysis.png')  # Save the plot as an image
plt.show()

# Visualize customer segments on a scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Time on App', y='Yearly Amount Spent', hue='Cluster', data=data, palette='viridis')
plt.title('Customer Segments')
plt.savefig('images/customer_segments.png')  # Save the plot as an image
plt.show()
```
