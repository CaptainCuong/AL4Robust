# Import seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# Load an example dataset
tips = sns.load_dataset("tips")

# Create a boxplot of the total bill by day
sns.boxplot(x="day", y="total_bill", data=tips)

# Show the plot
plt.show()