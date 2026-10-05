import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD CLEAN DATASET
# ==========================================

file_path = "dataset/student_performance_clean.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# ==========================================
# 2. BASIC STATISTICS
# ==========================================

print("\n===== FINAL SCORE STATISTICS =====")
print(df["final_score"].describe())


# ==========================================
# 3. AVERAGE SCORES BY DEPARTMENT
# ==========================================

print("\n===== DEPARTMENT-WISE AVERAGE FINAL SCORE =====")

department_scores = df.groupby("department")["final_score"].mean()

print(department_scores)


# ==========================================
# 4. AVERAGE SUBJECT SCORES
# ==========================================

subject_columns = [
    "mathematics_score",
    "programming_score",
    "dbms_score",
    "operating_systems_score",
    "computer_networks_score"
]

print("\n===== SUBJECT-WISE AVERAGE SCORES =====")

subject_averages = df[subject_columns].mean()

print(subject_averages)


# ==========================================
# 5. CORRELATION WITH FINAL SCORE
# ==========================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

correlation = df[numeric_columns].corr()["final_score"].sort_values(
    ascending=False
)

print("\n===== CORRELATION WITH FINAL SCORE =====")
print(correlation)


# ==========================================
# 6. FINAL SCORE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(df["final_score"], bins=20)

plt.title("Distribution of Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()


# ==========================================
# 7. ATTENDANCE VS FINAL SCORE
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["attendance_percentage"],
    df["final_score"],
    alpha=0.4
)

plt.title("Attendance vs Final Score")
plt.xlabel("Attendance Percentage")
plt.ylabel("Final Score")

plt.tight_layout()
plt.show()


# ==========================================
# 8. STUDY HOURS VS FINAL SCORE
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["study_hours_per_day"],
    df["final_score"],
    alpha=0.4
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours per Day")
plt.ylabel("Final Score")

plt.tight_layout()
plt.show()


# ==========================================
# 9. SUBJECT-WISE AVERAGE SCORE
# ==========================================

plt.figure(figsize=(9, 5))

subject_averages.plot(kind="bar")

plt.title("Average Score by Subject")
plt.xlabel("Subject")
plt.ylabel("Average Score")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# ==========================================
# 10. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    df[numeric_columns].corr(),
    annot=False
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()
