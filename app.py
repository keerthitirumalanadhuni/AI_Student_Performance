from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import json


# ============================================================
# 1. FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. LOAD MODEL AND DATASET
# ============================================================

model = joblib.load("models/student_performance_model.pkl")

df = pd.read_csv("dataset/student_performance_clean.csv")


# ============================================================
# 3. SUBJECT INFORMATION
# ============================================================

subject_columns = {
    "Mathematics": "mathematics_score",
    "Programming": "programming_score",
    "DBMS": "dbms_score",
    "Operating Systems": "operating_systems_score",
    "Computer Networks": "computer_networks_score"
}


# ============================================================
# 4. SUBJECT PERFORMANCE CATEGORY
# ============================================================

def get_subject_category(score):

    if score >= 70:
        return "Strong"

    elif score >= 50:
        return "Moderate"

    else:
        return "Weak"


# ============================================================
# 5. MAIN ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        student_id = request.form.get(
            "student_id",
            ""
        ).strip()

        student_data = df[
            df["student_id"].astype(str) == student_id
        ]


        # ====================================================
        # STUDENT NOT FOUND
        # ====================================================

        if student_data.empty:

            error = f"Student ID '{student_id}' not found."


        else:

            student = student_data.iloc[0]


            # =================================================
            # SUBJECT PERFORMANCE
            # =================================================

            subjects = {}

            for subject, column in subject_columns.items():

                score = float(student[column])

                category = get_subject_category(score)

                subjects[subject] = {
                    "score": score,
                    "category": category
                }


            # =================================================
            # SUBJECT GROUPS
            # =================================================

            weak_subjects = [
                subject
                for subject, data in subjects.items()
                if data["score"] < 50
            ]

            moderate_subjects = [
                subject
                for subject, data in subjects.items()
                if 50 <= data["score"] < 70
            ]

            strong_subjects = [
                subject
                for subject, data in subjects.items()
                if data["score"] >= 70
            ]


            # =================================================
            # LEARNING GAP LOGIC
            # =================================================

            if weak_subjects:

                priority_subjects = sorted(
                    weak_subjects,
                    key=lambda x: subjects[x]["score"]
                )

                other_improvement_subjects = sorted(
                    moderate_subjects,
                    key=lambda x: subjects[x]["score"]
                )

                learning_gap_type = "weak"


            elif len(strong_subjects) == len(subjects):

                weakest_subject = min(
                    subjects,
                    key=lambda x: subjects[x]["score"]
                )

                priority_subjects = [
                    weakest_subject
                ]

                other_improvement_subjects = []

                learning_gap_type = "all_strong"


            else:

                priority_subject = min(
                    moderate_subjects,
                    key=lambda x: subjects[x]["score"]
                )

                priority_subjects = [
                    priority_subject
                ]

                other_improvement_subjects = sorted(
                    [
                        subject
                        for subject in moderate_subjects
                        if subject != priority_subject
                    ],
                    key=lambda x: subjects[x]["score"]
                )

                learning_gap_type = "moderate"


            # =================================================
            # MODEL INPUT
            # =================================================

            student_input = student_data.drop(
                columns=[
                    "final_score",
                    "student_id"
                ]
            )


            # =================================================
            # PREDICTION
            # =================================================

            predicted_score = model.predict(
                student_input
            )[0]

            predicted_score = max(
                0,
                min(100, predicted_score)
            )


            # =================================================
            # PERFORMANCE CATEGORY
            # =================================================

            if predicted_score >= 80:

                performance_category = "Excellent"

            elif predicted_score >= 70:

                performance_category = "Good"

            elif predicted_score >= 50:

                performance_category = "Average"

            else:

                performance_category = "At Risk"


            # =================================================
            # RISK LEVEL
            # =================================================

            if predicted_score < 50:

                risk_level = "High Risk"

            elif predicted_score < 70:

                risk_level = "Medium Risk"

            else:

                risk_level = "Low Risk"


            # =================================================
            # RECOMMENDATIONS
            # =================================================

            recommendations = []


            attendance = float(
                student["attendance_percentage"]
            )

            if attendance < 75:

                recommendations.append(
                    "Improve attendance and maintain regular class participation."
                )


            assignment = float(
                student["assignment_average"]
            )

            if assignment < 60:

                recommendations.append(
                    "Practice and complete assignments regularly."
                )


            quiz = float(
                student["quiz_average"]
            )

            if quiz < 60:

                recommendations.append(
                    "Practice short quizzes regularly to improve concept retention."
                )


            study_hours = float(
                student["study_hours_per_day"]
            )

            if study_hours < 3:

                recommendations.append(
                    "Increase daily study time and maintain a consistent study schedule."
                )


            # =================================================
            # SUBJECT RECOMMENDATIONS
            # =================================================

            for subject in priority_subjects:

                score = subjects[subject]["score"]

                if score < 50:

                    recommendations.append(
                        f"Prioritize {subject} ({score:.2f}) and work on improving its concepts and practice."
                    )

                else:

                    recommendations.append(
                        f"Focus more on {subject} ({score:.2f}) to strengthen your performance."
                    )


            for subject in other_improvement_subjects:

                recommendations.append(
                    f"Continue practicing {subject} to improve its performance."
                )


            participation = float(
                student["participation_score"]
            )

            if participation < 60:

                recommendations.append(
                    "Increase participation in classes and academic activities."
                )


            if not recommendations:

                recommendations.append(
                    "Maintain the current study habits and continue regular revision."
                )


            # =================================================
            # CHART DATA
            # =================================================

            subject_chart_labels = list(
                subjects.keys()
            )

            subject_chart_values = [
                round(data["score"], 2)
                for data in subjects.values()
            ]


            academic_labels = [
                "Attendance",
                "Assignment",
                "Quiz",
                "Participation",
                "Midterm",
                "Project"
            ]

            academic_values = [

                round(
                    float(
                        student["attendance_percentage"]
                    ),
                    2
                ),

                round(
                    float(
                        student["assignment_average"]
                    ),
                    2
                ),

                round(
                    float(
                        student["quiz_average"]
                    ),
                    2
                ),

                round(
                    float(
                        student["participation_score"]
                    ),
                    2
                ),

                round(
                    float(
                        student["midterm_score"]
                    ),
                    2
                ),

                round(
                    float(
                        student["project_score"]
                    ),
                    2
                )
            ]


            # =================================================
            # FINAL RESULT
            # =================================================

            result = {

                "student_id": student_id,

                "student": student,

                "subjects": subjects,

                "weak_subjects":
                    weak_subjects,

                "moderate_subjects":
                    moderate_subjects,

                "strong_subjects":
                    strong_subjects,

                "priority_subjects":
                    priority_subjects,

                "other_improvement_subjects":
                    other_improvement_subjects,

                "learning_gap_type":
                    learning_gap_type,

                "actual_score":
                    float(
                        student["final_score"]
                    ),

                "predicted_score":
                    float(
                        predicted_score
                    ),

                "performance_category":
                    performance_category,

                "risk_level":
                    risk_level,

                "recommendations":
                    recommendations,

                "subject_labels":
                    json.dumps(
                        subject_chart_labels
                    ),

                "subject_values":
                    json.dumps(
                        subject_chart_values
                    ),

                "academic_labels":
                    json.dumps(
                        academic_labels
                    ),

                "academic_values":
                    json.dumps(
                        academic_values
                    )
            }


    # ========================================================
    # HTML
    # ========================================================

    html = """

<!DOCTYPE html>

<html>

<head>

<title>
AI Student Performance & Learning Analytics System
</title>


<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    font-family: Arial, sans-serif;

    background: #f4f7fb;

    color: #1f2937;

}


.header {

    background: #1f3c88;

    color: white;

    padding: 25px;

    text-align: center;

}


.header h1 {

    margin: 0;

}


.container {

    width: 90%;

    max-width: 1200px;

    margin: 30px auto;

}


.search-box {

    background: white;

    padding: 25px;

    border-radius: 12px;

    box-shadow: 0 3px 12px rgba(0,0,0,0.08);

    text-align: center;

    margin-bottom: 25px;

}


input {

    width: 60%;

    padding: 13px;

    border: 1px solid #cbd5e1;

    border-radius: 7px;

    font-size: 16px;

}


button {

    padding: 13px 28px;

    margin-top: 15px;

    border: none;

    border-radius: 7px;

    background: #2563eb;

    color: white;

    font-size: 16px;

    cursor: pointer;

}


button:hover {

    background: #1d4ed8;

}


.error {

    background: #fee2e2;

    color: #b91c1c;

    padding: 15px;

    border-radius: 8px;

    text-align: center;

    margin-bottom: 20px;

}


.section {

    background: white;

    padding: 25px;

    margin-bottom: 25px;

    border-radius: 12px;

    box-shadow: 0 3px 12px rgba(0,0,0,0.06);

}


.section h2 {

    margin-top: 0;

    color: #1f3c88;

    border-bottom: 2px solid #e5e7eb;

    padding-bottom: 10px;

}


.grid {

    display: grid;

    grid-template-columns: repeat(
        auto-fit,
        minmax(200px, 1fr)
    );

    gap: 15px;

}


.card {

    background: #f8fafc;

    padding: 18px;

    border-radius: 10px;

}


.card-title {

    font-size: 14px;

    color: #64748b;

}


.card-value {

    font-size: 22px;

    font-weight: bold;

    margin-top: 7px;

}


.charts {

    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 25px;

}


.chart-container {

    position: relative;

    height: 350px;

}


table {

    width: 100%;

    border-collapse: collapse;

}


th, td {

    padding: 13px;

    border-bottom: 1px solid #e5e7eb;

}


th {

    background: #f1f5f9;

    text-align: left;

}


.strong {

    color: #15803d;

    font-weight: bold;

}


.moderate {

    color: #ca8a04;

    font-weight: bold;

}


.weak {

    color: #dc2626;

    font-weight: bold;

}


.prediction-grid {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 15px;

}


.prediction-card {

    padding: 20px;

    background: #f8fafc;

    border-radius: 10px;

    text-align: center;

}


.prediction-number {

    font-size: 28px;

    font-weight: bold;

    margin-top: 8px;

}


.recommendations li {

    margin-bottom: 12px;

}


.summary {

    background: #eef2ff;

    padding: 20px;

    border-radius: 10px;

    line-height: 2;

}


@media(max-width: 800px) {

    .charts {

        grid-template-columns: 1fr;

    }

    .prediction-grid {

        grid-template-columns: 1fr 1fr;

    }

    input {

        width: 90%;

    }

}

</style>

</head>


<body>


<div class="header">

<h1>
AI Student Performance & Learning Analytics System
</h1>

<p>
Student Performance Prediction & Personalized Learning Analysis
</p>

</div>


<div class="container">


<!-- ===================================================== -->
<!-- SEARCH -->
<!-- ===================================================== -->

<div class="search-box">

<form method="POST">

<input

type="text"

name="student_id"

placeholder="Enter Student ID (e.g. STU100001)"

value="{{ request.form.get('student_id', '') }}"

required

>

<br>

<button type="submit">
Analyze Student
</button>

</form>

</div>


{% if error %}

<div class="error">

{{ error }}

</div>

{% endif %}


{% if result %}


<!-- ===================================================== -->
<!-- STUDENT PROFILE -->
<!-- ===================================================== -->

<div class="section">

<h2>Student Profile</h2>


<div class="grid">


<div class="card">

<div class="card-title">
Student ID
</div>

<div class="card-value">
{{ result.student_id }}
</div>

</div>


<div class="card">

<div class="card-title">
Gender
</div>

<div class="card-value">
{{ result.student["gender"] }}
</div>

</div>


<div class="card">

<div class="card-title">
Age
</div>

<div class="card-value">
{{ result.student["age"] }}
</div>

</div>


<div class="card">

<div class="card-title">
Department
</div>

<div class="card-value">
{{ result.student["department"] }}
</div>

</div>


<div class="card">

<div class="card-title">
Year
</div>

<div class="card-value">
{{ result.student["year"] }}
</div>

</div>


</div>

</div>


<!-- ===================================================== -->
<!-- ACADEMIC INFORMATION -->
<!-- ===================================================== -->

<div class="section">

<h2>Academic Information</h2>


<div class="grid">


<div class="card">

<div class="card-title">
Attendance
</div>

<div class="card-value">

{{ "%.2f"|format(
result.student["attendance_percentage"]
) }} %

</div>

</div>


<div class="card">

<div class="card-title">
Study Hours / Day
</div>

<div class="card-value">

{{ "%.2f"|format(
result.student["study_hours_per_day"]
) }}

</div>

</div>


<div class="card">

<div class="card-title">
Assignment Average
</div>

<div class="card-value">

{{ "%.2f"|format(
result.student["assignment_average"]
) }}

</div>

</div>


<div class="card">

<div class="card-title">
Quiz Average
</div>

<div class="card-value">

{{ "%.2f"|format(
result.student["quiz_average"]
) }}

</div>

</div>


<div class="card">

<div class="card-title">
Participation
</div>

<div class="card-value">

{{ "%.2f"|format(
result.student["participation_score"]
) }}

</div>

</div>


<div class="card">

<div class="card-title">
Previous Semester
</div>

<div class="card-value">

{{ "%.2f"|format(
result.student["previous_semester_score"]
) }}

</div>

</div>


<div class="card">

<div class="card-title">
Previous GPA
</div>

<div class="card-value">

{{ "%.2f"|format(
result.student["previous_gpa"]
) }}

</div>

</div>


<div class="card">

<div class="card-title">
Midterm
</div>

<div class="card-value">

{{ "%.2f"|format(
result.student["midterm_score"]
) }}

</div>

</div>


<div class="card">

<div class="card-title">
Project
</div>

<div class="card-value">

{{ "%.2f"|format(
result.student["project_score"]
) }}

</div>

</div>


</div>

</div>


<!-- ===================================================== -->
<!-- CHARTS -->
<!-- ===================================================== -->

<div class="section">

<h2>Performance Analytics</h2>


<div class="charts">


<div class="chart-container">

<canvas id="subjectChart"></canvas>

</div>


<div class="chart-container">

<canvas id="academicChart"></canvas>

</div>


</div>

</div>


<!-- ===================================================== -->
<!-- SUBJECT PERFORMANCE -->
<!-- ===================================================== -->

<div class="section">

<h2>Subject-Wise Performance</h2>


<table>

<tr>

<th>Subject</th>

<th>Score</th>

<th>Performance</th>

</tr>


{% for subject, data in result.subjects.items() %}

<tr>

<td>
{{ subject }}
</td>

<td>
{{ "%.2f"|format(data.score) }}
</td>

<td class="{{ data.category|lower }}">

{{ data.category }}

</td>

</tr>

{% endfor %}

</table>

</div>


<!-- ===================================================== -->
<!-- LEARNING GAP -->
<!-- ===================================================== -->

<div class="section">

<h2>Learning Gap Analysis</h2>


{% if result.learning_gap_type == "weak" %}


<p>
<strong>Critical Learning Gaps:</strong>
</p>


<ul>

{% for subject in result.weak_subjects %}

<li>

{{ subject }}:

{{ "%.2f"|format(
result.subjects[subject].score
) }}

(Weak)

</li>

{% endfor %}

</ul>


{% if result.other_improvement_subjects %}

<p>
<strong>Other Areas for Improvement:</strong>
</p>


<ul>

{% for subject in result.other_improvement_subjects %}

<li>

{{ subject }}:

{{ "%.2f"|format(
result.subjects[subject].score
) }}

</li>

{% endfor %}

</ul>

{% endif %}


{% elif result.learning_gap_type == "all_strong" %}


<p>
All subjects are in the Strong category.
</p>


<p>
<strong>Priority Area for Continuous Improvement:</strong>
</p>


<ul>

{% for subject in result.priority_subjects %}

<li>

{{ subject }}:

{{ "%.2f"|format(
result.subjects[subject].score
) }}

(Strong)

</li>

{% endfor %}

</ul>


{% else %}


<p>
No critical learning gaps identified.
</p>


<p>
<strong>Priority Area for Improvement:</strong>
</p>


<ul>

{% for subject in result.priority_subjects %}

<li>

{{ subject }}:

{{ "%.2f"|format(
result.subjects[subject].score
) }}

(Moderate)

</li>

{% endfor %}

</ul>


{% if result.other_improvement_subjects %}

<p>
<strong>Other Areas for Improvement:</strong>
</p>


<ul>

{% for subject in result.other_improvement_subjects %}

<li>

{{ subject }}:

{{ "%.2f"|format(
result.subjects[subject].score
) }}

</li>

{% endfor %}

</ul>

{% endif %}

{% endif %}

</div>


<!-- ===================================================== -->
<!-- PREDICTION -->
<!-- ===================================================== -->

<div class="section">

<h2>Performance Prediction</h2>


<div class="prediction-grid">


<div class="prediction-card">

<div>
Actual Final Score
</div>

<div class="prediction-number">

{{ "%.2f"|format(
result.actual_score
) }}

</div>

</div>


<div class="prediction-card">

<div>
Predicted Final Score
</div>

<div class="prediction-number">

{{ "%.2f"|format(
result.predicted_score
) }}

</div>

</div>


<div class="prediction-card">

<div>
Performance
</div>

<div class="prediction-number">

{{ result.performance_category }}

</div>

</div>


<div class="prediction-card">

<div>
Risk Level
</div>

<div class="prediction-number">

{{ result.risk_level }}

</div>

</div>


</div>

</div>


<!-- ===================================================== -->
<!-- ACTUAL VS PREDICTED -->
<!-- ===================================================== -->

<div class="section">

<h2>Actual vs Predicted Final Score</h2>


<div class="chart-container">

<canvas id="predictionChart"></canvas>

</div>

</div>


<!-- ===================================================== -->
<!-- RECOMMENDATIONS -->
<!-- ===================================================== -->

<div class="section">

<h2>Personalized Recommendations</h2>


<ol class="recommendations">

{% for recommendation in result.recommendations %}

<li>

{{ recommendation }}

</li>

{% endfor %}

</ol>

</div>


<!-- ===================================================== -->
<!-- FINAL SUMMARY -->
<!-- ===================================================== -->

<div class="section">

<h2>Final Summary</h2>


<div class="summary">

<strong>Student:</strong>

{{ result.student_id }}

<br>


<strong>Predicted Score:</strong>

{{ "%.2f"|format(
result.predicted_score
) }}

<br>


<strong>Performance:</strong>

{{ result.performance_category }}

<br>


<strong>Risk:</strong>

{{ result.risk_level }}

<br>


<strong>Priority Improvement:</strong>

{% for subject in result.priority_subjects %}

{{ subject }}

{% if not loop.last %}, {% endif %}

{% endfor %}

</div>

</div>


{% endif %}


</div>


{% if result %}

<script>


// =========================================================
// SUBJECT PERFORMANCE CHART
// =========================================================

const subjectLabels =
{{ result.subject_labels | safe }};

const subjectValues =
{{ result.subject_values | safe }};


new Chart(
    document.getElementById(
        "subjectChart"
    ),
    {

        type: "bar",

        data: {

            labels: subjectLabels,

            datasets: [{

                label: "Subject Score",

                data: subjectValues

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                y: {

                    beginAtZero: true,

                    max: 100

                }

            }

        }

    }
);


// =========================================================
// ACADEMIC METRICS CHART
// =========================================================

const academicLabels =
{{ result.academic_labels | safe }};

const academicValues =
{{ result.academic_values | safe }};


new Chart(
    document.getElementById(
        "academicChart"
    ),
    {

        type: "bar",

        data: {

            labels: academicLabels,

            datasets: [{

                label: "Academic Score",

                data: academicValues

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                y: {

                    beginAtZero: true,

                    max: 100

                }

            }

        }

    }
);


// =========================================================
// ACTUAL VS PREDICTED CHART
// =========================================================

new Chart(
    document.getElementById(
        "predictionChart"
    ),
    {

        type: "bar",

        data: {

            labels: [
                "Actual Final Score",
                "Predicted Final Score"
            ],

            datasets: [{

                label: "Score",

                data: [

                    {{ result.actual_score }},

                    {{ result.predicted_score }}

                ]

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                y: {

                    beginAtZero: true,

                    max: 100

                }

            }

        }

    }
);

</script>

{% endif %}


</body>

</html>

"""


    return render_template_string(
        html,
        result=result,
        error=error
    )


# ============================================================
# 6. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)
