from flask import Flask, request, render_template_string
import pandas as pd
import joblib


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
# 5. MAIN PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    student = None
    error = None
    result = None

    # ========================================================
    # WHEN USER CLICKS "ANALYZE STUDENT"
    # ========================================================

    if request.method == "POST":

        student_id = request.form.get("student_id", "").strip()

        # ----------------------------------------------------
        # FIND STUDENT
        # ----------------------------------------------------

        student_data = df[df["student_id"].astype(str) == student_id]

        if student_data.empty:

            error = f"Student ID '{student_id}' not found."

        else:

            student = student_data.iloc[0]


            # =================================================
            # SUBJECT-WISE PERFORMANCE
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
            # LEARNING GAP ANALYSIS
            # =================================================

            weak_subjects = [
                subject
                for subject, data in subjects.items()
                if data["score"] < 50
            ]


            strong_subjects = [
                subject
                for subject, data in subjects.items()
                if data["score"] >= 70
            ]


            moderate_subjects = [
                subject
                for subject, data in subjects.items()
                if 50 <= data["score"] < 70
            ]


            # -------------------------------------------------
            # CASE 1: WEAK SUBJECTS EXIST
            # -------------------------------------------------

            if weak_subjects:

                priority_subjects = sorted(
                    weak_subjects,
                    key=lambda x: subjects[x]["score"]
                )

                other_improvement_subjects = sorted(
                    moderate_subjects,
                    key=lambda x: subjects[x]["score"]
                )


            # -------------------------------------------------
            # CASE 2: ALL SUBJECTS ARE STRONG
            # -------------------------------------------------

            elif len(strong_subjects) == len(subjects):

                weakest_subject = min(
                    subjects,
                    key=lambda x: subjects[x]["score"]
                )

                priority_subjects = [weakest_subject]

                other_improvement_subjects = []


            # -------------------------------------------------
            # CASE 3: NO WEAK SUBJECTS
            # -------------------------------------------------

            else:

                priority_subject = min(
                    moderate_subjects,
                    key=lambda x: subjects[x]["score"]
                )

                priority_subjects = [priority_subject]

                other_improvement_subjects = [
                    subject
                    for subject in moderate_subjects
                    if subject != priority_subject
                ]

                other_improvement_subjects = sorted(
                    other_improvement_subjects,
                    key=lambda x: subjects[x]["score"]
                )


            # =================================================
            # PREPARE INPUT FOR ML MODEL
            # =================================================

            student_input = student_data.drop(
                columns=["final_score", "student_id"]
            )


            # =================================================
            # PREDICT FINAL SCORE
            # =================================================

            predicted_score = model.predict(student_input)[0]

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
            # PERSONALIZED RECOMMENDATIONS
            # =================================================

            recommendations = []


            # -------------------------------------------------
            # ATTENDANCE
            # -------------------------------------------------

            attendance = float(
                student["attendance_percentage"]
            )

            if attendance < 75:

                recommendations.append(
                    "Improve attendance and maintain regular class participation."
                )


            # -------------------------------------------------
            # ASSIGNMENT
            # -------------------------------------------------

            assignment = float(
                student["assignment_average"]
            )

            if assignment < 60:

                recommendations.append(
                    "Practice and complete assignments regularly."
                )


            # -------------------------------------------------
            # QUIZ
            # -------------------------------------------------

            quiz = float(
                student["quiz_average"]
            )

            if quiz < 60:

                recommendations.append(
                    "Practice short quizzes regularly to improve concept retention."
                )


            # -------------------------------------------------
            # STUDY HOURS
            # -------------------------------------------------

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

            # Priority subjects

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


            # Other moderate subjects

            for subject in other_improvement_subjects:

                score = subjects[subject]["score"]

                recommendations.append(
                    f"Continue practicing {subject} to improve its performance."
                )


            # -------------------------------------------------
            # PARTICIPATION
            # -------------------------------------------------

            participation = float(
                student["participation_score"]
            )

            if participation < 60:

                recommendations.append(
                    "Increase participation in classes and academic activities."
                )


            # -------------------------------------------------
            # IF EVERYTHING IS GOOD
            # -------------------------------------------------

            if len(recommendations) == 0:

                recommendations.append(
                    "Maintain the current study habits and continue regular revision."
                )


            # =================================================
            # LEARNING GAP DISPLAY
            # =================================================

            if weak_subjects:

                learning_gap_type = "weak"

            elif len(strong_subjects) == len(subjects):

                learning_gap_type = "all_strong"

            else:

                learning_gap_type = "moderate"


            # =================================================
            # STORE RESULT
            # =================================================

            result = {
                "student_id": student_id,

                "student": student,

                "subjects": subjects,

                "weak_subjects": weak_subjects,

                "strong_subjects": strong_subjects,

                "moderate_subjects": moderate_subjects,

                "priority_subjects": priority_subjects,

                "other_improvement_subjects":
                    other_improvement_subjects,

                "learning_gap_type":
                    learning_gap_type,

                "actual_score":
                    float(student["final_score"]),

                "predicted_score":
                    float(predicted_score),

                "performance_category":
                    performance_category,

                "risk_level":
                    risk_level,

                "recommendations":
                    recommendations
            }


    # ========================================================
    # HTML PAGE
    # ========================================================

    html = """

<!DOCTYPE html>

<html>

<head>

    <title>
        AI Student Performance & Learning Analytics System
    </title>


    <style>

        body {

            font-family: Arial, sans-serif;

            background-color: #f4f6f9;

            margin: 0;

            padding: 0;

        }


        .container {

            width: 85%;

            max-width: 1000px;

            margin: 30px auto;

        }


        h1 {

            text-align: center;

            color: #1f3c88;

            margin-bottom: 30px;

        }


        .search-box {

            background: white;

            padding: 25px;

            border-radius: 10px;

            box-shadow: 0 2px 10px rgba(0,0,0,0.1);

            text-align: center;

            margin-bottom: 25px;

        }


        input {

            width: 60%;

            padding: 13px;

            font-size: 16px;

            border: 1px solid #ccc;

            border-radius: 6px;

        }


        button {

            margin-top: 15px;

            padding: 12px 25px;

            background-color: #2563eb;

            color: white;

            border: none;

            border-radius: 6px;

            font-size: 16px;

            cursor: pointer;

        }


        button:hover {

            background-color: #1d4ed8;

        }


        .error {

            background-color: #fee2e2;

            color: #b91c1c;

            padding: 15px;

            border-radius: 6px;

            margin-bottom: 20px;

            text-align: center;

        }


        .section {

            background: white;

            padding: 25px;

            margin-bottom: 20px;

            border-radius: 10px;

            box-shadow: 0 2px 8px rgba(0,0,0,0.08);

        }


        .section h2 {

            color: #1f3c88;

            border-bottom: 2px solid #e5e7eb;

            padding-bottom: 10px;

        }


        .info-grid {

            display: grid;

            grid-template-columns: 1fr 1fr;

            gap: 12px;

        }


        .info-item {

            padding: 8px;

        }


        table {

            width: 100%;

            border-collapse: collapse;

        }


        th, td {

            padding: 12px;

            border-bottom: 1px solid #ddd;

            text-align: left;

        }


        th {

            background-color: #f1f5f9;

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


        .prediction {

            font-size: 20px;

            line-height: 1.8;

        }


        .recommendations li {

            margin-bottom: 10px;

        }


        .summary {

            background-color: #eef2ff;

            padding: 20px;

            border-radius: 8px;

            line-height: 1.8;

        }


    </style>

</head>


<body>


<div class="container">


    <h1>
        AI Student Performance & Learning Analytics System
    </h1>


    <!-- SEARCH -->

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


    <!-- ================================================= -->
    <!-- STUDENT PROFILE -->
    <!-- ================================================= -->

    <div class="section">

        <h2>Student Profile</h2>

        <div class="info-grid">

            <div class="info-item">
                <strong>Student ID:</strong>
                {{ result.student_id }}
            </div>

            <div class="info-item">
                <strong>Gender:</strong>
                {{ result.student["gender"] }}
            </div>

            <div class="info-item">
                <strong>Age:</strong>
                {{ result.student["age"] }}
            </div>

            <div class="info-item">
                <strong>Department:</strong>
                {{ result.student["department"] }}
            </div>

            <div class="info-item">
                <strong>Year:</strong>
                {{ result.student["year"] }}
            </div>

        </div>

    </div>


    <!-- ================================================= -->
    <!-- ACADEMIC INFORMATION -->
    <!-- ================================================= -->

    <div class="section">

        <h2>Academic Information</h2>

        <div class="info-grid">

            <div>
                <strong>Attendance:</strong>
                {{ "%.2f"|format(result.student["attendance_percentage"]) }} %
            </div>

            <div>
                <strong>Study Hours / Day:</strong>
                {{ "%.2f"|format(result.student["study_hours_per_day"]) }}
            </div>

            <div>
                <strong>Assignment Average:</strong>
                {{ "%.2f"|format(result.student["assignment_average"]) }}
            </div>

            <div>
                <strong>Quiz Average:</strong>
                {{ "%.2f"|format(result.student["quiz_average"]) }}
            </div>

            <div>
                <strong>Participation Score:</strong>
                {{ "%.2f"|format(result.student["participation_score"]) }}
            </div>

            <div>
                <strong>Previous Semester Score:</strong>
                {{ "%.2f"|format(result.student["previous_semester_score"]) }}
            </div>

            <div>
                <strong>Previous GPA:</strong>
                {{ "%.2f"|format(result.student["previous_gpa"]) }}
            </div>

            <div>
                <strong>Midterm Score:</strong>
                {{ "%.2f"|format(result.student["midterm_score"]) }}
            </div>

            <div>
                <strong>Project Score:</strong>
                {{ "%.2f"|format(result.student["project_score"]) }}
            </div>

        </div>

    </div>


    <!-- ================================================= -->
    <!-- SUBJECT PERFORMANCE -->
    <!-- ================================================= -->

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


    <!-- ================================================= -->
    <!-- LEARNING GAP ANALYSIS -->
    <!-- ================================================= -->

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
                    {{ "%.2f"|format(result.subjects[subject].score) }}
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
                    {{ "%.2f"|format(result.subjects[subject].score) }}

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
                    {{ "%.2f"|format(result.subjects[subject].score) }}
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
                    {{ "%.2f"|format(result.subjects[subject].score) }}
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
                    {{ "%.2f"|format(result.subjects[subject].score) }}

                </li>

                {% endfor %}

            </ul>

            {% endif %}

        {% endif %}

    </div>


    <!-- ================================================= -->
    <!-- PERFORMANCE PREDICTION -->
    <!-- ================================================= -->

    <div class="section">

        <h2>Performance Prediction</h2>

        <div class="prediction">

            <div>
                <strong>Actual Final Score:</strong>
                {{ "%.2f"|format(result.actual_score) }}
            </div>

            <div>
                <strong>Predicted Final Score:</strong>
                {{ "%.2f"|format(result.predicted_score) }}
            </div>

            <div>
                <strong>Performance Category:</strong>
                {{ result.performance_category }}
            </div>

            <div>
                <strong>Risk Level:</strong>
                {{ result.risk_level }}
            </div>

        </div>

    </div>


    <!-- ================================================= -->
    <!-- PERSONALIZED RECOMMENDATIONS -->
    <!-- ================================================= -->

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


    <!-- ================================================= -->
    <!-- FINAL SUMMARY -->
    <!-- ================================================= -->

    <div class="section">

        <h2>Final Summary</h2>

        <div class="summary">

            <strong>Student:</strong>
            {{ result.student_id }}

            <br>

            <strong>Predicted Score:</strong>
            {{ "%.2f"|format(result.predicted_score) }}

            <br>

            <strong>Performance:</strong>
            {{ result.performance_category }}

            <br>

            <strong>Risk:</strong>
            {{ result.risk_level }}

            <br>

            <strong>Priority Improvement:</strong>

            {% for subject in result.priority_subjects %}

                {{ subject }}{% if not loop.last %}, {% endif %}

            {% endfor %}

        </div>

    </div>


    {% endif %}


</div>


</body>

</html>

"""


    return render_template_string(
        html,
        result=result,
        error=error
    )


# ============================================================
# 6. RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)
