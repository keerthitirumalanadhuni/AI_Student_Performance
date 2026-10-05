import pandas as pd
import joblib


# ============================================================
# 1. LOAD MODEL AND DATASET
# ============================================================

model = joblib.load(
    "models/student_performance_model.pkl"
)

df = pd.read_csv(
    "dataset/student_performance_clean.csv"
)


# ============================================================
# 2. GET STUDENT ID
# ============================================================

student_id = input(
    "Enter Student ID: "
).strip()


# ============================================================
# 3. FIND STUDENT
# ============================================================

student = df[
    df["student_id"] == student_id
]


if student.empty:

    print("\nStudent ID not found.")


else:

    # ========================================================
    # 4. STUDENT PROFILE
    # ========================================================

    print("\n========================================")
    print("           STUDENT PROFILE")
    print("========================================")

    print(
        "Student ID :",
        student["student_id"].iloc[0]
    )

    print(
        "Gender     :",
        student["gender"].iloc[0]
    )

    print(
        "Age        :",
        student["age"].iloc[0]
    )

    print(
        "Department :",
        student["department"].iloc[0]
    )

    print(
        "Year       :",
        student["year"].iloc[0]
    )


    # ========================================================
    # 5. ACADEMIC INFORMATION
    # ========================================================

    print("\n========================================")
    print("        ACADEMIC INFORMATION")
    print("========================================")

    print(
        "Attendance             :",
        round(
            student["attendance_percentage"].iloc[0],
            2
        ),
        "%"
    )

    print(
        "Study Hours / Day      :",
        round(
            student["study_hours_per_day"].iloc[0],
            2
        )
    )

    print(
        "Assignment Average     :",
        round(
            student["assignment_average"].iloc[0],
            2
        )
    )

    print(
        "Quiz Average           :",
        round(
            student["quiz_average"].iloc[0],
            2
        )
    )

    print(
        "Participation Score    :",
        round(
            student["participation_score"].iloc[0],
            2
        )
    )

    print(
        "Previous Semester Score:",
        round(
            student["previous_semester_score"].iloc[0],
            2
        )
    )

    print(
        "Previous GPA           :",
        round(
            student["previous_gpa"].iloc[0],
            2
        )
    )

    print(
        "Midterm Score          :",
        round(
            student["midterm_score"].iloc[0],
            2
        )
    )

    print(
        "Project Score          :",
        round(
            student["project_score"].iloc[0],
            2
        )
    )


    # ========================================================
    # 6. SUBJECT-WISE PERFORMANCE
    # ========================================================

    subject_scores = {

        "Mathematics":
            student["mathematics_score"].iloc[0],

        "Programming":
            student["programming_score"].iloc[0],

        "DBMS":
            student["dbms_score"].iloc[0],

        "Operating Systems":
            student["operating_systems_score"].iloc[0],

        "Computer Networks":
            student["computer_networks_score"].iloc[0]
    }


    # ========================================================
    # 7. CLASSIFY SUBJECT PERFORMANCE
    # ========================================================

    def get_subject_status(score):

        if score < 50:

            return "Weak"

        elif score < 70:

            return "Moderate"

        else:

            return "Strong"


    subject_status = {}

    for subject, score in subject_scores.items():

        subject_status[subject] = get_subject_status(
            score
        )


    # ========================================================
    # 8. DISPLAY SUBJECT-WISE PERFORMANCE
    # ========================================================

    print("\n========================================")
    print("        SUBJECT-WISE PERFORMANCE")
    print("========================================")

    for subject, score in subject_scores.items():

        print(
            f"{subject:<22}: "
            f"{score:.2f} "
            f"({subject_status[subject]})"
        )


    # ========================================================
    # 9. LEARNING GAP ANALYSIS
    # ========================================================

    weak_subjects = {

        subject: score

        for subject, score in subject_scores.items()

        if score < 50
    }


    moderate_subjects = {

        subject: score

        for subject, score in subject_scores.items()

        if 50 <= score < 70
    }


    strong_subjects = {

        subject: score

        for subject, score in subject_scores.items()

        if score >= 70
    }


    print("\n========================================")
    print("          LEARNING GAP ANALYSIS")
    print("========================================")


    # --------------------------------------------------------
    # CASE 1: WEAK SUBJECTS EXIST
    # --------------------------------------------------------

    if len(weak_subjects) > 0:

        print(
            "Critical Learning Gaps:"
        )

        # Lowest weak subject first
        weak_subjects_sorted = sorted(
            weak_subjects.items(),
            key=lambda x: x[1]
        )

        for subject, score in weak_subjects_sorted:

            print(
                f"- {subject}: "
                f"{score:.2f} (Weak)"
            )


        # Moderate subjects are also shown separately
        if len(moderate_subjects) > 0:

            print(
                "\nOther Areas for Improvement:"
            )

            moderate_sorted = sorted(
                moderate_subjects.items(),
                key=lambda x: x[1]
            )

            for subject, score in moderate_sorted:

                print(
                    f"- {subject}: "
                    f"{score:.2f} (Moderate)"
                )


    # --------------------------------------------------------
    # CASE 2: NO WEAK SUBJECTS
    # --------------------------------------------------------

    elif len(moderate_subjects) > 0:

        # Lowest moderate subject becomes priority
        priority_subject = min(
            moderate_subjects,
            key=moderate_subjects.get
        )

        priority_score = moderate_subjects[
            priority_subject
        ]


        print(
            "No critical learning gaps identified."
        )

        print(
            "\nPriority Area for Improvement:"
        )

        print(
            f"- {priority_subject}: "
            f"{priority_score:.2f} (Moderate)"
        )


        # Remaining moderate subjects
        other_moderate = {

            subject: score

            for subject, score in moderate_subjects.items()

            if subject != priority_subject
        }


        if len(other_moderate) > 0:

            print(
                "\nOther Areas for Improvement:"
            )

            other_moderate_sorted = sorted(
                other_moderate.items(),
                key=lambda x: x[1]
            )

            for subject, score in other_moderate_sorted:

                print(
                    f"- {subject}: "
                    f"{score:.2f}"
                )


    # --------------------------------------------------------
    # CASE 3: ALL SUBJECTS ARE STRONG
    # --------------------------------------------------------

    else:

        lowest_strong_subject = min(
            strong_subjects,
            key=strong_subjects.get
        )

        lowest_strong_score = strong_subjects[
            lowest_strong_subject
        ]


        print(
            "No critical learning gaps identified."
        )

        print(
            "\nAll subjects are Strong."
        )

        print(
            "\nArea for Further Improvement:"
        )

        print(
            f"- {lowest_strong_subject}: "
            f"{lowest_strong_score:.2f} (Strong)"
        )

        print(
            "Continue improving this subject because "
            "it has the lowest score among the strong subjects."
        )


    # ========================================================
    # 10. PREPARE DATA FOR ML MODEL
    # ========================================================

    student_input = student.drop(
        columns=[
            "final_score",
            "student_id"
        ]
    )


    # ========================================================
    # 11. PREDICT FINAL SCORE
    # ========================================================

    predicted_score = model.predict(
        student_input
    )[0]


    # Keep prediction between 0 and 100
    predicted_score = max(
        0,
        min(
            100,
            predicted_score
        )
    )


    # ========================================================
    # 12. PERFORMANCE CATEGORY
    # ========================================================

    if predicted_score >= 80:

        performance_category = "Excellent"

    elif predicted_score >= 70:

        performance_category = "Good"

    elif predicted_score >= 50:

        performance_category = "Average"

    else:

        performance_category = "At Risk"


    # ========================================================
    # 13. RISK LEVEL
    # ========================================================

    if predicted_score < 50:

        risk_level = "High Risk"

    elif predicted_score < 70:

        risk_level = "Medium Risk"

    else:

        risk_level = "Low Risk"


    # ========================================================
    # 14. PERFORMANCE PREDICTION
    # ========================================================

    print("\n========================================")
    print("       PERFORMANCE PREDICTION")
    print("========================================")

    print(
        "Actual Final Score    :",
        round(
            student["final_score"].iloc[0],
            2
        )
    )

    print(
        "Predicted Final Score :",
        round(
            predicted_score,
            2
        )
    )

    print(
        "Performance Category  :",
        performance_category
    )

    print(
        "Risk Level             :",
        risk_level
    )


    # ========================================================
    # 15. PERSONALIZED RECOMMENDATIONS
    # ========================================================

    recommendations = []


    # --------------------------------------------------------
    # Attendance
    # --------------------------------------------------------

    attendance = student[
        "attendance_percentage"
    ].iloc[0]


    if attendance < 75:

        recommendations.append(
            "Improve attendance and maintain "
            "regular class participation."
        )


    # --------------------------------------------------------
    # Assignment
    # --------------------------------------------------------

    assignment = student[
        "assignment_average"
    ].iloc[0]


    if assignment < 60:

        recommendations.append(
            "Practice and complete assignments "
            "regularly."
        )


    # --------------------------------------------------------
    # Quiz
    # --------------------------------------------------------

    quiz = student[
        "quiz_average"
    ].iloc[0]


    if quiz < 60:

        recommendations.append(
            "Practice short quizzes regularly "
            "to improve concept retention."
        )


    # --------------------------------------------------------
    # Study Hours
    # --------------------------------------------------------

    study_hours = student[
        "study_hours_per_day"
    ].iloc[0]


    if study_hours < 3:

        recommendations.append(
            "Increase daily study time and maintain "
            "a consistent study schedule."
        )


    # --------------------------------------------------------
    # SUBJECT RECOMMENDATIONS
    # --------------------------------------------------------

    # Weak subjects get highest priority

    for subject, score in sorted(
        weak_subjects.items(),
        key=lambda x: x[1]
    ):

        recommendations.append(
            f"Give high priority to {subject} and "
            f"revise its fundamentals."
        )


    # If there are no weak subjects,
    # recommend the lowest moderate subject first

    if len(weak_subjects) == 0:

        if len(moderate_subjects) > 0:

            priority_subject = min(
                moderate_subjects,
                key=moderate_subjects.get
            )

            priority_score = moderate_subjects[
                priority_subject
            ]


            recommendations.append(
                f"Prioritize {priority_subject} "
                f"({priority_score:.2f}) and work on "
                f"improving its concepts and practice."
            )


            # Other moderate subjects
            other_moderate = {

                subject: score

                for subject, score in moderate_subjects.items()

                if subject != priority_subject
            }


            for subject, score in sorted(
                other_moderate.items(),
                key=lambda x: x[1]
            ):

                recommendations.append(
                    f"Continue practicing {subject} "
                    f"to improve its performance."
                )


        else:

            # All subjects are strong
            lowest_subject = min(
                strong_subjects,
                key=strong_subjects.get
            )

            lowest_score = strong_subjects[
                lowest_subject
            ]


            recommendations.append(
                f"Continue improving {lowest_subject} "
                f"({lowest_score:.2f}) as it is the "
                f"lowest among the strong subjects."
            )


    # --------------------------------------------------------
    # Participation
    # --------------------------------------------------------

    participation = student[
        "participation_score"
    ].iloc[0]


    if participation < 60:

        recommendations.append(
            "Increase participation in classes "
            "and academic activities."
        )


    # --------------------------------------------------------
    # Default recommendation
    # --------------------------------------------------------

    if len(recommendations) == 0:

        recommendations.append(
            "Maintain the current study habits "
            "and continue regular revision."
        )


    # ========================================================
    # 16. DISPLAY RECOMMENDATIONS
    # ========================================================

    print("\n========================================")
    print("       PERSONALIZED RECOMMENDATIONS")
    print("========================================")

    for number, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{number}. {recommendation}"
        )


    # ========================================================
    # 17. FINAL SUMMARY
    # ========================================================

    print("\n========================================")
    print("            FINAL SUMMARY")
    print("========================================")

    print(
        "Student:",
        student_id
    )

    print(
        "Predicted Score:",
        round(
            predicted_score,
            2
        )
    )

    print(
        "Performance:",
        performance_category
    )

    print(
        "Risk:",
        risk_level
    )


    # Summary learning gap
    if len(weak_subjects) > 0:

        gap_text = ", ".join(
            weak_subjects.keys()
        )

        print(
            "Learning Gaps:",
            gap_text
        )

    elif len(moderate_subjects) > 0:

        priority_subject = min(
            moderate_subjects,
            key=moderate_subjects.get
        )

        print(
            "Priority Improvement:",
            priority_subject
        )

    else:

        lowest_subject = min(
            strong_subjects,
            key=strong_subjects.get
        )

        print(
            "Further Improvement:",
            lowest_subject
        )


    print(
        "========================================"
    )
    
