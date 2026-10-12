
import pandas as pd

test_cases = [
    {
        "title": "Junior Data Analyst",
        "company_name": "Northstar Digital",
        "contract_type": "Full-time",
        "experience_letter_terms": "Written confirmation of employment available after completion of probation.",
        "package_benefits_detail": "Salary and health insurance details provided in the signed offer.",
        "contract_withholding_clause": 0,
        "has_official_email": 1,
        "expected_label": "Review manually"
    },
    {
        "title": "Remote Data Entry Trainee",
        "company_name": "Quick Career Hub",
        "contract_type": "Remote",
        "experience_letter_terms": "Pay a processing fee before an experience certificate can be issued.",
        "package_benefits_detail": "Guaranteed high earnings and expensive gifts with no interview.",
        "contract_withholding_clause": 1,
        "has_official_email": 0,
        "expected_label": "Review manually"
    },
]

df = pd.DataFrame(test_cases)
df.to_csv("training/manual_test_cases.csv", index=False)

print("Manual test cases created successfully!")
print(df[["title", "company_name", "expected_label"]].to_string(index=False))
