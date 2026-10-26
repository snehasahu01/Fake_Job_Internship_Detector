const API_BASE = "https://jobshield-ai-1f2r.onrender.com";

const payload = {
    title: "Machine Learning Engineer",
    company_name: "Microsoft",
    contract_type: "Full-time",
    experience_letter_terms:
        "Experience letter provided after employment",
    package_benefits_detail:
        "Salary and benefits discussed during interview",
    contract_withholding_clause: 0,
    has_official_email: 1
};

const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
});

const result = await response.json();
console.log(result);