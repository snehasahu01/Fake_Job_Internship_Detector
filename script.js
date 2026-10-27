const API_URL = "https://jobshield-ai-1f2r.onrender.com/predict";

const payload = {
    title: document.getElementById("title").value,
    company_name: document.getElementById("company_name").value,
    contract_type: document.getElementById("contract_type").value,
    experience_letter_terms:
        document.getElementById("experience_letter_terms").value,
    package_benefits_detail:
        document.getElementById("package_benefits_detail").value,
    contract_withholding_clause:
        Number(document.getElementById("contract_withholding_clause").value),
    has_official_email:
        Number(document.getElementById("has_official_email").value)
};

const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
});

if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error ${response.status}: ${errorText}`);
}

const result = await response.json();

console.log(result);