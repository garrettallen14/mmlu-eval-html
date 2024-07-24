const subjectSelect = document.getElementById('subjectSelect');
const overallScores = document.getElementById('overallScores');
const subjectResults = document.getElementById('subjectResults');
const subjectTitle = document.getElementById('subjectTitle');
const resultsSummary = document.getElementById('resultsSummary');
const resultsTable = document.getElementById('resultsTable');
const homeButton = document.getElementById('homeButton');
const mainHeader = document.getElementById('mainHeader');

const subjects = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions"
];

// Populate subject select
subjects.forEach(subject => {
    const option = document.createElement('option');
    option.value = subject;
    option.textContent = subject.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    subjectSelect.appendChild(option);
});

// Load overall scores
function loadOverallScores() {
    fetch('./scores/overall_scores.json')
        .then(response => response.json())
        .then(data => {
            const tableBody = document.querySelector('#scoresTable tbody');
            tableBody.innerHTML = ''; // Clear existing content
            subjects.forEach(subject => {
                const score = data[subject];
                const row = `
                    <tr>
                        <td><span class="subject-link" data-subject="${subject}">${subject.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span></td>
                        <td class="score">${(score * 100).toFixed(2)}%</td>
                    </tr>
                `;
                tableBody.innerHTML += row;
            });

            // Add click event listeners to subject links
            document.querySelectorAll('.subject-link').forEach(link => {
                link.addEventListener('click', (event) => {
                    const subject = event.target.dataset.subject;
                    loadSubjectResults(subject);
                    subjectSelect.value = subject;
                });
            });
        })
        .catch(error => console.error('Error loading overall scores:', error));
}

// Initial load of overall scores
loadOverallScores();

// Handle subject selection
subjectSelect.addEventListener('change', (event) => {
    const subject = event.target.value;
    if (subject) {
        loadSubjectResults(subject);
    } else {
        showOverallScores();
    }
});

// Home button functionality
homeButton.addEventListener('click', showOverallScores);

// Header functionality
mainHeader.addEventListener('click', showOverallScores);

function showOverallScores() {
    overallScores.style.display = 'block';
    subjectResults.style.display = 'none';
    subjectSelect.value = '';
}

function loadSubjectResults(subject) {
    fetch(`./scores/results_${subject}.json`)
        .then(response => response.json())
        .then(data => {
            overallScores.style.display = 'none';
            subjectResults.style.display = 'block';
            subjectTitle.textContent = subject.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            
            const correctCount = data.filter(item => item.is_correct).length;
            const totalCount = data.length;
            const accuracy = (correctCount / totalCount * 100).toFixed(2);
            
            resultsSummary.textContent = `Accuracy: ${accuracy}% (${correctCount}/${totalCount})`;
            
            const tableBody = resultsTable.querySelector('tbody');
            tableBody.innerHTML = '';
            
            data.forEach((item, index) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.prompt.split('\n\nQuestion: ')[1].split('\n\nChoices:')[0]}</td>
                    <td>${item.prompt.split('\n\nChoices:\n')[1].split('\n\nAnswer:')[0].replace(/\n/g, '<br>')}</td>
                    <td class="${item.is_correct ? 'correct' : 'incorrect'}">${item.model_answer}</td>
                    <td>${item.correct_answer}</td>
                    <td>${item.is_correct ? 'Correct' : 'Incorrect'}</td>
                `;
                tableBody.appendChild(row);
            });
        })
        .catch(error => console.error(`Error loading results for ${subject}:`, error));
}