document.getElementById('prediction-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const spinner = document.getElementById('spinner');

    submitBtn.classList.add('loading');
    spinner.classList.remove('class-hidden');
    resultContainer.classList.remove('show');

    const data = {
        temperature: document.getElementById('temperature').value,
        dissolved_oxygen: document.getElementById('dissolved_oxygen').value,
        ph: document.getElementById('ph').value,
        ammonia: document.getElementById('ammonia').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (response.ok && result.success) {
            displayResult(result);
        } else {
            alert('Failed to get prediction: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Network error occurred while fetching prediction.');
        console.error(error);
    } finally {
        submitBtn.classList.remove('loading');
        spinner.classList.add('class-hidden');
    }
});

function displayResult(result) {
    const prediction = result.prediction;
    const confidence = typeof result.confidence === 'number'
        ? `${(result.confidence * 100).toFixed(1)}%`
        : 'N/A';
    const topFactors = Array.isArray(result.top_factors) ? result.top_factors : [];
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    const ruleAlerts = Array.isArray(result.rule_alerts) ? result.rule_alerts : [];
    const probabilities = result.probabilities || {};
    const suggestion = result.suggestion || '';
    const resultContainer = document.getElementById('result-container');
    const state = result.status || prediction.toLowerCase();

    let iconClass = '';
    let colorClass = '';

    if (state === 'stable' || prediction.toLowerCase() === 'low') {
        iconClass = 'ph-check-circle';
        colorClass = 'low';
    } else if (state === 'critical' || prediction.toLowerCase() === 'high') {
        iconClass = 'ph-warning';
        colorClass = 'high';
    } else {
        iconClass = 'ph-warning-circle';
        colorClass = 'medium';
    }

    const modelMarkup = result.selected_model
        ? `<div class="result-meta"><strong>Model:</strong> ${result.selected_model}</div>`
        : '';

    const factorsMarkup = topFactors.length
        ? `<div class="result-meta"><strong>Top factors:</strong> ${topFactors
            .map((item) => `${item.feature} (${Number(item.importance).toFixed(3)})`)
            .join(', ')}</div>`
        : '';

    const probabilitiesMarkup = Object.keys(probabilities).length
        ? `<div class="probability-grid">${Object.entries(probabilities)
            .map(([label, value]) => `
                <div class="probability-item">
                    <div class="probability-row">
                        <span>${label}</span>
                        <strong>${(Number(value) * 100).toFixed(1)}%</strong>
                    </div>
                    <div class="probability-bar">
                        <span style="width: ${(Number(value) * 100).toFixed(1)}%"></span>
                    </div>
                </div>
            `)
            .join('')}</div>`
        : '';

    const suggestionMarkup = suggestion
        ? `<div class="suggestion-box">${suggestion}</div>`
        : '';

    const warningMarkup = warnings.length
        ? `<div class="warning-box"><strong>Input warnings:</strong><br>${warnings.join('<br>')}</div>`
        : '';

    const alertMarkup = ruleAlerts.length
        ? `<div class="alert-box"><strong>Critical signals:</strong><br>${ruleAlerts.join('<br>')}</div>`
        : '';

    resultContainer.innerHTML = `
        <div class="result-card ${colorClass}">
            <div class="result-label">Water Quality Risk Level</div>
            <div class="result-value ${colorClass}">
                <i class="ph-fill ${iconClass}"></i>
                <span>${prediction} Risk</span>
            </div>
            <div class="result-meta"><strong>Confidence:</strong> ${confidence}</div>
            ${modelMarkup}
            ${suggestionMarkup}
            ${alertMarkup}
            ${warningMarkup}
            ${probabilitiesMarkup}
            ${factorsMarkup}
        </div>
    `;

    setTimeout(() => {
        resultContainer.classList.add('show');
    }, 100);
}
