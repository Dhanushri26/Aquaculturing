document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // UI Elements
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const spinner = document.getElementById('spinner');
    
    // Set Loading State
    submitBtn.classList.add('loading');
    spinner.classList.remove('class-hidden');
    resultContainer.classList.remove('show');
    
    // Gather values
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
            displayResult(result.prediction);
        } else {
            console.error(result.error || 'Server error occurred');
            alert('Failed to get prediction: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Network error occurred while fetching prediction.');
    } finally {
        // Remove Loading State
        submitBtn.classList.remove('loading');
        spinner.classList.add('class-hidden');
    }
});

function displayResult(prediction) {
    const resultContainer = document.getElementById('result-container');
    
    // Map prediction string to state class for CSS (low, medium, high)
    const state = prediction.toLowerCase();
    
    let iconClass = '';
    let colorClass = '';
    
    if (state === 'low') {
        iconClass = 'ph-check-circle';
        colorClass = 'low';
    } else if (state === 'medium') {
        iconClass = 'ph-warning-circle';
        colorClass = 'medium';
    } else if (state === 'high') {
        iconClass = 'ph-warning';
        colorClass = 'high';
    }
    
    // Inject HTML Result
    resultContainer.innerHTML = `
        <div class="result-card ${colorClass}">
            <div class="result-label">Water Quality Risk Level</div>
            <div class="result-value ${colorClass}">
                <i class="ph-fill ${iconClass}"></i>
                <span>${prediction} Risk</span>
            </div>
        </div>
    `;
    
    // Trigger animation
    setTimeout(() => {
        resultContainer.classList.add('show');
    }, 100);
}
