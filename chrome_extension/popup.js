document.addEventListener('DOMContentLoaded', () => {
    // UI Elements map
    const form = document.getElementById('fraud-form');
    const submitBtn = document.getElementById('submit-btn');
    const errorContainer = document.getElementById('error-message');
    const outputSection = document.getElementById('output-section');
    
    // Output UI labels
    const riskLevelEl = document.getElementById('risk-level');
    const fraudProbEl = document.getElementById('fraud-prob');
    const fraudDecisionEl = document.getElementById('fraud-decision');

    const API_URL = "http://127.0.0.1:8000/predict";

    form.addEventListener('submit', async (e) => {
        // Prevent default submit navigation
        e.preventDefault();

        // Hide old states
        errorContainer.classList.add('hidden');
        outputSection.classList.add('hidden');
        errorContainer.textContent = '';

        // Safely extract values out of the DOM fields
        const payload = {
            transaction_amount: parseFloat(document.getElementById('transaction_amount').value),
            card_type: document.getElementById('card_type').value,
            user_location: document.getElementById('user_location').value.toUpperCase(),
            transaction_frequency: parseInt(document.getElementById('transaction_frequency').value, 10),
            device_type: document.getElementById('device_type').value
        };

        // Basic validation checking
        if (isNaN(payload.transaction_amount) || payload.transaction_amount <= 0) {
            showError("Please enter a valid amount.");
            return;
        }

        // Apply loading UX
        setLoading(true);

        try {
            // Ping the FastAPI Server configured with Uvicorn
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            // Handle API errors explicitly
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `Server returned ${response.status}`);
            }

            const data = await response.json();

            // Populate successful response to the UI
            displayOutput(data);

        } catch (error) {
            console.error("Fetch Error:", error);
            showError(`Failed to reach the API server: ${error.message}. Is your backend running?`);
        } finally {
            // Restore button logic
            setLoading(false);
        }
    });

    /**
     * UI HELPER FUNCTIONS 
     */

    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Checking Risk...';
            submitBtn.classList.add('loading');
        } else {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Check Fraud Risk';
            submitBtn.classList.remove('loading');
        }
    }

    function showError(msg) {
        errorContainer.textContent = msg;
        errorContainer.classList.remove('hidden');
    }

    function displayOutput(data) {
        const { risk_level, fraud_probability, is_fraud } = data;
        
        // Populate text content
        riskLevelEl.textContent = risk_level;
        fraudProbEl.textContent = `${(fraud_probability * 100).toFixed(2)} %`;
        fraudDecisionEl.textContent = is_fraud === 1 ? "BLOCKED ❌" : "CLEARED ✅";

        // Remove any previous color coding classes on the badge
        riskLevelEl.className = 'badge';

        // Add correct CSS class based on returned Risk schema logic
        if (risk_level === 'LOW') {
            riskLevelEl.classList.add('bg-green');
        } else if (risk_level === 'MEDIUM') {
            riskLevelEl.classList.add('bg-orange');
        } else if (risk_level === 'HIGH') {
            riskLevelEl.classList.add('bg-red');
        }

        // Show completed output block overlay
        outputSection.classList.remove('hidden');
    }
});
