const getStoredApiUrl = () => localStorage.getItem('housing_buddy_api_url');
const setStoredApiUrl = (url) => localStorage.setItem('housing_buddy_api_url', url);

// HARDCODED LOCALHOST - No caching, no confusion.
let API_BASE_URL = 'http://localhost:9000';

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const imageInput = document.getElementById('image-input');
const imagePreview = document.getElementById('image-preview');
const dropZoneContent = document.querySelector('.drop-zone-content');
const featureForm = document.getElementById('feature-form');
const submitBtn = document.getElementById('submit-btn');
const resultsSection = document.getElementById('results-section');
const priceDisplay = document.getElementById('price-display');
const heatmapImg = document.getElementById('heatmap-img');
const resetBtn = document.getElementById('reset-btn');

// Settings Modal Elements (Keeping logic but simplifying)
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const apiUrlInput = document.getElementById('api-url-input');
const saveSettingsBtn = document.getElementById('save-settings-btn');
const closeSettingsBtn = document.getElementById('close-settings-btn');

let selectedFile = null;

// Initialize Settings
apiUrlInput.value = API_BASE_URL;

settingsBtn.addEventListener('click', () => settingsModal.classList.remove('hidden'));
closeSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));

saveSettingsBtn.addEventListener('click', () => {
    let url = apiUrlInput.value.trim();
    if (url) {
        url = url.replace(/\/$/, "");
        API_BASE_URL = url;
        // setStoredApiUrl(url); // Don't save to avoid bad cache
        settingsModal.classList.add('hidden');
        console.log('API URL updated to:', url);
    }
});

// Handle Image Selection
dropZone.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('active');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('active');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) return;

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.hidden = false;
        dropZoneContent.hidden = true;
    };
    reader.readAsDataURL(file);
}

// Handle Form Submission
featureForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!selectedFile) {
        alert('Please upload a satellite image first.');
        return;
    }

    const formData = new FormData();
    const features = {};

    // Collect all inputs
    const inputs = featureForm.querySelectorAll('input');
    inputs.forEach(input => {
        if (input.value !== "") {
            features[input.name] = parseFloat(input.value);
        }
    });

    formData.append('image', selectedFile);
    formData.append('features', JSON.stringify(features));

    try {
        submitBtn.disabled = true;
        submitBtn.innerText = 'Calculating...';
        resultsSection.classList.add('hidden');

        // FORCE CORRECT ENDPOINT
        const response = await fetch(`http://localhost:9000/predict-explain`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to get valuation');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        // Show error in UI instead of modal popup loop
        resultsSection.classList.remove('hidden');
        priceDisplay.innerText = "Error";
        priceDisplay.style.color = "red";
        document.getElementById('confidence-badge').innerText = "Connection Failed";
        // settingsModal.classList.remove('hidden'); // DISABLE AUTO-OPEN
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = 'Calculate Valuation';
    }
});

function displayResults(data) {
    priceDisplay.innerText = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0
    }).format(data.predicted_price);

    if (data.explanation_url) {
        heatmapImg.src = `${API_BASE_URL}${data.explanation_url}`;
    }

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Reset functionality
resetBtn.addEventListener('click', () => {
    featureForm.reset();
    imagePreview.src = '';
    imagePreview.hidden = true;
    dropZoneContent.hidden = false;
    selectedFile = null;
    resultsSection.classList.add('hidden');
});
