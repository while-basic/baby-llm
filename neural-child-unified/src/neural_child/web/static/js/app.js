// Neural Child Development System - Minimal JavaScript

// Utility function for API calls
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Update state display
async function updateState() {
    try {
        const state = await apiCall('/api/state');
        console.log('Current state:', state);
        // State will be updated by the template's inline script
    } catch (error) {
        console.error('Failed to update state:', error);
    }
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', updateState);
} else {
    updateState();
}

