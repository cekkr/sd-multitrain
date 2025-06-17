// /advanced-sd2-lora-trainer/javascript/custom.js

// This function will be called when the DOM is fully loaded for the Gradio app
function onGradioAppLoaded() {
    // We need to find our specific button. Since Gradio IDs can be dynamic,
    // we'll add a more reliable way to select it.
    // For this example, let's assume the button has a specific class or we can navigate the DOM.
    // A more robust method would be to assign a unique element ID via the Python script if Gradio allows.
    
    // This is a simplified selector for demonstration.
    const trainingButton = document.querySelector('#sd2_lora_trainer_tab .gradio-button.primary');

    if (trainingButton) {
        trainingButton.addEventListener('click', () => {
            console.log('Advanced LoRA Trainer: Training button clicked. Sending request to backend.');
            
            // Example of a simple client-side confirmation.
            // Note: This does not prevent the backend call, which has already been triggered by Gradio.
            // It serves as an immediate visual feedback cue for the user.
            const originalText = trainingButton.textContent;
            trainingButton.textContent = 'Request Sent...';

            // Reset the button text after a short delay
            setTimeout(() => {
                // Check if the button is not disabled by Gradio's processing state
                if (!trainingButton.hasAttribute('disabled')) {
                    trainingButton.textContent = originalText;
                }
            }, 3000);
        });
    }
}

// The AUTOMATIC1111 WebUI calls this function when the Gradio app is ready.
// We check if it's already defined to avoid conflicts.
if (typeof onUiLoaded === 'function') {
    onUiLoaded(onGradioAppLoaded);
} else {
    // Fallback for different WebUI versions or direct script load
    document.addEventListener('DOMContentLoaded', onGradioAppLoaded);
}