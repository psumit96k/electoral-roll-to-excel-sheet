/* Main JavaScript for Election PDF to Excel Converter */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initApp();
});

function initApp() {
    // Handle file upload
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleUpload);
    }

    // Load initial data
    loadDocuments();

    // Set up periodic updates (optional)
    // setInterval(loadDocuments, 5000); // Update every 5 seconds
}

function handleUpload(e) {
    e.preventDefault();

    const formData = new FormData();
    const batchNameInput = document.getElementById('batchName');
    const fileInput = document.getElementById('pdfFiles');

    // Add batch name if provided
    if (batchNameInput && batchNameInput.value.trim()) {
        formData.append('batchName', batchNameInput.value.trim());
    }

    // Add files
    const files = fileInput.files;
    if (files.length === 0) {
        showMessage('uploadResults', 'Please select at least one PDF file', 'error');
        return;
    }

    for (let i = 0; i < files.length; i++) {
        formData.append('pdfFiles', files[i]);
    }

    // Show loading state
    showMessage('uploadResults', 'Uploading files...', 'info');

    // Disable submit button during upload
    const submitBtn = uploadForm.querySelector('button[type="submit"]');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Uploading...';
    }

    // Send upload request
    fetch('/api/v1/upload/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Re-enable button
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Upload PDFs';
        }

        if (data.message) {
            showMessage('uploadResults', data.message, 'success');
            // Clear form
            uploadForm.reset();
            // Refresh document list
            loadDocuments();
        } else {
            showMessage('uploadResults', 'Upload failed: Unknown error', 'error');
        }
    })
    .catch(error => {
        // Re-enable button
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Upload PDFs';
        }

        showMessage('uploadResults', 'Upload failed: ' + error.message, 'error');
        console.error('Upload error:', error);
    });
}

function loadDocuments() {
    fetch('/api/v1/upload/')
    .then(response => response.json())
    .then(documents => {
        const documentsList = document.getElementById('documentsList');
        if (!documentsList) return;

        if (documents.length === 0) {
            documentsList.innerHTML = '<p>No documents uploaded yet.</p>';
            return;
        }

        let html = '';
        documents.forEach(doc => {
            html += `
                <div class="document-item">
                    <div class="document-header">
                        <div class="document-title">${doc.original_filename}</div>
                        <div class="document-status status-${doc.processing_status.toLowerCase()}">
                            ${doc.processing_status}
                        </div>
                    </div>
                    <div>
                        <strong>Batch:</strong> ${doc.batch ? doc.batch.name : 'N/A'}<br>
                        <strong>Size:</strong> ${formatFileSize(doc.file_size)}<br>
                        <strong>Pages:</strong> ${doc.page_count || 'Processing...'}<br>
                        <strong>Uploaded:</strong> ${formatDate(doc.upload_timestamp)}
                    </div>
                </div>
            `;
        });

        documentsList.innerHTML = html;
    })
    .catch(error => {
        console.error('Error loading documents:', error);
        const documentsList = document.getElementById('documentsList');
        if (documentsList) {
            documentsList.innerHTML = '<p>Error loading documents.</p>';
        }
    });
}

function showMessage(elementId, message, type) {
    const element = document.getElementById(elementId);
    if (!element) return;

    // Remove existing alert classes
    element.className = '';

    // Set message
    element.textContent = message;

    // Add appropriate class based on type
    if (type === 'success') {
        element.classList.add('alert', 'alert-success');
    } else if (type === 'error') {
        element.classList.add('alert', 'alert-error');
    } else if (type === 'info') {
        element.classList.add('alert', 'alert-info');
    } else {
        element.classList.add('alert');
    }

    // Auto-hide after 5 seconds for non-error messages
    if (type !== 'error') {
        setTimeout(() => {
            element.textContent = '';
            element.className = '';
        }, 5000);
    }
}

// Helper functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    if (!dateString) return 'Unknown';

    const options = { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

// Add CSS for alerts (since we're adding them dynamically)
const style = document.createElement('style');
style.textContent = `
    .alert {
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .alert-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .alert-error {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    .alert-info {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
    }
`;
document.head.appendChild(style);