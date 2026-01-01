// ThermoSense AI - Smart Extraction Feature
// This module handles the "Add Metadata" modal logic

(function () {
    console.log('Extraction feature module loading...');

    function initializeExtractionFeature() {
        console.log('Initializing extraction feature...');

        // DOM Elements
        const extractionModal = document.getElementById('extractionModal');
        const extractFileSelect = document.getElementById('extractFileSelect');
        const sheetGroup = document.getElementById('sheetGroup');
        const sheetSelect = document.getElementById('sheetSelect');
        const columnGroup = document.getElementById('columnGroup');
        const columnList = document.getElementById('columnSelect');
        const extractParams = document.getElementById('extractParams');
        const extractParamsLabel = document.getElementById('extractParamsLabel');
        const performExtractionBtn = document.getElementById('performExtraction');
        const cancelExtractionBtn = document.getElementById('cancelExtraction');
        const closeExtractionModalBtn = document.getElementById('closeExtractionModal');
        const extractUploadBtn = document.getElementById('extractUploadBtn');

        if (!extractionModal || !extractFileSelect) {
            console.error('Extraction modal elements not found!');
            return;
        }

        // Helper: Populate file select dropdown
        function populateFiles() {
            if (!extractFileSelect) return;
            console.log('Populating files dropdown. window.uploadedFiles:', window.uploadedFiles);

            // Keep track of current selections if any
            const selectedValues = Array.from(extractFileSelect.selectedOptions).map(opt => opt.value);

            extractFileSelect.innerHTML = '';
            if (window.uploadedFiles && window.uploadedFiles.length > 0) {
                window.uploadedFiles.forEach(function (file) {
                    const opt = document.createElement('option');
                    // Handle both object and string formats
                    const fname = typeof file === 'string' ? file : (file.filename || file.name);
                    if (!fname) return;

                    opt.value = fname;
                    opt.textContent = fname;
                    if (selectedValues.includes(fname)) {
                        opt.selected = true;
                    }
                    extractFileSelect.appendChild(opt);
                });
            } else {
                extractFileSelect.innerHTML = '<option value="" disabled>No files uploaded</option>';
            }

            updateLabels();
        }

        async function updateLabels() {
            const selectedOptions = Array.from(extractFileSelect.selectedOptions);
            const selectedFiles = selectedOptions.map(opt => opt.value).filter(v => v !== "");
            const hasPpt = selectedFiles.some(f => f.toLowerCase().endsWith('.pptx') || f.toLowerCase().endsWith('.ppt'));
            const hasExcel = selectedFiles.some(f => f.toLowerCase().endsWith('.xlsx') || f.toLowerCase().endsWith('.xls'));

            console.log('Updating labels. Selected files:', selectedFiles);

            if (hasPpt) {
                if (extractParamsLabel) extractParamsLabel.textContent = 'Slide Numbers/Ranges (e.g. 1, 3-5)';
                if (extractParams) extractParams.placeholder = 'e.g. 1, 3-5';
            } else {
                if (extractParamsLabel) extractParamsLabel.textContent = 'Extraction Parameters (Optional)';
                if (extractParams) extractParams.placeholder = 'e.g. Extract only sensor names';
            }

            // Show/hide sheet selection for single Excel file
            if (selectedFiles.length === 1 && hasExcel) {
                await fetchSheets(selectedFiles[0]);
            } else {
                if (sheetGroup) sheetGroup.style.display = 'none';
                if (columnGroup) columnGroup.style.display = 'none';
            }
        }

        async function fetchSheets(filename) {
            if (!sheetSelect) return;
            console.log('Fetching sheets for:', filename);
            try {
                const sid = window.sessionId ? window.sessionId() : 'default';
                const resp = await fetch('/api/file/sheets', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename, session_id: sid })
                });
                const data = await resp.json();
                console.log('Sheets response:', data);
                if (data.success && data.sheets && data.sheets.length > 0) {
                    sheetSelect.innerHTML = '<option value="">-- Select Sheet --</option>';
                    data.sheets.forEach(sheet => {
                        const opt = document.createElement('option');
                        opt.value = sheet;
                        opt.textContent = sheet;
                        sheetSelect.appendChild(opt);
                    });
                    if (sheetGroup) sheetGroup.style.display = 'block';
                } else {
                    if (sheetGroup) sheetGroup.style.display = 'none';
                    console.warn('No sheets found or error:', data.error || 'Unknown error');
                }
            } catch (e) {
                console.error('Error fetching sheets:', e);
                if (sheetGroup) sheetGroup.style.display = 'none';
            }
        }

        async function fetchColumns(filename, sheetName) {
            if (!columnList) return;
            console.log('Fetching columns for:', filename, sheetName);
            try {
                const sid = window.sessionId ? window.sessionId() : 'default';
                const resp = await fetch('/api/file/columns', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename, sheet_name: sheetName, session_id: sid })
                });
                const data = await resp.json();
                console.log('Columns response:', data);
                if (data.success && data.columns && data.columns.length > 0) {
                    columnList.innerHTML = '';
                    data.columns.forEach(col => {
                        const div = document.createElement('div');
                        div.className = 'form-check';
                        div.innerHTML = `
                            <input class="form-check-input" type="checkbox" value="${col}" id="col-${col}">
                            <label class="form-check-label" for="col-${col}">${col}</label>
                        `;
                        columnList.appendChild(div);
                    });
                    if (columnGroup) columnGroup.style.display = 'block';
                } else {
                    if (columnGroup) columnGroup.style.display = 'none';
                }
            } catch (e) {
                console.error('Error fetching columns:', e);
                if (columnGroup) columnGroup.style.display = 'none';
            }
        }

        // Event Listeners
        extractFileSelect.addEventListener('change', updateLabels);

        sheetSelect.addEventListener('change', function () {
            const filename = extractFileSelect.value;
            const sheetName = sheetSelect.value;
            if (filename && sheetName) {
                fetchColumns(filename, sheetName);
            } else {
                if (columnGroup) columnGroup.style.display = 'none';
            }
        });

        if (extractUploadBtn) {
            extractUploadBtn.onclick = function () {
                const fileInput = document.getElementById('fileInput');
                if (fileInput) fileInput.click();
            };
        }

        // Listen for file-uploaded event from app.js
        document.addEventListener('file-uploaded', function () {
            console.log('Extraction feature: File uploaded event received');
            populateFiles();
        });

        // Listen for files-refreshed event from app.js
        document.addEventListener('files-refreshed', function (e) {
            console.log('Extraction feature: Files refreshed event received', e.detail);
            populateFiles();
        });

        performExtractionBtn.onclick = async function () {
            const selectedFiles = Array.from(extractFileSelect.selectedOptions).map(opt => opt.value);
            if (selectedFiles.length === 0) {
                if (window.showToast) window.showToast('Please select at least one file', 'warning');
                return;
            }

            const params = extractParams.value;
            const sheetName = sheetSelect.value;
            const selectedColumns = Array.from(columnList.querySelectorAll('input:checked')).map(cb => cb.value);
            const section = document.getElementById('sectionSelect').value;

            performExtractionBtn.disabled = true;
            performExtractionBtn.textContent = 'Extracting...';

            try {
                const sid = window.sessionId ? window.sessionId() : 'default';
                const sensorsList = params.split(',').map(s => s.trim()).filter(s => s !== "");
                const strictMode = document.getElementById('extractStrict')?.checked || false;

                const resp = await fetch('/api/extract', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sid,
                        filenames: selectedFiles,
                        sensors: sensorsList,
                        sheet_name: sheetName,
                        columns: selectedColumns,
                        section: section,
                        strict: strictMode
                    })
                });
                const data = await resp.json();

                if (data.success) {
                    if (window.showToast) window.showToast('Extraction complete!', 'success');

                    let messageText = data.message || data.summary || 'Data extracted and added to report.';
                    if (data.download_url) {
                        messageText += `\n\n[Download Extracted Data](${data.download_url})`;
                    }

                    if (window.addMessage) window.addMessage(messageText, 'bot');
                    extractionModal.style.display = 'none';
                } else {
                    if (window.showToast) window.showToast('Extraction failed: ' + data.error, 'error');
                }
            } catch (e) {
                console.error('Extraction error:', e);
                if (window.showToast) window.showToast('Extraction error: ' + e.message, 'error');
            } finally {
                performExtractionBtn.disabled = false;
                performExtractionBtn.textContent = 'Perform Extraction';
            }
        };

        cancelExtractionBtn.onclick = () => extractionModal.style.display = 'none';
        closeExtractionModalBtn.onclick = () => extractionModal.style.display = 'none';

        // Expose open function
        window.openExtractionModal = async function () {
            extractionModal.style.display = 'flex';
            if (window.refreshUploads) await window.refreshUploads();
            populateFiles();
        };

        // Initial population
        populateFiles();
    }

    // Wait for DOM
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeExtractionFeature);
    } else {
        initializeExtractionFeature();
    }
})();
