/**
 * Comparison Feature Module
 * Enterprise-grade implementation for multi-file data comparison
 */

(function () {
    'use strict';

    // Wait for DOM and main app to be ready
    document.addEventListener('DOMContentLoaded', function () {
        initializeComparisonFeature();
    });

    function initializeComparisonFeature() {
        // DOM Elements
        const comparisonModal = document.getElementById('comparisonModal');
        const closeComparisonModal = document.getElementById('closeComparisonModal');
        const compBaseFile = document.getElementById('compBaseFile');
        const compBaseX = document.getElementById('compBaseX');
        const compSeriesContainer = document.getElementById('compSeriesContainer');
        const addCompSeriesBtn = document.getElementById('addCompSeriesBtn');
        const runComparisonBtn = document.getElementById('runComparisonBtn');
        const composerCompareBtn = document.getElementById('composerCompareBtn');

        // Check if all required elements exist
        if (!comparisonModal || !runComparisonBtn) {
            console.warn('Comparison feature: Required DOM elements not found');
            return;
        }

        // Open comparison modal
        window.openComparisonModal = function () {
            if (comparisonModal) {
                comparisonModal.style.display = 'flex';
                populateBaseFiles();
            }
        };

        // Close modal handler
        if (closeComparisonModal) {
            closeComparisonModal.onclick = function () {
                comparisonModal.style.display = 'none';
            };
        }

        // Composer button handler
        if (composerCompareBtn) {
            composerCompareBtn.onclick = function () {
                window.openComparisonModal();
                const dropdown = document.getElementById('composerDropdown');
                if (dropdown) dropdown.classList.remove('show');
            };
        }

        // Populate base files dropdown
        function populateBaseFiles() {
            if (!compBaseFile) return;

            compBaseFile.innerHTML = '<option value="">-- Select Base File --</option>';
            if (window.uploadedFiles) {
                window.uploadedFiles.forEach(function (file) {
                    const opt = document.createElement('option');
                    opt.value = file.filename;
                    opt.textContent = file.filename;
                    compBaseFile.appendChild(opt);
                });
            }
        }

        // Base file change handler - populate X columns
        if (compBaseFile) {
            compBaseFile.onchange = function () {
                const filename = compBaseFile.value;
                if (!compBaseX) return;

                compBaseX.innerHTML = '<option value="">Loading...</option>';

                if (!filename) return;

                // Find file info from uploaded files
                const file = window.uploadedFiles ? window.uploadedFiles.find(f => f.filename === filename) : null;
                if (file && file.columns) {
                    compBaseX.innerHTML = '<option value="">-- Select X Column --</option>';
                    file.columns.forEach(function (col) {
                        const opt = document.createElement('option');
                        opt.value = col;
                        opt.textContent = col;
                        compBaseX.appendChild(opt);
                    });

                    // Auto-select time column if available
                    if (file.time_column) {
                        compBaseX.value = file.time_column;
                    }
                }
            };
        }

        // Add comparison series
        if (addCompSeriesBtn) {
            addCompSeriesBtn.onclick = function () {
                if (!compSeriesContainer) return;

                const row = document.createElement('div');
                row.className = 'comp-series-row';
                row.style.display = 'flex';
                row.style.gap = '0.5rem';
                row.style.marginBottom = '0.5rem';
                row.style.alignItems = 'center';

                row.innerHTML = `
                    <select class="form-select comp-file-select" style="flex:1;">
                        <option value="">Select File</option>
                    </select>
                    <select class="form-select comp-y-select" style="flex:1;">
                        <option value="">Select Y Column</option>
                    </select>
                    <input type="text" class="form-input comp-label" placeholder="Label" style="flex:1;">
                    <button class="btn-outline remove-series" style="color:red; border:none;">✕</button>
                `;

                compSeriesContainer.appendChild(row);

                // Populate files
                const fileSelect = row.querySelector('.comp-file-select');
                if (window.uploadedFiles) {
                    window.uploadedFiles.forEach(function (file) {
                        const opt = document.createElement('option');
                        opt.value = file.filename;
                        opt.textContent = file.filename;
                        fileSelect.appendChild(opt);
                    });
                }

                // Handle file change to populate columns
                fileSelect.onchange = function () {
                    const filename = fileSelect.value;
                    const ySelect = row.querySelector('.comp-y-select');
                    ySelect.innerHTML = '<option value="">Select Y Column</option>';

                    const file = window.uploadedFiles ? window.uploadedFiles.find(f => f.filename === filename) : null;
                    if (file && file.columns) {
                        file.columns.forEach(function (col) {
                            const opt = document.createElement('option');
                            opt.value = col;
                            opt.textContent = col;
                            ySelect.appendChild(opt);
                        });
                    }
                };

                // Handle remove
                row.querySelector('.remove-series').onclick = function () {
                    row.remove();
                };
            };
        }

        // Run comparison
        if (runComparisonBtn) {
            runComparisonBtn.onclick = async function () {
                const baseFile = compBaseFile ? compBaseFile.value : '';
                const baseX = compBaseX ? compBaseX.value : '';

                if (!baseFile || !baseX) {
                    if (window.showToast) {
                        window.showToast('Please select base file and X column', 'warning');
                    }
                    return;
                }

                // Collect series
                const series = [];
                const rows = document.querySelectorAll('.comp-series-row');
                rows.forEach(function (row) {
                    const file = row.querySelector('.comp-file-select').value;
                    const y = row.querySelector('.comp-y-select').value;
                    const label = row.querySelector('.comp-label').value;

                    if (file && y) {
                        series.push({
                            filename: file,
                            y_column: y,
                            label: label || `${file} - ${y}`
                        });
                    }
                });

                if (series.length === 0) {
                    if (window.showToast) {
                        window.showToast('Please add at least one comparison series', 'warning');
                    }
                    return;
                }

                // Disable button and show loading
                runComparisonBtn.disabled = true;
                runComparisonBtn.textContent = 'Comparing...';

                try {
                    const response = await fetch('/api/tools/compare', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: window.sessionId ? window.sessionId() : 'default',
                            base_file: baseFile,
                            base_x_column: baseX,
                            comparison_files: series
                        })
                    });

                    const result = await response.json();

                    if (result.success) {
                        if (window.showToast) {
                            window.showToast('Comparison successful!', 'success');
                        }
                        comparisonModal.style.display = 'none';

                        // Display comparison chart inline
                        if (window.addMessage) {
                            window.addMessage(
                                `✅ **Comparison Complete**\n${result.summary || 'Comparison generated successfully.'}`,
                                'bot',
                                result.chart_image_url,
                                result.chart_id,
                                result.plotly_json
                            );
                        }
                    } else {
                        if (window.showToast) {
                            window.showToast(result.error || 'Comparison failed', 'error');
                        }
                    }
                } catch (error) {
                    console.error('Comparison error:', error);
                    if (window.showToast) {
                        window.showToast('Comparison failed', 'error');
                    }
                } finally {
                    runComparisonBtn.disabled = false;
                    runComparisonBtn.innerHTML = '<span>Run Comparison</span>';
                }
            };
        }
    }
})();
