// ThermoSense AI - Complete Working Frontend
// This version uses both event listeners AND exposes functions globally for inline onclick

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const els = {
        sidebar: document.getElementById('sidebar'),
        menuBtn: document.getElementById('menuBtn'),
        sidebarClose: document.getElementById('sidebarClose'),

        // Composer
        composerBtn: document.getElementById('composerBtn'),
        composerDropdown: document.getElementById('composerDropdown'),
        composerUploadBtn: document.getElementById('composerUploadBtn'),
        composerChartBtn: document.getElementById('composerChartBtn'),
        composerExtractBtn: document.getElementById('composerExtractBtn'),
        composerCompareBtn: document.getElementById('composerCompareBtn'),

        // File upload
        fileInput: document.getElementById('fileInput'),
        uploadedFilesList: document.getElementById('uploadedFilesList'),

        // Column selector modal
        columnSelector: document.getElementById('columnSelector'),
        closeModal: document.getElementById('closeModal'),
        fileSelect: document.getElementById('fileSelect'),
        xSearch: document.getElementById('xSearch'),
        ySearch: document.getElementById('ySearch'),
        xSelect: document.getElementById('xSelect'),
        ySelect: document.getElementById('ySelect'),
        clearSelection: document.getElementById('clearSelection'),
        skipSelection: document.getElementById('skipSelection'),
        moveSelectedBtn: document.getElementById('moveSelected'),
        showGlobalColumns: document.getElementById('showGlobalColumns'),
        applyColumnsBtn: document.getElementById('applyColumns'),

        // Chart type modal
        chartTypeModal: document.getElementById('chartTypeModal'),
        closeChartTypeModal: document.getElementById('closeChartTypeModal'),

        // Smart Extraction Modal
        extractionModal: document.getElementById('extractionModal'),
        closeExtractionModal: document.getElementById('closeExtractionModal'),
        extractFileSelect: document.getElementById('extractFileSelect'),
        extractParams: document.getElementById('extractParams'),
        cancelExtraction: document.getElementById('cancelExtraction'),
        performExtraction: document.getElementById('performExtraction'),

        // Compare Files Modal
        compareModal: document.getElementById('comparisonModal'),
        closeCompareModal: document.getElementById('closeComparisonModal'),
        compareFileSelect: document.getElementById('compareFileSelect'),
        compareParam: document.getElementById('compareParam'),
        cancelCompare: document.getElementById('cancelCompare'),
        performCompare: document.getElementById('performCompare'),

        // Chat
        chatInput: document.getElementById('chatInput'),
        sendBtn: document.getElementById('sendBtn'),
        messagesDiv: document.getElementById('messages'),
        toast: document.getElementById('toast'),
        sessionList: document.getElementById('sessionList'),
        sessionInput: document.getElementById('sessionInput'),
        newChatBtn: document.getElementById('newChatBtn'),

        // Parameter Selector
        parameterSelector: document.getElementById('parameterSelector'),
        closeParamModal: document.getElementById('closeParamModal'),
        paramSearch: document.getElementById('paramSearch'),
        paramList: document.getElementById('paramList'),
        skipParamSelection: document.getElementById('skipParamSelection'),
        nextToChartType: document.getElementById('nextToChartType'),

        // Updated Column Selector
        showOnlySelectedParams: document.getElementById('showOnlySelectedParams'),
        generateChartBtn: document.getElementById('generateChartBtn')
    };

    // Global state
    let currentFileData = {};
    let selectedFile = null;
    let isProcessing = false;
    let isHeatmapMode = false;
    let selectedChartType = 'line';
    let pendingChartData = null;
    let selectedParameters = []; // New state for parameter selection
    let allCommonColumns = []; // Store unique common columns across all files

    // Utility: Get session ID
    function sessionId() {
        return localStorage.getItem('thermosense_session') || els.sessionInput?.value || 'default';
    }
    window.sessionId = sessionId;

    // Utility: Show toast notification
    function showToast(msg, type = 'info') {
        els.toast.textContent = msg;
        els.toast.className = `toast toast-${type} show`;
        setTimeout(() => els.toast.classList.remove('show'), 3000);
    }
    window.showToast = showToast;

    // Auto-resize textarea
    function autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }

    // Initialize
    function init() {
        // Create or retrieve session ID
        const urlParams = new URLSearchParams(window.location.search);
        let sid = urlParams.get('session_id') || localStorage.getItem('thermosense_session');

        if (!sid || sid === 'default') {
            sid = `session_${Date.now()}`;
            localStorage.setItem('thermosense_session', sid);
        } else {
            // Store it if it came from URL
            localStorage.setItem('thermosense_session', sid);
        }

        if (els.sessionInput) {
            els.sessionInput.value = sid;
        }

        console.log('Session initialized:', sid);

        // Explicitly hide modals on page load
        if (els.columnSelector) els.columnSelector.style.display = 'none';
        if (els.chartTypeModal) els.chartTypeModal.style.display = 'none';
        if (els.parameterSelector) els.parameterSelector.style.display = 'none';

        // Wire all event listeners
        wireEvents();

        // Load initial data
        refreshUploads();
        loadSessions();
        loadChatHistory();
    }

    // Wire all event listeners
    function wireEvents() {
        // Sidebar
        if (els.menuBtn) els.menuBtn.onclick = () => els.sidebar?.classList.add('active');
        if (els.sidebarClose) els.sidebarClose.onclick = () => els.sidebar?.classList.remove('active');

        // Composer button
        if (els.composerBtn) {
            els.composerBtn.onclick = (e) => {
                console.log('Composer button clicked');
                e.preventDefault();
                e.stopPropagation();
                const dropdown = els.composerDropdown;
                if (dropdown) {
                    dropdown.classList.toggle('show');
                }
            };
        }

        // Close dropdown when clicking outside
        window.addEventListener('click', (e) => {
            if (els.composerDropdown && els.composerDropdown.classList.contains('show')) {
                if (!els.composerBtn.contains(e.target) && !els.composerDropdown.contains(e.target)) {
                    els.composerDropdown.classList.remove('show');
                }
            }
        });

        // Composer options
        if (els.composerUploadBtn) els.composerUploadBtn.onclick = () => {
            els.fileInput?.click();
            els.composerDropdown.classList.remove('show');
        };

        // Select Chart button - opens parameter selector first
        if (els.composerChartBtn) {
            els.composerChartBtn.onclick = () => {
                openParameterSelector();
                els.composerDropdown.classList.remove('show');
            };
        }

        // File input
        if (els.fileInput) els.fileInput.onchange = () => uploadFiles();

        // File select dropdown in modal
        if (els.fileSelect) {
            els.fileSelect.onchange = () => {
                refreshColumnLists();
            };
        }

        // Modal close buttons
        if (els.closeModal) els.closeModal.onclick = () => els.columnSelector.style.display = 'none';
        if (els.closeChartTypeModal) els.closeChartTypeModal.onclick = () => els.chartTypeModal.style.display = 'none';
        if (els.closeParamModal) els.closeParamModal.onclick = () => els.parameterSelector.style.display = 'none';

        // Composer buttons
        if (els.composerExtractBtn) {
            els.composerExtractBtn.onclick = () => {
                if (window.openExtractionModal) {
                    window.openExtractionModal();
                } else {
                    els.extractionModal.style.display = 'flex';
                }
                els.composerDropdown.classList.remove('show');
            };
        }
        if (els.composerCompareBtn) {
            els.composerCompareBtn.onclick = () => {
                if (window.openComparisonModal) {
                    window.openComparisonModal();
                } else {
                    els.compareModal.style.display = 'flex';
                }
                els.composerDropdown.classList.remove('show');
            };
        }

        // Extraction Modal Handlers
        if (els.closeExtractionModal) els.closeExtractionModal.onclick = () => els.extractionModal.style.display = 'none';
        if (els.cancelExtraction) els.cancelExtraction.onclick = () => els.extractionModal.style.display = 'none';

        // Comparison Modal Handlers
        if (els.closeCompareModal) els.closeCompareModal.onclick = () => els.compareModal.style.display = 'none';
        if (els.cancelCompare) els.cancelCompare.onclick = () => els.compareModal.style.display = 'none';

        // Clear selection
        if (els.clearSelection) els.clearSelection.onclick = () => clearColumnSelection();

        // Global columns toggle
        if (els.showGlobalColumns) {
            els.showGlobalColumns.onchange = () => {
                refreshColumnLists();
            };
        }

        // Show only selected params toggle
        if (els.showOnlySelectedParams) {
            els.showOnlySelectedParams.onchange = () => {
                refreshColumnLists();
            };
        }

        // Parameter selection handlers
        if (els.skipParamSelection) els.skipParamSelection.onclick = () => handleParameterSkip();
        if (els.nextToChartType) els.nextToChartType.onclick = () => handleParameterNext();
        if (els.paramSearch) els.paramSearch.oninput = () => filterColumnList(els.paramSearch, els.paramList);

        // Generate chart button
        if (els.generateChartBtn) {
            els.generateChartBtn.onclick = () => handleGenerateChart();
        }

        // Chart type selection
        document.querySelectorAll('.chart-type-option').forEach(btn => {
            btn.onclick = () => {
                selectedChartType = btn.dataset.chartType;
                document.querySelectorAll('.chart-type-option').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Next step: Axis Configuration
                if (els.chartTypeModal) els.chartTypeModal.style.display = 'none';
                if (els.columnSelector) {
                    els.columnSelector.style.display = 'flex';
                    refreshColumnLists();
                }
            };
        });

        // Search inputs for column filtering
        if (els.xSearch) {
            els.xSearch.oninput = () => filterColumnList(els.xSearch, els.xSelect);
        }
        if (els.ySearch) {
            els.ySearch.oninput = () => filterColumnList(els.ySearch, els.ySelect);
        }

        // Chat input
        if (els.chatInput) {
            els.chatInput.oninput = () => autoResize(els.chatInput);
            els.chatInput.onkeydown = (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            };
        }

        if (els.sendBtn) els.sendBtn.onclick = () => sendMessage();

        // New chat button
        if (els.newChatBtn) els.newChatBtn.onclick = () => startNewChat();
    }

    // Upload files
    async function uploadFiles() {
        const files = els.fileInput.files;
        if (!files.length) return;

        for (let file of files) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId());

            try {
                const resp = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await resp.json();

                if (data.success) {
                    currentFileData[data.filename] = data;
                    showToast(`${data.filename} uploaded successfully`, 'success');
                    addMessage(`File "${data.filename}" uploaded.`, 'bot');
                } else {
                    showToast(`Upload failed: ${data.error}`, 'error');
                }
            } catch (e) {
                showToast(`Upload error: ${e.message}`, 'error');
            }
        }

        await refreshUploads();
        els.fileInput.value = '';

        // NEW: Open parameter selector after all uploads
        setTimeout(() => {
            openParameterSelector();
        }, 800);

        // Dispatch event for other modules
        document.dispatchEvent(new CustomEvent('file-uploaded'));
    }
    window.uploadFiles = uploadFiles;

    // Refresh uploaded files list
    async function refreshUploads() {
        try {
            const resp = await fetch(`/api/uploads?session_id=${sessionId()}`);
            const data = await resp.json();

            // CRITICAL: Expose files for other modules
            window.uploadedFiles = data.files || [];
            document.dispatchEvent(new CustomEvent('files-refreshed', { detail: window.uploadedFiles }));

            els.uploadedFilesList.innerHTML = '';
            els.fileSelect.innerHTML = '<option value="">Select a file...</option>';

            if (data.files && data.files.length > 0) {
                data.files.forEach((f) => {
                    // Sidebar list
                    const li = document.createElement('li');
                    li.textContent = f.filename;
                    li.onclick = () => {
                        els.fileSelect.value = f.filename;
                        els.sidebar.classList.remove('active');
                        refreshColumnLists();
                    };
                    els.uploadedFilesList.appendChild(li);

                    // Select dropdown
                    const opt = document.createElement('option');
                    opt.value = f.filename;
                    opt.textContent = f.filename;
                    els.fileSelect.appendChild(opt);

                    // Store file data
                    if (!currentFileData[f.filename]) {
                        currentFileData[f.filename] = f;
                    }
                });
            } else {
                els.uploadedFilesList.innerHTML = '<li style="color:#888;">No files</li>';
            }
        } catch (e) {
            console.error('Refresh error:', e);
        }
    }
    window.refreshUploads = refreshUploads;

    // --- New Workflow Functions ---

    function openParameterSelector() {
        if (!els.parameterSelector) return;

        // Collect all unique columns from all uploaded files
        const allCols = new Set();
        Object.values(currentFileData).forEach(file => {
            if (file.columns) {
                file.columns.forEach(col => allCols.add(col));
            }
        });

        allCommonColumns = Array.from(allCols).sort();

        if (allCommonColumns.length === 0) {
            console.log('No columns available for parameter selection');
            return;
        }

        populateParameterList(allCommonColumns);
        els.parameterSelector.style.display = 'flex';
    }

    function populateParameterList(columns) {
        if (!els.paramList) return;
        els.paramList.innerHTML = '';

        columns.forEach(col => {
            const div = document.createElement('div');
            div.className = 'column-item';
            div.innerHTML = `<input type="checkbox" class="column-checkbox"> <span>${col}</span>`;
            div.onclick = (e) => {
                const cb = div.querySelector('.column-checkbox');
                if (e.target !== cb) cb.checked = !cb.checked;
                div.classList.toggle('selected', cb.checked);
            };
            els.paramList.appendChild(div);
        });
    }

    function handleParameterNext() {
        const selected = Array.from(els.paramList.querySelectorAll('.column-item.selected'))
            .map(el => el.textContent.trim());

        if (selected.length === 0) {
            showToast('Please select at least one parameter or click Skip', 'warning');
            return;
        }

        selectedParameters = selected;
        els.parameterSelector.style.display = 'none';
        if (els.chartTypeModal) els.chartTypeModal.style.display = 'flex';
        fetchChartRecommendations();
    }

    function handleParameterSkip() {
        selectedParameters = []; // Empty means show all
        els.parameterSelector.style.display = 'none';
        if (els.chartTypeModal) els.chartTypeModal.style.display = 'flex';
        fetchChartRecommendations();
    }

    function refreshColumnLists() {
        const filename = els.fileSelect.value;
        if (els.showGlobalColumns && els.showGlobalColumns.checked) {
            populateGlobalColumns();
        } else if (filename && currentFileData[filename]) {
            const fileData = currentFileData[filename];
            populateColumnLists(fileData.columns, fileData.time_column, fileData.numeric_columns);
        } else {
            els.xSelect.innerHTML = '';
            els.ySelect.innerHTML = '';
        }
    }

    async function handleGenerateChart() {
        const xEl = els.xSelect.querySelector('.column-item.selected');
        const yEls = Array.from(els.ySelect.querySelectorAll('.column-item.selected'));

        if (!xEl || yEls.length === 0) {
            showToast('Please select X-axis and at least one Y-axis column', 'warning');
            return;
        }

        const filename = xEl.dataset.filename || els.fileSelect.value;
        if (!filename) {
            showToast('Please select a file', 'warning');
            return;
        }

        pendingChartData = {
            filename: filename,
            x_column: xEl.dataset.originalCol || xEl.textContent.trim(),
            y_columns: yEls.map(el => el.dataset.originalCol || el.textContent.trim())
        };

        if (els.columnSelector) els.columnSelector.style.display = 'none';
        handleColumnSelectionDirect();
    }

    // Populate column selection lists
    function populateColumnLists(columns, timeColumn, numericColumns, filename = null) {
        if (!els.xSelect || !els.ySelect) return;

        els.xSelect.innerHTML = '';
        els.ySelect.innerHTML = '';

        if (!columns || columns.length === 0) {
            els.xSelect.innerHTML = '<div style="padding:10px;color:#888;">No columns</div>';
            return;
        }

        // Filter by selectedParameters if toggle is on
        let displayCols = columns;
        if (els.showOnlySelectedParams && els.showOnlySelectedParams.checked && selectedParameters.length > 0) {
            displayCols = columns.filter(col => selectedParameters.includes(col));
        }

        displayCols.forEach(col => {
            // X-Axis (single select)
            const xDiv = document.createElement('div');
            xDiv.className = 'column-item';
            let xHtml = `<span class="column-indicator"></span> <span>${col}</span>`;
            if (filename) xHtml += `<span class="column-file-label" title="${filename}">${filename}</span>`;
            xDiv.innerHTML = xHtml;
            xDiv.dataset.originalCol = col;
            if (filename) xDiv.dataset.filename = filename;

            if (col === timeColumn) xDiv.classList.add('selected');
            xDiv.onclick = () => {
                Array.from(els.xSelect.children).forEach(c => c.classList.remove('selected'));
                xDiv.classList.add('selected');
            };
            els.xSelect.appendChild(xDiv);

            // Y-Axis (multi select)
            const yDiv = document.createElement('div');
            yDiv.className = 'column-item';
            let yHtml = `<input type="checkbox" class="column-checkbox"> <span>${col}</span>`;
            if (filename) yHtml += `<span class="column-file-label" title="${filename}">${filename}</span>`;
            yDiv.innerHTML = yHtml;
            yDiv.dataset.originalCol = col;
            if (filename) yDiv.dataset.filename = filename;

            yDiv.onclick = (e) => {
                const cb = yDiv.querySelector('.column-checkbox');
                if (e.target !== cb) {
                    cb.checked = !cb.checked;
                }
                yDiv.classList.toggle('selected', cb.checked);
            };
            els.ySelect.appendChild(yDiv);
        });
    }

    function populateGlobalColumns() {
        if (!els.xSelect || !els.ySelect) return;
        els.xSelect.innerHTML = '';
        els.ySelect.innerHTML = '';

        const files = Object.keys(currentFileData);
        if (files.length === 0) {
            els.xSelect.innerHTML = '<div style="padding:10px;color:#888;">No files uploaded</div>';
            return;
        }

        files.forEach(fname => {
            const data = currentFileData[fname];
            if (data.columns) {
                // Filter by selectedParameters if toggle is on
                let displayCols = data.columns;
                if (els.showOnlySelectedParams && els.showOnlySelectedParams.checked && selectedParameters.length > 0) {
                    displayCols = data.columns.filter(col => selectedParameters.includes(col));
                }

                displayCols.forEach(col => {
                    // X-Axis
                    const xDiv = document.createElement('div');
                    xDiv.className = 'column-item';
                    xDiv.innerHTML = `<span class="column-indicator"></span> <span>${col}</span> <span class="column-file-label" title="${fname}">${fname}</span>`;
                    xDiv.dataset.originalCol = col;
                    xDiv.dataset.filename = fname;
                    if (col === data.time_column) xDiv.classList.add('selected');
                    xDiv.onclick = () => {
                        Array.from(els.xSelect.children).forEach(c => c.classList.remove('selected'));
                        xDiv.classList.add('selected');
                    };
                    els.xSelect.appendChild(xDiv);

                    // Y-Axis
                    const yDiv = document.createElement('div');
                    yDiv.className = 'column-item';
                    yDiv.innerHTML = `<input type="checkbox" class="column-checkbox"> <span>${col}</span> <span class="column-file-label" title="${fname}">${fname}</span>`;
                    yDiv.dataset.originalCol = col;
                    yDiv.dataset.filename = fname;
                    yDiv.onclick = (e) => {
                        const cb = yDiv.querySelector('.column-checkbox');
                        if (e.target !== cb) {
                            cb.checked = !cb.checked;
                        }
                        yDiv.classList.toggle('selected', cb.checked);
                    };
                    els.ySelect.appendChild(yDiv);
                });
            }
        });
    }

    async function fetchChartRecommendations() {
        console.log('fetchChartRecommendations triggered');
        // If we have selected parameters, use them for recommendations
        const yCols = selectedParameters.length > 0 ? selectedParameters : [];

        if (yCols.length === 0) {
            console.log('No columns for recommendations');
            return;
        }

        try {
            const resp = await fetch('/api/chart/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId(),
                    y_columns: yCols
                })
            });

            if (!resp.ok) {
                throw new Error(`HTTP error! status: ${resp.status}`);
            }

            const data = await resp.json();
            console.log('Recommendation data received:', data);

            if (data.success && data.recommendations) {
                // Clear previous recommendations
                document.querySelectorAll('.chart-type-option').forEach(btn => {
                    btn.classList.remove('recommended');
                    const oldBadge = btn.querySelector('.recommendation-badge');
                    if (oldBadge) oldBadge.remove();
                    const oldReason = btn.querySelector('.recommendation-reason');
                    if (oldReason) oldReason.remove();
                });

                // Apply new recommendations
                data.recommendations.forEach(rec => {
                    const btn = document.querySelector(`.chart-type-option[data-chart-type="${rec.type}"]`);
                    if (btn) {
                        console.log('Applying recommendation for:', rec.type);
                        btn.classList.add('recommended');

                        const badge = document.createElement('div');
                        badge.className = 'recommendation-badge';
                        badge.textContent = rec.source === 'history' ? '⭐ Preferred' : '✨ Suggested';
                        btn.appendChild(badge);

                        const reason = document.createElement('div');
                        reason.className = 'recommendation-reason';
                        reason.textContent = rec.reason;
                        btn.appendChild(reason);
                    }
                });
            }
        } catch (e) {
            console.error('Failed to fetch recommendations:', e);
        }
    }

    // Handle column selection and generate chart
    async function handleColumnSelectionDirect() {
        console.log('handleColumnSelectionDirect triggered');
        let filename, xCol, yCols;

        if (pendingChartData) {
            console.log('Using pendingChartData:', pendingChartData);
            filename = pendingChartData.filename;
            xCol = pendingChartData.x_column;
            yCols = pendingChartData.y_columns;
            pendingChartData = null; // Clear after use
        } else {
            filename = els.fileSelect.value;
            const xEl = els.xSelect.querySelector('.column-item.selected');
            const yEls = Array.from(els.ySelect.querySelectorAll('.column-item.selected'));

            if (!filename || !xEl || yEls.length === 0) {
                showToast('Please select a file, X-axis, and at least one Y-axis column', 'warning');
                return;
            }

            xCol = xEl.dataset.originalCol || xEl.textContent.trim();
            yCols = yEls.map(el => el.dataset.originalCol || el.textContent.trim());
        }

        showToast('Generating chart...', 'info');

        try {
            const fileData = currentFileData[filename];
            const sheetName = fileData && fileData.sheets && fileData.sheets.length > 0 ? fileData.sheets[0] : null;

            const resp = await fetch('/api/chart/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId(),
                    filename: filename,
                    x_column: xCol,
                    y_columns: yCols,
                    chart_type: selectedChartType || 'line',
                    sheet_name: sheetName
                })
            });
            const data = await resp.json();

            if (data.success) {
                addMessage(data.summary || `Generated ${selectedChartType} chart for ${yCols.join(', ')}`, 'bot', data.chart_url, data.chart_id, data.plotly_json);
            } else {
                showToast(`Chart generation failed: ${data.error}`, 'error');
            }
        } catch (e) {
            showToast(`Error: ${e.message}`, 'error');
        }
    }
    window.handleColumnSelectionDirect = handleColumnSelectionDirect;

    function clearColumnSelection() {
        if (els.xSelect) Array.from(els.xSelect.children).forEach(c => c.classList.remove('selected'));
        if (els.ySelect) Array.from(els.ySelect.children).forEach(c => c.classList.remove('selected'));
    }

    function filterColumnList(searchInput, columnList) {
        const term = searchInput.value.toLowerCase();
        Array.from(columnList.children).forEach(item => {
            if (item.classList.contains('column-item')) {
                item.style.display = item.textContent.toLowerCase().includes(term) ? '' : 'none';
            }
        });
    }

    // Add message to chat
    function addMessage(text, sender, imageUrl = null, chartId = null, plotlyJson = null, images = []) {
        const div = document.createElement('div');
        div.className = `message ${sender}`;
        if (chartId) div.dataset.chartId = chartId;

        let htmlContent = text.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Convert markdown links
        htmlContent = htmlContent.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, linkText, url) => {
            if (url.includes('.docx') || url.includes('/workspace/')) {
                return `<a href="${url}" download class="download-link" style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1.5rem; background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); color: white; text-decoration: none; border-radius: var(--radius); font-weight: 600; margin: 0.5rem 0;">📥 ${linkText}</a>`;
            }
            return `<a href="${url}" target="_blank" style="color: var(--primary-color); text-decoration: underline;">${linkText}</a>`;
        });

        div.innerHTML = htmlContent;

        // Render Chart (Interactive or Static)
        if (plotlyJson) {
            const chartContainerId = `plotly-${chartId || Date.now()}`;
            const chartDiv = document.createElement('div');
            chartDiv.id = chartContainerId;
            chartDiv.className = 'plotly-chart-container';
            div.appendChild(chartDiv);

            setTimeout(() => {
                try {
                    let figure = typeof plotlyJson === 'string' ? JSON.parse(plotlyJson) : plotlyJson;
                    if (figure.layout) {
                        figure.layout.autosize = true;
                        delete figure.layout.width;
                        delete figure.layout.height;
                        figure.layout.margin = { t: 40, b: 40, l: 60, r: 40 };
                    }
                    Plotly.newPlot(chartContainerId, figure.data, figure.layout, {
                        responsive: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['select2d', 'lasso2d']
                    });
                } catch (e) {
                    console.error('Plotly render error:', e);
                    if (imageUrl) {
                        chartDiv.innerHTML = '';
                        const img = document.createElement('img');
                        img.src = imageUrl;
                        img.className = 'fallback-chart-img';
                        chartDiv.appendChild(img);
                    }
                }
            }, 100);
        } else if (imageUrl) {
            const img = document.createElement('img');
            img.src = imageUrl;
            img.onclick = () => window.open(imageUrl, '_blank');
            div.appendChild(img);
        }

        // Render additional RAG images
        if (images && images.length > 0) {
            const imageGrid = document.createElement('div');
            imageGrid.className = 'image-grid';
            images.forEach(imgUrl => {
                const img = document.createElement('img');
                img.src = imgUrl;
                img.onclick = () => window.open(imgUrl, '_blank');
                imageGrid.appendChild(img);
            });
            div.appendChild(imageGrid);
        }

        els.messagesDiv.appendChild(div);

        // Add Feedback Buttons for Charts
        if (chartId) {
            const feedbackDiv = document.createElement('div');
            feedbackDiv.className = 'feedback-buttons';
            feedbackDiv.innerHTML = `
                <button class="btn-feedback" onclick="sendFeedback('${chartId}', 'positive')" title="Approve for report">👍</button>
                <button class="btn-feedback" onclick="sendFeedback('${chartId}', 'negative')" title="Remove chart">👎</button>
            `;
            div.appendChild(feedbackDiv);
        }

        els.messagesDiv.scrollTop = els.messagesDiv.scrollHeight;
    }
    window.addMessage = addMessage;

    async function sendFeedback(chartId, feedback) {
        try {
            const msgDiv = document.querySelector(`[data-chart-id="${chartId}"]`);
            const feedbackBtns = msgDiv?.querySelector('.feedback-buttons');

            if (feedback === 'positive') {
                // Show section selection UI
                if (feedbackBtns) {
                    feedbackBtns.innerHTML = `
                        <div style="display: flex; flex-direction: column; gap: 1rem; padding: 1rem; background: var(--bg-color); border-radius: var(--radius); border: 1px solid var(--border-color); margin-top: 1rem;">
                            <div style="font-weight: 600; color: var(--text-primary);">✅ Add this chart to report</div>
                            <div style="display: flex; gap: 0.5rem; align-items: center;">
                                <label style="font-size: 0.9rem; color: var(--text-secondary);">Section:</label>
                                <select id="section-select-${chartId}" style="flex: 1; padding: 0.5rem; border: 1px solid var(--border-color); border-radius: var(--radius); background: white;">
                                    <option value="">-- Select Section --</option>
                                    <option value="Executive Summary">Executive Summary</option>
                                    <option value="Introduction">Introduction</option>
                                    <option value="Test Setup">Test Setup</option>
                                    <option value="Analysis and Results" selected>Analysis and Results</option>
                                    <option value="Discussion">Discussion</option>
                                    <option value="Conclusion">Conclusion</option>
                                    <option value="Appendices">Appendices</option>
                                </select>
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn-primary" onclick="addChartToReport('${chartId}')" style="flex: 1;">
                                    ➕ Add to Report
                                </button>
                                <button class="btn-outline" onclick="cancelFeedback('${chartId}')" style="padding: 0.5rem 1rem;">
                                    Cancel
                                </button>
                            </div>
                        </div>
                    `;
                }
            } else {
                // Thumbs down - send feedback and mark as rejected
                const resp = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId(),
                        chart_id: chartId,
                        feedback: feedback
                    })
                });
                const data = await resp.json();
                if (data.success) {
                    showToast('Chart marked as not useful', 'info');
                    if (feedbackBtns) {
                        feedbackBtns.innerHTML = '<div style="color: var(--text-muted); font-size: 0.9rem;">👎 Feedback recorded</div>';
                    }
                }
            }
        } catch (e) {
            showToast(`Feedback error: ${e.message}`, 'error');
        }
    }
    window.sendFeedback = sendFeedback;

    async function addChartToReport(chartId) {
        const sectionSelect = document.getElementById(`section-select-${chartId}`);
        const section = sectionSelect?.value;

        if (!section) {
            showToast('Please select a section', 'warning');
            return;
        }

        try {
            const resp = await fetch('/api/report/add_chart', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId(),
                    chart_id: chartId,
                    section: section,
                    chart_path: null,
                    description: null
                })
            });
            const data = await resp.json();
            if (data.success) {
                showToast(`✅ Chart added to "${section}" section`, 'success');
                const msgDiv = document.querySelector(`[data-chart-id="${chartId}"]`);
                const feedbackBtns = msgDiv?.querySelector('.feedback-buttons');
                if (feedbackBtns) {
                    feedbackBtns.innerHTML = `<div style="color: var(--success-color); font-size: 0.9rem;">✅ Added to "${section}"</div>`;
                }
            } else {
                showToast(`Failed to add chart: ${data.error || 'Unknown error'}`, 'error');
            }
        } catch (e) {
            showToast(`Error: ${e.message}`, 'error');
        }
    }
    window.addChartToReport = addChartToReport;

    function cancelFeedback(chartId) {
        const msgDiv = document.querySelector(`[data-chart-id="${chartId}"]`);
        const feedbackBtns = msgDiv?.querySelector('.feedback-buttons');
        if (feedbackBtns) {
            // Restore original feedback buttons
            feedbackBtns.innerHTML = `
                <button class="btn-feedback" onclick="sendFeedback('${chartId}', 'positive')" title="Approve for report">👍</button>
                <button class="btn-feedback" onclick="sendFeedback('${chartId}', 'negative')" title="Remove chart">👎</button>
            `;
        }
    }
    window.cancelFeedback = cancelFeedback;

    // Send chat message
    async function sendMessage() {
        const text = els.chatInput.value.trim();
        if (!text) return;

        addMessage(text, 'user');
        els.chatInput.value = '';
        els.chatInput.style.height = 'auto';

        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId(),
                    message: text
                })
            });
            const data = await resp.json();

            if (data.response) {
                let messageText = data.response;
                if (data.download_ready && data.report_url) {
                    messageText += `\n\n[Download Report](${data.report_url})`;
                }
                addMessage(messageText, 'bot', data.chart_url, data.chart_id, data.plotly_json, data.images);
            }
        } catch (e) {
            addMessage(`Error: ${e.message}`, 'bot');
        }
    }
    window.sendMessage = sendMessage;

    // Start new chat session
    function startNewChat() {
        const newId = `session_${Date.now()}`;
        localStorage.setItem('thermosense_session', newId);
        if (els.sessionInput) els.sessionInput.value = newId;
        els.messagesDiv.innerHTML = '';
        showToast('New chat started', 'info');
        refreshUploads();
        loadSessions();
        addMessage("Hello! I'm ThermoSense. How can I help you today?", 'bot');
    }
    window.startNewChat = startNewChat;

    // Load session history
    async function loadSessions() {
        try {
            const resp = await fetch('/api/sessions');
            const data = await resp.json();
            els.sessionList.innerHTML = '';
            if (data.sessions && data.sessions.length > 0) {
                data.sessions.forEach(s => {
                    const li = document.createElement('li');
                    li.textContent = s.id === sessionId() ? `${s.title || s.id} (current)` : (s.title || s.id);
                    li.onclick = () => switchSession(s.id);
                    els.sessionList.appendChild(li);
                });
            } else {
                els.sessionList.innerHTML = '<li style="color:#888;">No history</li>';
            }
        } catch (e) {
            console.error('Load sessions error:', e);
        }
    }

    // Switch to different session
    function switchSession(id) {
        localStorage.setItem('thermosense_session', id);
        els.sessionInput.value = id;
        location.reload();
    }

    // Load chat history for current session
    async function loadChatHistory() {
        try {
            const resp = await fetch(`/api/chat/history?session_id=${sessionId()}`);
            const data = await resp.json();
            if (data.messages && data.messages.length > 0) {
                data.messages.forEach(msg => {
                    addMessage(msg.content, msg.role === 'user' ? 'user' : 'bot', msg.chart_url, msg.chart_id, msg.plotly_json);
                });
            }
        } catch (e) {
            console.error('Load chat history error:', e);
        }
    }

    // Poll for proactive insights
    async function pollForInsights() {
        console.log('Polling for proactive insights...');
        let attempts = 0;
        const maxAttempts = 10;
        const interval = 3000; // 3 seconds

        const poll = async () => {
            try {
                const resp = await fetch(`/api/analysis/proactive?session_id=${sessionId()}`);
                const data = await resp.json();

                if (data.success && data.insight) {
                    addMessage(data.insight.content, 'bot');
                    return true; // Stop polling
                }
            } catch (e) {
                console.error('Polling error:', e);
            }
            return false;
        };

        const timer = setInterval(async () => {
            attempts++;
            const success = await poll();
            if (success || attempts >= maxAttempts) {
                clearInterval(timer);
            }
        }, interval);
    }
    window.pollForInsights = pollForInsights;

    // Initialize app
    init();
});
