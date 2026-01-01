/**
 * Signal Selector - UI component for selecting signals/parameters after file upload
 * Displays bot recommendations and allows multi-select
 */

class SignalSelector {
    constructor() {
        this.sessionId = getCurrentSessionId();
        this.selectedSignals = new Set();
        this.allSignals = [];
        this.recommendations = [];

        this.setupEventListeners();
    }

    /**
     * Show signal selector modal after file upload
     * Gets signals from ALL uploaded files
     */
    async showAfterUpload(uploadedFiles) {
        try {
            // Get signal recommendations from ALL files
            const response = await fetch('/api/signals/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    // Don't specify filename to get signals from all files
                    filename: null,
                    top_n: 15  // Get top 15 recommendations across all files
                })
            });

            const data = await response.json();

            if (!data.success) {
                console.warn('No signal recommendations:', data.error);
                return this.showManualSelection(uploadedFiles);
            }

            this.recommendations = data.recommendations;

            // Get ALL signals from all files (not just recommendations)
            const allSignals = await this.getAllSignalsFromFiles(uploadedFiles);
            this.allSignals = allSignals;

            // Show modal with recommendations
            this.renderModal();

        } catch (error) {
            console.error('Failed to get signal recommendations:', error);
            this.showManualSelection(uploadedFiles);
        }
    }

    /**
     * Get all signals from all uploaded files
     */
    async getAllSignalsFromFiles(uploadedFiles) {
        try {
            const response = await fetch(`/api/uploads?session_id=${this.sessionId}`);
            const data = await response.json();

            const signalMap = new Map(); // Use Map to track signal -> file mapping

            if (data.files) {
                data.files.forEach(fileInfo => {
                    if (fileInfo.numeric_columns) {
                        fileInfo.numeric_columns.forEach(col => {
                            if (!signalMap.has(col)) {
                                signalMap.set(col, []);
                            }
                            signalMap.set(col, [...signalMap.get(col), fileInfo.filename]);
                        });
                    }
                });
            }

            // Return array of {name, files} objects
            return Array.from(signalMap.entries()).map(([name, files]) => ({
                name,
                files,
                fileCount: files.length
            }));

        } catch (error) {
            console.error('Failed to get all signals:', error);
            return [];
        }
    }

    /**
     * Fallback: Manual selection without recommendations
     */
    async showManualSelection(uploadedFiles) {
        // Get all columns from uploaded files
        const allColumns = await this.getAllColumns(uploadedFiles);
        this.allSignals = allColumns;
        this.recommendations = [];
        this.renderModal();
    }

    /**
     * Get all columns from uploaded files
     */
    async getAllColumns(files) {
        const columns = new Set();

        for (const file of files) {
            try {
                const response = await fetch(`/api/uploads?session_id=${this.sessionId}`);
                const data = await response.json();

                const fileInfo = data.files.find(f => f.filename === file);
                if (fileInfo && fileInfo.numeric_columns) {
                    fileInfo.numeric_columns.forEach(col => columns.add(col));
                }
            } catch (error) {
                console.error('Failed to get columns for', file, error);
            }
        }

        return Array.from(columns);
    }

    /**
     * Render signal selector modal
     */
    renderModal() {
        const modal = document.getElementById('signalSelectorModal');
        if (!modal) {
            this.createModal();
            return this.renderModal();
        }

        const container = document.getElementById('signalList');
        container.innerHTML = '';

        // Display recommendations first (highlighted)
        if (this.recommendations.length > 0) {
            const recHeader = document.createElement('div');
            recHeader.className = 'signal-section-header';
            recHeader.innerHTML = `
                <h3>🎯 Recommended Critical Parameters</h3>
                <p>These signals show high variance or are commonly analyzed in thermal testing</p>
            `;
            container.appendChild(recHeader);

            this.recommendations.forEach((rec, idx) => {
                const signalItem = this.createSignalItem(rec.name, true, rec);
                container.appendChild(signalItem);

                // Auto-select top 5 recommendations
                if (idx < 5) {
                    this.selectedSignals.add(rec.name);
                    signalItem.querySelector('input[type="checkbox"]').checked = true;
                }
            });
        }

        // Display other signals
        const otherSignals = this.allSignals.filter(signalInfo => {
            const signalName = typeof signalInfo === 'string' ? signalInfo : signalInfo.name;
            return !this.recommendations.some(r => r.name === signalName);
        });

        if (otherSignals.length > 0) {
            const otherHeader = document.createElement('div');
            otherHeader.className = 'signal-section-header';
            otherHeader.innerHTML = '<h3>Other Available Signals</h3>';
            container.appendChild(otherHeader);

            otherSignals.forEach(signal => {
                const signalItem = this.createSignalItem(signal, false);
                container.appendChild(signalItem);
            });
        }

        // Update selected count
        this.updateSelectedCount();

        // Show modal
        modal.style.display = 'block';
    }

    /**
     * Create signal item element
     */
    createSignalItem(name, isRecommended, recommendation = null) {
        const item = document.createElement('div');
        item.className = `signal-item ${isRecommended ? 'recommended' : ''}`;

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `signal-${name}`;
        checkbox.value = name;
        checkbox.onchange = (e) => {
            if (e.target.checked) {
                this.selectedSignals.add(name);
            } else {
                this.selectedSignals.delete(name);
            }
            this.updateSelectedCount();
        };

        const label = document.createElement('label');
        label.htmlFor = `signal-${name}`;
        label.innerHTML = `
            <span class="signal-name">${name}</span>
            ${isRecommended ? '<span class="badge-recommended">⭐ Recommended</span>' : ''}
        `;

        if (recommendation) {
            const details = document.createElement('div');
            details.className = 'signal-details';
            details.innerHTML = `
                <small>${recommendation.reason}</small>
                <small class="signal-score">Score: ${(recommendation.score * 100).toFixed(0)}%</small>
            `;
            label.appendChild(details);
        }

        item.appendChild(checkbox);
        item.appendChild(label);

        return item;
    }

    /**
     * Create modal HTML structure
     */
    createModal() {
        const modal = document.createElement('div');
        modal.id = 'signalSelectorModal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content signal-selector-modal">
                <div class="modal-header">
                    <h2>Select Signals to Analyze</h2>
                    <span class="close" onclick="signalSelector.closeModal()">&times;</span>
                </div>
                <div class="modal-body">
                    <div id="signalList"></div>
                </div>
                <div class="modal-footer">
                    <div class="selected-count">
                        <span id="selectedCount">0</span> signals selected
                    </div>
                    <button class="btn-secondary" onclick="signalSelector.selectAll()">Select All</button>
                    <button class="btn-secondary" onclick="signalSelector.clearAll()">Clear All</button>
                    <button class="btn-primary" onclick="signalSelector.confirmSelection()">Continue</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    /**
     * Update selected count display
     */
    updateSelectedCount() {
        const countEl = document.getElementById('selectedCount');
        if (countEl) {
            countEl.textContent = this.selectedSignals.size;
        }
    }

    /**
     * Select all signals
     */
    selectAll() {
        this.allSignals.forEach(signal => {
            const name = typeof signal === 'string' ? signal : signal.name;
            this.selectedSignals.add(name);
        });
        document.querySelectorAll('#signalList input[type="checkbox"]').forEach(cb => {
            cb.checked = true;
        });
        this.updateSelectedCount();
    }

    /**
     * Clear all selections
     */
    clearAll() {
        this.selectedSignals.clear();
        document.querySelectorAll('#signalList input[type="checkbox"]').forEach(cb => {
            cb.checked = false;
        });
        this.updateSelectedCount();
    }

    /**
     * Confirm and save selection
     */
    async confirmSelection() {
        if (this.selectedSignals.size === 0) {
            alert('Please select at least one signal to continue');
            return;
        }

        try {
            // Save selection to backend
            const response = await fetch('/api/signals/select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    signals: Array.from(this.selectedSignals)
                })
            });

            const data = await response.json();

            if (data.success) {
                // Store in local state
                window.selectedSignals = Array.from(this.selectedSignals);

                // Close modal
                this.closeModal();

                // Show success message
                addBotMessage(`✅ ${this.selectedSignals.size} signals selected. You can now create charts with these parameters.`);

            } else {
                throw new Error(data.error || 'Failed to save selection');
            }

        } catch (error) {
            console.error('Failed to save signal selection:', error);
            alert('Failed to save selection: ' + error.message);
        }
    }

    /**
     * Close modal
     */
    closeModal() {
        const modal = document.getElementById('signalSelectorModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Close modal on outside click
        window.addEventListener('click', (event) => {
            const modal = document.getElementById('signalSelectorModal');
            if (event.target === modal) {
                this.closeModal();
            }
        });
    }
}

// Global instance
let signalSelector = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    signalSelector = new SignalSelector();
});

// Helper function to get current session ID
function getCurrentSessionId() {
    return window.currentSessionId || 'default';
}
