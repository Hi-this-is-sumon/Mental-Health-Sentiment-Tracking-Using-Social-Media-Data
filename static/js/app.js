// Mental Health Sentiment Tracker - Enhanced JavaScript

class SentimentTracker {
    constructor() {
        this.init();
        this.bindEvents();
        this.loadTheme();
        this.loadHistory();
        this.isExporting = false;
        this.inputChart = null;
        this.isMobile = window.innerWidth < 768;
        this.setupResponsiveListeners();
        this.optimizeForMobile();
    }

    init() {
        this.form = document.getElementById('analyze-form');
        this.results = document.getElementById('results');
        this.loading = document.getElementById('loading');
        this.progressBar = document.querySelector('.progress-fill');
        this.confidenceFill = document.querySelector('.confidence-fill');
        this.themeToggle = document.querySelector('.theme-toggle');
        this.historyList = document.getElementById('history-list');
        this.exportBtn = document.getElementById('export-btn');
        this.clearHistoryBtn = document.getElementById('clear-history-btn');
    }

    /**
     * Setup responsive event listeners for viewport changes
     */
    setupResponsiveListeners() {
        window.addEventListener('resize', () => {
            const newIsMobile = window.innerWidth < 768;
            if (newIsMobile !== this.isMobile) {
                this.isMobile = newIsMobile;
                // Recreate charts on breakpoint change for optimal responsive layout
                if (this.inputChart) {
                    try { this.inputChart.destroy(); } catch (e) { /* ignore */ }
                    this.inputChart = null;
                    // Re-render chart with new dimensions
                    const canvas = document.getElementById('input-probs-chart');
                    if (canvas && this.lastProbabilities) {
                        this.renderInputProbabilities(this.lastProbabilities);
                    }
                }
            }
        });
    }

    bindEvents() {
        if (this.form) {
            this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        }

        if (this.themeToggle) {
            this.themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        if (this.exportBtn) {
            this.exportBtn.addEventListener('click', (e) => {
                if (this.isExporting) {
                    e.preventDefault();
                    return;
                }
                this.isExporting = true;
                this.exportBtn.disabled = true;
                this.exportBtn.textContent = 'Exporting...';
                this.exportResults();
            });
        }

        if (this.clearHistoryBtn) {
            this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }

    async handleSubmit(e) {
        e.preventDefault();
        const text = document.getElementById('text-input').value.trim();

        if (!text) {
            this.showError('Please enter some text to analyze.');
            return;
        }

        this.showLoading(true);
        this.updateProgress(0);

        try {
            this.updateProgress(25);
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            this.updateProgress(50);

            if (!response.ok) {
                throw new Error('Analysis failed. Please try again.');
            }

            const data = await response.json();
            this.updateProgress(75);

            this.displayResults(data, text);
            this.saveToHistory(data, text);
            this.updateProgress(100);

        } catch (error) {
            console.error('Error:', error);
            this.showError(error.message || 'An error occurred. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(data, text) {
        const emotions = Array.isArray(data.prediction) ? data.prediction : [data.prediction];
        const probabilities = data.probabilities || { [data.prediction]: data.confidence };

        // Display emotions grid
        const emotionsGrid = document.getElementById('emotions-grid');
        if (emotionsGrid) {
            const sortedEmotions = Object.entries(probabilities)
                .sort(([,a], [,b]) => b - a);
            
            emotionsGrid.innerHTML = sortedEmotions
                .map(([emotion, prob]) => `
                    <div class="emotion-card ${emotions.includes(emotion) ? 'detected' : ''}">
                        <div class="emotion-label">${emotion}</div>
                        <div class="emotion-score">${(prob * 100).toFixed(1)}%</div>
                        <div class="emotion-bar">
                            <div class="emotion-bar-fill" style="width: ${prob * 100}%"></div>
                        </div>
                    </div>
                `).join('');
        }

        // Display summary
        document.getElementById('prediction').innerHTML = `
            <h4>Detected Emotions</h4>
            <div class="emotion-tags">
                ${emotions.map(e => `<span class="emotion-tag">${e}</span>`).join(' ')}
            </div>
        `;

        // Display confidence summary if single emotion
        const confidenceElement = document.getElementById('confidence');
        if (emotions.length === 1 && data.confidence) {
            confidenceElement.innerHTML = `
                <div class="confidence-meter">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${(data.confidence * 100)}%"></div>
                    </div>
                    <div>Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
                </div>
            `;
        } else {
            confidenceElement.innerHTML = '';
        }

        // Generate explanation based on emotions
        const explanation = this.generateExplanation(emotions, probabilities);
        document.getElementById('explanation').innerHTML = `
            <h4>Analysis Explanation</h4>
            ${explanation}
        `;

        // Generate suggestions based on emotions
        const suggestion = this.generateSuggestion(emotions, probabilities);
        document.getElementById('suggestion').innerHTML = `
            <h4>Supportive Suggestions</h4>
            ${suggestion}
        `;

        // Load visualizations with loading states
        this.loadVisualization('sentiment-plot', '/plot_sentiment');
        this.loadVisualization('wordcloud-plot', '/wordcloud');

        // Render per-input probability chart (client-side)
        try {
            const probs = data.probabilities || { [data.prediction]: data.confidence };
            this.renderInputProbabilities(probs);
        } catch (e) {
            console.warn('Failed to render input probabilities chart', e);
        }

        this.results.style.display = 'block';
        this.results.scrollIntoView({ behavior: 'smooth' });
    }

    renderInputProbabilities(probabilities) {
        // Store for responsive re-renders
        this.lastProbabilities = probabilities;
        
        // probabilities: { emotion: score }
        const canvas = document.getElementById('input-probs-chart');
        if (!canvas) return;

        const labels = Object.keys(probabilities);
        const data = labels.map(l => Math.round((probabilities[l] || 0) * 1000) / 10); // one decimal

        // Destroy existing chart if present
        if (this.inputChart) {
            try { this.inputChart.destroy(); } catch (e) { /* ignore */ }
            this.inputChart = null;
        }

        // Determine chart orientation based on screen size
        const isSmallScreen = window.innerWidth < 640;
        const chartConfig = {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability (%)',
                    data: data,
                    backgroundColor: labels.map(() => 'rgba(75, 135, 200, 0.9)'),
                    borderColor: labels.map(() => 'rgba(75, 135, 200, 1)'),
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: isSmallScreen ? 'x' : 'y', // Vertical bars on mobile for better space usage
                scales: {
                    x: { beginAtZero: true, max: 100 },
                    y: { ticks: { font: { size: isSmallScreen ? 10 : 12 } } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: true }
                },
                responsive: true,
                maintainAspectRatio: false
            }
        };

        // Create a horizontal bar chart
        const ctx = canvas.getContext('2d');
        this.inputChart = new Chart(ctx, chartConfig);
    }

    async loadVisualization(elementId, url) {
        const img = document.getElementById(elementId);
        const loadingDiv = document.getElementById(elementId.replace('-plot', '-loading'));
        if (img && loadingDiv) {
            console.log(`Loading visualization: ${elementId} from ${url}`);

            // Clear any previous handlers and avoid setting an empty src which
            // can trigger a spurious onerror in some browsers / tracking modes.
            img.removeAttribute('src');
            img.onload = null;
            img.onerror = null;

            // Set up event handlers before setting src
            img.onload = () => {
                console.log(`Visualization loaded successfully: ${elementId}`);
                console.log(`${elementId} natural size:`, img.naturalWidth, img.naturalHeight);
                img.style.display = 'block';
                img.style.opacity = '1';
                loadingDiv.style.display = 'none';
            };
            img.onerror = (e) => {
                console.error(`Failed to load visualization: ${elementId}`, e);
                loadingDiv.innerHTML = '<p style="color: #f5576c;">Failed to load visualization</p>';
            };

            // First try a fast direct static cached file (if server persisted it to static/plots)
            const mapping = {
                'sentiment-plot': 'plot_sentiment.png',
                'wordcloud-plot': 'wordcloud.png'
            };
            const staticFile = mapping[elementId];
            if (staticFile) {
                const staticUrl = '/static/plots/' + staticFile;
                console.log(`Attempting fast load from static cache: ${staticUrl}`);
                try {
                    const c = new AbortController();
                    const t = setTimeout(() => c.abort(), 5000);
                    const headResp = await fetch(staticUrl, { method: 'GET', signal: c.signal });
                    clearTimeout(t);
                    console.log(`Static fetch status for ${elementId}:`, headResp.status);
                    if (headResp.ok) {
                        const blob = await headResp.blob();
                        console.log(`Static blob size for ${elementId}:`, blob.size);
                        if (blob.size && blob.size > 1500) {
                            // Use the fetched blob to avoid serving a stale/broken cached
                            // static file that the browser might have stored. Create an
                            // object URL and assign that to the <img> so content is
                            // exactly what the fetch returned.
                            const objectUrl = URL.createObjectURL(blob);
                            img.onload = () => {
                                console.log(`Visualization loaded successfully from static blob: ${elementId}`);
                                console.log(`${elementId} natural size:`, img.naturalWidth, img.naturalHeight);
                                img.style.display = 'block';
                                img.style.opacity = '1';
                                loadingDiv.style.display = 'none';
                                try { URL.revokeObjectURL(objectUrl); } catch (e) { /* ignore */ }
                            };
                            img.onerror = (e) => {
                                console.error(`Failed to render static blob for ${elementId}`, e);
                                loadingDiv.innerHTML = '<p style="color: #f5576c;">Failed to load visualization</p>';
                                try { URL.revokeObjectURL(objectUrl); } catch (e) { /* ignore */ }
                            };
                            img.src = objectUrl;
                            console.log(`Loaded ${elementId} from static cache (blob)`);
                            return;
                        }
                    }
                } catch (e) {
                    console.warn(`Fast static load failed for ${elementId}:`, e);
                }
            }

            // Do not append a cache-busting timestamp so server-side caching can work.
            const finalUrl = url;
            console.log(`Final URL: ${finalUrl}`);

            // Helper to fetch with retries and increasing timeouts
            const fetchWithRetries = async (attempts = 3) => {
                let attempt = 0;
                let backoff = 1000; // start with 1s
                while (attempt < attempts) {
                    attempt += 1;
                    const controller = new AbortController();
                    const fetchTimeoutMs = 60000 + (attempt - 1) * 10000; // 60s, 70s, 80s
                    const timeout = setTimeout(() => controller.abort(), fetchTimeoutMs);
                    const fetchStart = Date.now();
                    try {
                        const response = await fetch(finalUrl, { method: 'GET', signal: controller.signal });
                        clearTimeout(timeout);
                        console.log(`GET request status for ${elementId} (attempt ${attempt}): ${response.status}`);
                        try {
                            console.log(`Response Content-Type for ${elementId}:`, response.headers.get('Content-Type'));
                        } catch (e) { /* ignore */ }
                        if (!response.ok) throw new Error(`HTTP ${response.status}`);
                        const blob = await response.blob();
                        const elapsed = Date.now() - fetchStart;
                        console.log(`Fetch elapsed for ${elementId} (attempt ${attempt}): ${elapsed}ms`);
                        return blob;
                    } catch (err) {
                        clearTimeout(timeout);
                        console.warn(`Fetch attempt ${attempt} failed for ${elementId}:`, err);
                        if (attempt >= attempts) throw err;
                        // Wait before retrying
                        await new Promise(res => setTimeout(res, backoff));
                        backoff *= 2;
                    }
                }
            };

            // Run the fetch with retries
            fetchWithRetries(3)
                .then(blob => {
                    console.log(`Received blob for ${elementId}:`, blob);
                    // Treat very small blobs as an error response (server returned an error image/message)
                    if (!blob || (typeof blob.size === 'number' && blob.size < 1500)) {
                        console.error(`Visualization blob is unexpectedly small for ${elementId}:`, blob);
                        // Hide any partially-loaded image and show an error message
                        img.style.display = 'none';
                        loadingDiv.style.display = 'block';
                        loadingDiv.innerHTML = '<p style="color: #f5576c;">Visualization failed or returned empty image.</p>';
                        return;
                    }

                    console.log(`Blob size for ${elementId}:`, blob.size);

                    // Create a temporary object URL and assign to image src
                    const objectUrl = URL.createObjectURL(blob);
                    img.onload = () => {
                        // Reuse existing onload handler behavior then revoke URL
                        console.log(`Visualization loaded successfully: ${elementId}`);
                        console.log(`${elementId} natural size:`, img.naturalWidth, img.naturalHeight);
                        img.style.display = 'block';
                        img.style.opacity = '1';
                        loadingDiv.style.display = 'none';
                        try { URL.revokeObjectURL(objectUrl); } catch (e) { /* ignore */ }
                    };
                    img.onerror = (e) => {
                        console.error(`Failed to render visualization blob: ${elementId}`, e);
                        loadingDiv.innerHTML = '<p style="color: #f5576c;">Failed to load visualization</p>';
                        try { URL.revokeObjectURL(objectUrl); } catch (e) { /* ignore */ }
                    };
                    img.src = objectUrl;
                })
                .catch(error => {
                    console.error(`Failed to fetch ${elementId}:`, error);
                    if (error.name === 'AbortError') {
                        loadingDiv.innerHTML = '<p style="color: #f5576c;">Visualization timed out (server slow)</p>';
                    } else {
                        loadingDiv.innerHTML = '<p style="color: #f5576c;">Failed to load visualization</p>';
                    }
                });
            // Fallback: update the loading placeholder after 20 seconds to inform the user
            setTimeout(() => {
                if (loadingDiv.style.display !== 'none') {
                    console.warn(`Visualization taking longer than expected: ${elementId}`);
                    loadingDiv.innerHTML = '<p style="color: #f5576c;">Visualization taking longer than expected... still loading</p>';
                }
            }, 20000);
        }
    }

    showLoading(show) {
        if (this.loading) {
            this.loading.style.display = show ? 'block' : 'none';
        }
        if (this.form) {
            const submitBtn = this.form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = show;
                submitBtn.innerHTML = show ? '<span class="loading"></span> Analyzing...' : 'Analyze My Emotions';
            }
        }
    }

    updateProgress(percent) {
        if (this.progressBar) {
            this.progressBar.style.width = `${percent}%`;
        }
    }

    showError(message) {
        // Create error toast
        const toast = document.createElement('div');
        toast.className = 'error-toast';
        toast.innerHTML = `
            <div style="background: #f5576c; color: white; padding: 15px; border-radius: 10px; margin: 20px 0; box-shadow: var(--shadow);">
                <strong>Error:</strong> ${message}
            </div>
        `;

        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(toast, container.firstChild);
            setTimeout(() => {
                toast.remove();
            }, 5000);
        }
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    }

    loadTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
    }

    generateExplanation(emotions, probabilities) {
        if (emotions.length === 0) {
            return '<p>No clear emotions were detected in your text.</p>';
        }

        if (emotions.length === 1) {
            return `<p>Your message primarily expresses <strong>${emotions[0]}</strong>. This analysis is based on linguistic patterns and emotional markers in your text.</p>`;
        }

        // Sort emotions by probability for multi-emotion explanation
        const sortedEmotions = Object.entries(probabilities)
            .filter(([emotion]) => emotions.includes(emotion))
            .sort(([,a], [,b]) => b - a);

        const primary = sortedEmotions[0][0];
        const secondary = sortedEmotions[1] ? sortedEmotions[1][0] : null;

        let explanation = `<p>Your message shows a complex emotional state, primarily expressing <strong>${primary}</strong>`;
        if (secondary) {
            explanation += ` combined with <strong>${secondary}</strong>`;
        }
        explanation += '. This combination suggests:</p>';

        explanation += '<ul class="emotion-analysis">';
        sortedEmotions.forEach(([emotion, prob]) => {
            const percentage = (prob * 100).toFixed(1);
            explanation += `<li><strong>${emotion}</strong> (${percentage}%) is detected in your expression</li>`;
        });
        explanation += '</ul>';

        return explanation;
    }

    generateSuggestion(emotions, probabilities) {
        const suggestions = {
            anger: "Take a moment to pause and breathe. It's okay to feel angry, but try to identify what's triggering these feelings.",
            anxiety: "Practice grounding techniques: focus on your breath and name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
            depression: "Your feelings are valid. Consider reaching out to someone you trust or a mental health professional. Small self-care acts can help too.",
            fear: "Remember that your fears are valid, but they don't define you. Try to stay present and focus on what you can control.",
            frustration: "Take a step back and break down what's bothering you into smaller, manageable parts.",
            hope: "Hold onto this positive feeling. You're showing resilience and strength in your outlook.",
            joy: "Celebrate this positive moment! Consider sharing your happiness with others who might need it.",
            loneliness: "Remember that feeling lonely is a common human experience. Consider reaching out to friends, family, or joining community activities.",
            sadness: "Be gentle with yourself during this time. It's okay to feel sad, and it's also okay to seek support.",
            stress: "Try taking short breaks and practice self-care. Even small moments of relaxation can help manage stress.",
            worry: "Focus on what you can control. Making a list of actionable steps might help reduce overwhelming thoughts."
        };

        // Get the primary emotion (highest probability)
        const sortedEmotions = Object.entries(probabilities)
            .filter(([emotion]) => emotions.includes(emotion))
            .sort(([,a], [,b]) => b - a);
        
        let suggestionText = '<p>';
        
        if (sortedEmotions.length > 0) {
            const primaryEmotion = sortedEmotions[0][0].toLowerCase();
            suggestionText += suggestions[primaryEmotion] || "Take a moment to acknowledge and sit with your feelings. They're all valid parts of your experience.";
        }

        if (sortedEmotions.length > 1) {
            suggestionText += '</p><p>With multiple emotions present, consider:</p><ul>';
            suggestionText += '<li>Writing down your thoughts to better understand their interplay</li>';
            suggestionText += '<li>Taking time to process each feeling individually</li>';
            suggestionText += '<li>Being patient with yourself as you navigate complex emotions</li>';
            suggestionText += '</ul>';
        } else {
            suggestionText += '</p>';
        }

        return suggestionText;
    }

    saveToHistory(data, text) {
        const history = this.getHistory();
        const entry = {
            id: Date.now(),
            text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
            prediction: Array.isArray(data.prediction) ? data.prediction : [data.prediction],
            probabilities: data.probabilities || { [data.prediction]: data.confidence },
            timestamp: new Date().toISOString()
        };

        history.unshift(entry);
        // Keep only last 10 entries
        if (history.length > 10) {
            history.splice(10);
        }

        localStorage.setItem('analysisHistory', JSON.stringify(history));
        this.updateHistoryDisplay();
    }

    getHistory() {
        const history = localStorage.getItem('analysisHistory');
        return history ? JSON.parse(history) : [];
    }

    updateHistoryDisplay() {
        if (!this.historyList) return;

        const history = this.getHistory();
        this.historyList.innerHTML = '';

        if (history.length === 0) {
            this.historyList.innerHTML = '<li class="history-item">No analysis history yet.</li>';
            return;
        }

        history.forEach(entry => {
            const li = document.createElement('li');
            li.className = 'history-item';

            // Get top 3 emotions by probability
            const topEmotions = Object.entries(entry.probabilities || {})
                .sort(([,a], [,b]) => b - a)
                .slice(0, 3)
                .map(([emotion, prob]) => `${emotion} (${(prob * 100).toFixed(1)}%)`)
                .join(', ');

            li.innerHTML = `
                <div class="history-timestamp">${new Date(entry.timestamp).toLocaleString()}</div>
                <div><strong>Text:</strong> ${entry.text}</div>
                <div><strong>Detected Emotions:</strong> ${topEmotions}</div>
            `;
            this.historyList.appendChild(li);
        });
    }

    loadHistory() {
        this.updateHistoryDisplay();
    }

    async exportResults() {
        const history = this.getHistory();
        if (history.length === 0) {
            alert('No results to export.');
            return;
        }

        try {
            if (history.length > 1) {
                // Multiple entries - offer combined PDF or CSV
                const exportType = confirm('Export all history as single PDF? (Cancel for CSV)') ? 'pdf_all' : 'csv';
                if (exportType === 'pdf_all') {
                    await this.exportAllHistoryAsPDF();
                } else {
                    await this.exportAsCSV();
                }
            } else {
                // Single entry - ask for export type
                const exportType = confirm('Export as PDF? (Cancel for CSV)') ? 'pdf' : 'csv';
                if (exportType === 'pdf') {
                    await this.exportAsPDF();
                } else {
                    await this.exportAsCSV();
                }
            }
        } finally {
            this.isExporting = false;
            if (this.exportBtn) {
                this.exportBtn.disabled = false;
                this.exportBtn.textContent = 'Export Results';
            }
        }
    }

    async exportAsCSV() {
        // Prefer server-generated CSV so we include cleaned text and consistent columns
        const history = this.getHistory();
        try {
            const resp = await fetch('/export_history_csv', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ history: history })
            });
            if (!resp.ok) throw new Error('CSV export failed');
            const blob = await resp.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'analysis_history.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (e) {
            console.error('CSV export failed, falling back to client CSV', e);
            // Fallback to client-side CSV generation (previous behavior)
            const allEmotions = new Set();
            history.forEach(entry => Object.keys(entry.probabilities || {}).forEach(emotion => allEmotions.add(emotion)));
            const emotionHeaders = Array.from(allEmotions).sort();
            const csvContent = 'data:text/csv;charset=utf-8,'
                + 'Timestamp,Text,' + emotionHeaders.map(e => `"${e}"`).join(',') + '\n'
                + history.map(entry => {
                    const emotions = entry.probabilities || {};
                    const emotionValues = emotionHeaders
                        .map(emotion => emotions[emotion] ? (emotions[emotion] * 100).toFixed(1) + '%' : '0%');
                    return `"${entry.timestamp}","${entry.text.replace(/"/g, '""')}",${emotionValues.join(',')}`;
                }).join('\n');
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement('a');
            link.setAttribute('href', encodedUri);
            link.setAttribute('download', 'emotion_analysis_history.csv');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    async exportAsPDF() {
        // Get the latest analysis result
        const latestResult = this.getLatestResult();
        if (!latestResult) {
            alert('No analysis result to export. Please analyze some text first.');
            return;
        }

        try {
            const response = await fetch('/export_pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(latestResult),
            });

            if (!response.ok) {
                throw new Error('Failed to generate PDF');
            }

            // Download the PDF
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'emotion_analysis_result.pdf';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Error exporting PDF:', error);
            alert('Failed to export PDF. Please try again.');
        }
    }

    async exportAllHistoryAsPDF() {
        const history = this.getHistory();
        if (history.length === 0) {
            alert('No history to export.');
            return;
        }

        try {
            const response = await fetch('/export_history_pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ history: history }),
            });

            if (!response.ok) {
                throw new Error('Failed to generate history PDF');
            }

            // Download the PDF
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'emotion_analysis_history.pdf';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Error exporting history PDF:', error);
            alert('Failed to export history PDF. Please try again.');
        }
    }

    getLatestResult() {
        // Get the most recent analysis from localStorage or current session
        const history = this.getHistory();
        if (history.length > 0) {
            const latest = history[0];  // Get the first item since we unshift new entries
            return {
                text: latest.text,
                prediction: latest.prediction,
                probabilities: latest.probabilities
            };
        }

        // Check if there's current analysis data in the DOM
        const predictionElement = document.getElementById('prediction');
        const emotionsGrid = document.getElementById('emotions-grid');
        const textInput = document.getElementById('text-input');

        if (predictionElement && emotionsGrid && textInput) {
            const emotionTags = predictionElement.querySelectorAll('.emotion-tag');
            const emotions = Array.from(emotionTags).map(tag => tag.textContent);
            
            const probabilities = {};
            emotionsGrid.querySelectorAll('.emotion-card').forEach(card => {
                const emotion = card.querySelector('.emotion-label').textContent;
                const scoreText = card.querySelector('.emotion-score').textContent;
                probabilities[emotion] = parseFloat(scoreText) / 100;
            });

            return {
                text: textInput.value,
                prediction: emotions,
                probabilities: probabilities
            };
        }

        return null;
    }

    clearHistory() {
        if (confirm('Are you sure you want to clear all analysis history?')) {
            localStorage.removeItem('analysisHistory');
            this.updateHistoryDisplay();
        }
    }

    handleKeyboard(e) {
        // Ctrl/Cmd + K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const input = document.getElementById('text-input');
            if (input) input.focus();
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal');
            modals.forEach(modal => modal.style.display = 'none');
        }

        // Enter to submit form on Ctrl/Cmd + Enter (mobile friendly)
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (this.form) {
                this.form.dispatchEvent(new Event('submit'));
            }
        }
    }

    /**
     * Optimize display for mobile screens
     */
    optimizeForMobile() {
        if (window.innerWidth < 768) {
            // Adjust form inputs for better mobile interaction
            const inputs = document.querySelectorAll('input, textarea, select');
            inputs.forEach(input => {
                input.setAttribute('autocomplete', 'off');
                input.addEventListener('focus', () => {
                    // Scroll into view on mobile focus
                    setTimeout(() => {
                        input.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }, 300);
                });
            });
        }
    }
}

// Quotes rotation for landing page
class QuoteRotator {
    constructor() {
        this.quotes = [
            "You are not alone in this journey. Your feelings are valid.",
            "Every storm runs out of rain. This too shall pass.",
            "Your mental health is a priority. Your happiness is essential.",
            "Be kind to yourself. You're doing the best you can.",
            "Your emotions are important. Take the time to understand them.",
            "Small steps can lead to big changes in how you feel."
        ];
        this.currentQuote = 0;
        this.quoteElement = document.getElementById('quote');
        this.init();
    }

    init() {
        if (this.quoteElement) {
            this.rotateQuote();
            setInterval(() => this.rotateQuote(), 6000);
        }
    }

    rotateQuote() {
        if (this.quoteElement) {
            this.quoteElement.style.opacity = 0;
            setTimeout(() => {
                this.quoteElement.textContent = this.quotes[this.currentQuote];
                this.quoteElement.style.opacity = 1;
                this.currentQuote = (this.currentQuote + 1) % this.quotes.length;
            }, 500);
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SentimentTracker();
    new QuoteRotator();
});
