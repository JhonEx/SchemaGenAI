// Global variables
let currentData = {};
let currentTables = [];
let selectedTable = '';
let chatEnabled = false;

// DOM elements
const navTabs = document.querySelectorAll('.nav-tab');
const tabContents = document.querySelectorAll('.tab-content');
const schemaUpload = document.getElementById('schema-upload');
const uploadStatus = document.getElementById('upload-status');
const generateBtn = document.getElementById('generate-btn');
const promptText = document.getElementById('prompt-text');
const temperatureSlider = document.getElementById('temperature');
const tempValue = document.getElementById('temp-value');
const maxTokens = document.getElementById('max-tokens');
const dataPreview = document.getElementById('data-preview');
const tableSelector = document.getElementById('table-selector');
const dataTable = document.getElementById('data-table');
const editInstructions = document.getElementById('edit-instructions');
const submitEdit = document.getElementById('submit-edit');
const downloadBtn = document.getElementById('download-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const chatInput = document.getElementById('chat-input');
const sendChat = document.getElementById('send-chat');
const chatMessages = document.getElementById('chat-messages');
const chatStatus = document.querySelector('.chat-status');
// In your existing success handler for /generate_data:
document.getElementById('save-db-btn').disabled = false;



// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing Data Assistant...');
    initializeEventListeners();
    updateTemperatureDisplay();
    checkForExistingData();
    initializeCollapsibleSections();
});

document.getElementById('save-db-btn').addEventListener('click', async () => {
  const btn = document.getElementById('save-db-btn');
  const status = document.getElementById('save-db-status');
  status.style.display = 'block';
  status.textContent = 'Saving data to database...';
  btn.disabled = true;

  try {
    const res = await fetch('/save_data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const payload = await res.json();
    if (!payload.success) throw new Error(payload.error || 'Failed to save data');

    const saved = payload.saved || {};
    const lines = Object.entries(saved).map(([t, n]) => `${t}: ${n} rows`);
    status.textContent = `✅ Saved to DB:\n${lines.join('\n')}`;
  } catch (err) {
    status.textContent = `❌ ${err.message}`;
  } finally {
    btn.disabled = false;
  }
});


function initializeEventListeners() {
    console.log('📝 Setting up event listeners...');

    // Tab navigation
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Schema upload
    if (schemaUpload) {
        schemaUpload.addEventListener('change', handleSchemaUpload);
        console.log('✅ Schema upload listener added');
    }

    // Generate button
    if (generateBtn) {
        generateBtn.addEventListener('click', generateData);
        console.log('✅ Generate button listener added');
    }

    // Temperature slider
    if (temperatureSlider) {
        temperatureSlider.addEventListener('input', updateTemperatureDisplay);
        console.log('✅ Temperature slider listener added');
    }

    // Table selector
    if (tableSelector) {
        tableSelector.addEventListener('change', handleTableSelection);
        console.log('✅ Table selector listener added');
    }

    // Edit submission
    if (submitEdit) {
        submitEdit.addEventListener('click', submitTableEdit);
        console.log('✅ Edit submit listener added');
    }

    // Download button
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadData);
        console.log('✅ Download button listener added');
    }

    // Chat functionality
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
        console.log('✅ Chat input listener added');
    }

    if (sendChat) {
        sendChat.addEventListener('click', sendChatMessage);
        console.log('✅ Send chat listener added');
    }
}

function initializeCollapsibleSections() {
    const collapsibleBtns = document.querySelectorAll('.collapsible-btn');
    collapsibleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const expanded = btn.getAttribute('aria-expanded') === 'true';
            btn.setAttribute('aria-expanded', !expanded);
            const content = btn.nextElementSibling;
            if (content) {
                content.style.maxHeight = expanded ? 0 : `${content.scrollHeight}px`;
            }
        });
    });
}

function switchTab(tabId) {
    console.log(`🔄 Switching to tab: ${tabId}`);

    // Update nav tabs
    navTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabId);
    });

    // Update tab contents
    tabContents.forEach(content => {
        content.classList.toggle('active', content.id === tabId);
    });
}

function updateTemperatureDisplay() {
    if (tempValue && temperatureSlider) {
        tempValue.textContent = temperatureSlider.value;
    }
}

async function checkForExistingData() {
  try {
    const res = await fetch('/session_status');
    const s = await res.json();
    if (!s.success) return;

    if (s.has_data) {
      // hydrate globals the UI expects
      currentTables = s.tables || [];
      currentData.generatedData = currentData.generatedData || {};
      // you can lazy-load rows per table later; for now just enable chat
      enableChatInterface();
      // populate the selector right away
      populateTableSelector(currentTables);
      // reveal the preview section
      if (dataPreview) dataPreview.style.display = 'block';
      showSuccessMessage(`Restored session. ${currentTables.length} tables ready.`);
    } else if (s.has_schema) {
      // schema uploaded earlier—make sure Generate button is enabled
      if (generateBtn) generateBtn.disabled = false;
    }
  } catch (e) {
    console.warn('session_status check failed', e);
  }
}

async function handleSchemaUpload(event) {
    console.log('📤 Starting schema upload...');
    const file = event.target.files[0];
    if (!file) {
        console.log('❌ No file selected');
        return;
    }

    console.log(`📁 File selected: ${file.name} (${file.size} bytes)`);
    showLoading('Uploading and parsing schema...');

    const formData = new FormData();
    formData.append('schema_file', file);

    try {
        console.log('🌐 Sending schema to server...');
        const response = await fetch('/upload_schema', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('📥 Server response:', result);

        if (result.success) {
            const tableCount = Object.keys(result.schema_info).length;
            showUploadStatus(
                `Schema uploaded successfully! Found ${tableCount} tables.`,
                'success'
            );
            generateBtn.disabled = false;

            // Store schema info for reference
            currentData.schemaInfo = result.schema_info;

            // Display schema info
            displaySchemaInfo(result.schema_info);

            console.log('✅ Schema upload successful');
        } else {
            console.error('❌ Schema upload failed:', result.error);
            showUploadStatus(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('❌ Network error during schema upload:', error);
        showUploadStatus(`Error uploading file: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function displaySchemaInfo(schemaInfo) {
    console.log('📊 Displaying schema info:', schemaInfo);
    const tableNames = Object.keys(schemaInfo);
    const statusDiv = document.getElementById('upload-status');

    let infoHTML = `<div class="schema-info">`;
    infoHTML += `<h4><i class="fas fa-database"></i> Schema loaded successfully!</h4>`;
    infoHTML += `<p><strong>Tables found:</strong> ${tableNames.length}</p>`;

    tableNames.forEach(tableName => {
        const columns = schemaInfo[tableName].columns || [];
        infoHTML += `<div class="table-info">`;
        infoHTML += `<h5><i class="fas fa-table"></i> ${tableName}</h5>`;
        infoHTML += `<p><strong>Columns (${columns.length}):</strong> ${columns.map(col => `${col.name} (${col.type})`).join(', ')}</p>`;
        infoHTML += `</div>`;
    });

    infoHTML += `</div>`;
    statusDiv.innerHTML = infoHTML;
    statusDiv.className = 'upload-status success';
}

async function generateData() {
    console.log('🎲 Starting data generation...');

    if (!currentData.schemaInfo) {
        showErrorMessage('Please upload a schema first.');
        return;
    }

    const instructions = promptText ? promptText.value.trim() : '';
    const temperature = temperatureSlider ? parseFloat(temperatureSlider.value) : 0.7;
    const numRows = maxTokens ? parseInt(maxTokens.value) : 5;

    console.log('⚙️ Generation parameters:', { instructions, temperature, numRows });

    showLoading('Generating synthetic data with AI...');
    generateBtn.classList.add('btn-loading');
    generateBtn.disabled = true;

    try {
        console.log('🌐 Sending generation request...');
        const response = await fetch('/generate_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                instructions,
                temperature,
                num_rows: numRows
            })
        });

        const result = await response.json();
        console.log('📥 Generation response:', result);

        if (result.success) {
            currentData.generatedData = result.data;
            currentTables = result.tables;

            console.log(`✅ Generated data for ${result.tables.length} tables`);

            // Populate table selector
            populateTableSelector(result.tables);

            // Show data preview section
            if (dataPreview) {
                dataPreview.style.display = 'block';
                dataPreview.scrollIntoView({ behavior: 'smooth' });
            }

            // Enable chat functionality
            enableChatInterface();

            showSuccessMessage(`Data generated successfully! Created ${result.tables.length} tables with synthetic data.`);
        } else {
            console.error('❌ Data generation failed:', result.error);
            showErrorMessage(`Error generating data: ${result.error}`);
        }
    } catch (error) {
        console.error('❌ Network error during generation:', error);
        showErrorMessage(`Error: ${error.message}`);
    } finally {
        hideLoading();
        generateBtn.classList.remove('btn-loading');
        generateBtn.disabled = false;
    }
}

function populateTableSelector(tables) {
    console.log('📋 Populating table selector with:', tables);

    if (!tableSelector) return;

    tableSelector.innerHTML = '<option value="">Select a table...</option>';
    tables.forEach(tableName => {
        const option = document.createElement('option');
        option.value = tableName;
        const rowCount = currentData.generatedData[tableName]?.length || 0;
        option.textContent = `${tableName} (${rowCount} rows)`;
        tableSelector.appendChild(option);
    });
}

async function handleTableSelection() {
    const tableName = tableSelector ? tableSelector.value : '';
    console.log(`🔄 Table selected: ${tableName}`);

    if (!tableName) {
        clearDataTable();
        return;
    }

    selectedTable = tableName;
    showLoading('Loading table data...');

    try {
        console.log(`🌐 Fetching data for table: ${tableName}`);
        const response = await fetch(`/get_table_data/${tableName}`);
        const result = await response.json();

        if (result.success) {
            console.log(`✅ Loaded ${result.count} rows for table ${tableName}`);
            displayTableData(result.data, tableName);
        } else {
            console.error('❌ Failed to load table data:', result.error);
            showErrorMessage(`Error loading table data: ${result.error}`);
        }
    } catch (error) {
        console.error('❌ Network error loading table:', error);
        showErrorMessage(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function displayTableData(data, tableName) {
    console.log(`📊 Displaying ${data.length} rows for table ${tableName}`);

    if (!data || data.length === 0) {
        clearDataTable();
        showInfoMessage('No data found for this table.');
        return;
    }

    if (!dataTable) {
        console.error('❌ Data table element not found');
        return;
    }

    // Clear existing table
    const thead = dataTable.querySelector('thead');
    const tbody = dataTable.querySelector('tbody');
    if (thead) thead.innerHTML = '';
    if (tbody) tbody.innerHTML = '';

    // Create header
    const headerRow = document.createElement('tr');
    const columns = Object.keys(data[0]);
    columns.forEach(column => {
        const th = document.createElement('th');
        th.textContent = column;
        th.title = `Column: ${column}`;
        headerRow.appendChild(th);
    });
    if (thead) thead.appendChild(headerRow);

    // Create body rows (limit to first 100 rows for performance)
    const displayData = data.slice(0, 100);
    displayData.forEach((row, index) => {
        const tr = document.createElement('tr');
        tr.title = `Row ${index + 1}`;
        columns.forEach(column => {
            const td = document.createElement('td');
            const value = row[column];

            // Format different data types appropriately
            if (value === null || value === undefined) {
                td.textContent = 'NULL';
                td.className = 'null-value';
            } else if (typeof value === 'object') {
                td.textContent = JSON.stringify(value);
            } else if (typeof value === 'boolean') {
                td.textContent = value ? 'TRUE' : 'FALSE';
                td.className = value ? 'boolean-true' : 'boolean-false';
            } else {
                td.textContent = value.toString();
            }

            tr.appendChild(td);
        });
        if (tbody) tbody.appendChild(tr);
    });

    // Add table stats
    addTableStats(data.length, displayData.length, tableName, columns.length);
}

function addTableStats(totalRows, displayedRows, tableName, columnCount) {
    // Remove existing stats
    const existingStats = document.querySelector('.table-stats');
    if (existingStats) {
        existingStats.remove();
    }

    const statsDiv = document.createElement('div');
    statsDiv.className = 'table-stats';
    statsDiv.innerHTML = `
        <div class="stat-item">
            <i class="fas fa-table"></i>
            <span>Table: <strong>${tableName}</strong></span>
        </div>
        <div class="stat-item">
            <i class="fas fa-database"></i>
            <span>Total Records: <span class="record-count">${totalRows.toLocaleString()}</span></span>
        </div>
        <div class="stat-item">
            <i class="fas fa-eye"></i>
            <span>Displayed: <span class="record-count">${displayedRows.toLocaleString()}</span></span>
        </div>
        <div class="stat-item">
            <i class="fas fa-columns"></i>
            <span>Columns: <span class="record-count">${columnCount}</span></span>
        </div>
    `;

    const tableContainer = document.getElementById('table-container');
    if (tableContainer) {
        tableContainer.insertBefore(statsDiv, tableContainer.firstChild);
    }
}

function clearDataTable() {
    if (!dataTable) return;

    const thead = dataTable.querySelector('thead');
    const tbody = dataTable.querySelector('tbody');
    if (thead) thead.innerHTML = '';
    if (tbody) tbody.innerHTML = '';

    // Remove table stats
    const existingStats = document.querySelector('.table-stats');
    if (existingStats) {
        existingStats.remove();
    }
}

async function submitTableEdit() {
    console.log('✏️ Submitting table edit...');

    if (!selectedTable || !editInstructions || !editInstructions.value.trim()) {
        showErrorMessage('Please select a table and enter edit instructions.');
        return;
    }

    const instructions = editInstructions.value.trim();
    const temperature = temperatureSlider ? parseFloat(temperatureSlider.value) : 0.7;

    console.log('⚙️ Edit parameters:', { selectedTable, instructions, temperature });

    showLoading('Modifying data with AI...');
    submitEdit.classList.add('btn-loading');
    submitEdit.disabled = true;

    try {
        console.log('🌐 Sending modification request...');
        const response = await fetch('/modify_table_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                table_name: selectedTable,
                instructions,
                temperature
            })
        });

        const result = await response.json();
        console.log('📥 Modification response:', result);

        if (result.success) {
            // Update current data
            currentData.generatedData[selectedTable] = result.data;

            // Refresh table display
            displayTableData(result.data, selectedTable);

            // Clear edit instructions
            editInstructions.value = '';

            console.log(`✅ Modified table ${selectedTable} - ${result.count} rows`);
            showSuccessMessage(`Table "${selectedTable}" modified successfully! Updated ${result.count} rows.`);
        } else {
            console.error('❌ Table modification failed:', result.error);
            showErrorMessage(`Error modifying data: ${result.error}`);
        }
    } catch (error) {
        console.error('❌ Network error during modification:', error);
        showErrorMessage(`Error: ${error.message}`);
    } finally {
        hideLoading();
        submitEdit.classList.remove('btn-loading');
        submitEdit.disabled = false;
    }
}

async function downloadData() {
    console.log('💾 Starting data download...');

    if (!currentTables.length) {
        showErrorMessage('No data available to download.');
        return;
    }

    showLoading('Preparing download...');
    downloadBtn.classList.add('btn-loading');
    downloadBtn.disabled = true;

    try {
        console.log('🌐 Requesting data download...');
        const response = await fetch('/download_data');

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `synthetic_data_${new Date().getTime()}.zip`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            console.log('✅ Data download completed');
            showSuccessMessage('Data downloaded successfully!');
        } else {
            const errorData = await response.json();
            console.error('❌ Download failed:', errorData);
            showErrorMessage(`Error downloading data: ${errorData.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('❌ Network error during download:', error);
        showErrorMessage(`Error: ${error.message}`);
    } finally {
        hideLoading();
        downloadBtn.classList.remove('btn-loading');
        downloadBtn.disabled = false;
    }
}

function enableChatInterface() {
    console.log('💬 Enabling chat interface...');

    chatEnabled = true;
    if (chatInput) chatInput.disabled = false;
    if (sendChat) sendChat.disabled = false;

    if (chatStatus) {
        chatStatus.innerHTML = `
            <p><i class="fas fa-check-circle text-success"></i> Chat enabled! Ask questions about your ${currentTables.length} generated tables.</p>
        `;
    }

    // Add welcome message to chat
    addChatMessage(
        `Great! I've analyzed your generated data. You now have ${currentTables.length} tables: ${currentTables.join(', ')}. What would you like to know about your data?`,
        'bot'
    );
}

async function sendChatMessage() {
    if (!chatInput) return;

    const message = chatInput.value.trim();
    if (!message || !chatEnabled) return;

    console.log('💬 Sending chat message:', message);

    // Add user message to chat
    addChatMessage(message, 'user');

    // Clear input
    chatInput.value = '';

    // Disable input while processing
    chatInput.disabled = true;
    if (sendChat) {
        sendChat.disabled = true;
        sendChat.classList.add('btn-loading');
    }

    // Show typing indicator
    const typingId = addTypingIndicator();

    try {
        console.log('🌐 Sending chat request to server...');
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message
            })
        });

        const result = await response.json();
        console.log('📥 Chat response:', result);

        // Remove typing indicator
        removeTypingIndicator(typingId);

        if (result.success) {
            addChatMessage(result.response, 'bot');
            console.log('✅ Chat response added');
        } else {
            console.error('❌ Chat request failed:', result.error);
            addChatMessage('Sorry, I encountered an error processing your message. Please try again.', 'bot');
        }

    } catch (error) {
        console.error('❌ Network error in chat:', error);
        removeTypingIndicator(typingId);
        addChatMessage('Sorry, I encountered a network error. Please check your connection and try again.', 'bot');
    } finally {
        // Re-enable input
        chatInput.disabled = false;
        if (sendChat) {
            sendChat.disabled = false;
            sendChat.classList.remove('btn-loading');
        }
        chatInput.focus();
    }
}

function addChatMessage(message, type) {
    if (!chatMessages) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `${type}-message`;

    const timestamp = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

    if (type === 'user') {
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <i class="fas fa-user"></i>
                    <span class="message-time">${timestamp}</span>
                </div>
                <p>${escapeHtml(message)}</p>
            </div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <i class="fas fa-robot"></i>
                    <span class="message-time">${timestamp}</span>
                </div>
                <div class="message-text">${formatBotMessage(message)}</div>
            </div>
        `;
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Animate in
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(20px)';
    requestAnimationFrame(() => {
        messageDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
    });
}

function addTypingIndicator() {
    if (!chatMessages) return null;

    const typingId = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = typingId;
    typingDiv.className = 'bot-message typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="message-header">
                <i class="fas fa-robot"></i>
                <span class="message-time">typing...</span>
            </div>
            <div class="typing-animation">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return typingId;
}

function removeTypingIndicator(typingId) {
    if (typingId) {
        const typingDiv = document.getElementById(typingId);
        if (typingDiv) {
            typingDiv.remove();
        }
    }
}

function formatBotMessage(message) {
    // Convert line breaks to paragraphs
    const paragraphs = message.split('\n\n').filter(p => p.trim());

    let formattedMessage = '';
    paragraphs.forEach(paragraph => {
        // Handle lists
        if (paragraph.includes('\n-') || paragraph.includes('\n•')) {
            const lines = paragraph.split('\n');
            const heading = lines[0];
            const listItems = lines.slice(1).filter(line => line.trim().startsWith('-') || line.trim().startsWith('•'));

            if (heading && listItems.length > 0) {
                formattedMessage += `<p><strong>${escapeHtml(heading)}</strong></p><ul>`;
                listItems.forEach(item => {
                    const cleanItem = item.replace(/^[−•-]\s*/, '').trim();
                    formattedMessage += `<li>${escapeHtml(cleanItem)}</li>`;
                });
                formattedMessage += '</ul>';
            } else {
                formattedMessage += `<p>${escapeHtml(paragraph)}</p>`;
            }
        } else {
            // Regular paragraph
            formattedMessage += `<p>${escapeHtml(paragraph)}</p>`;
        }
    });

    return formattedMessage || `<p>${escapeHtml(message)}</p>`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Utility functions
function showLoading(message = 'Loading...') {
    if (loadingOverlay) {
        const spinner = loadingOverlay.querySelector('.loading-spinner p');
        if (spinner) spinner.textContent = message;
        loadingOverlay.style.display = 'flex';
    }
}

function hideLoading() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

function showUploadStatus(message, type) {
    if (uploadStatus) {
        uploadStatus.innerHTML = message;
        uploadStatus.className = `upload-status ${type}`;
        uploadStatus.style.display = 'block';
    }
}

function showSuccessMessage(message) {
    showNotification(message, 'success');
}

function showErrorMessage(message) {
    showNotification(message, 'error');
}

function showInfoMessage(message) {
    showNotification(message, 'info');
}

function showNotification(message, type) {
    console.log(`📢 ${type.toUpperCase()}: ${message}`);

    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => notification.remove());

    const notification = document.createElement('div');
    notification.className = `notification ${type}-message`;

    let icon = 'fas fa-info-circle';
    if (type === 'success') icon = 'fas fa-check-circle';
    else if (type === 'error') icon = 'fas fa-exclamation-circle';
    else if (type === 'warning') icon = 'fas fa-exclamation-triangle';

    notification.innerHTML = `
        <i class="${icon}"></i>
        ${message}
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;

    // Position the notification
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '10000';
    notification.style.maxWidth = '400px';
    notification.style.padding = '1rem 1.5rem';
    notification.style.borderRadius = '12px';
    notification.style.boxShadow = '0 4px 20px rgba(0,0,0,0.15)';
    notification.style.transform = 'translateX(100%)';
    notification.style.transition = 'transform 0.3s ease';

    document.body.appendChild(notification);

    // Animate in
    requestAnimationFrame(() => {
        notification.style.transform = 'translateX(0)';
    });

    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 300);
        }
    }, 5000);
}

// Enhanced CSS for the corrected version
const enhancedStyles = `
<style>
.schema-info {
    margin-top: 1.25rem;
    padding: 1.25rem;
    background: linear-gradient(135deg, #dcfce7, #d4f4e2);
    border-radius: 12px;
    border: 1px solid #bbf7d0;
}

.schema-info h4 {
    color: #22c55e;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.table-info {
    margin: 0.75rem 0;
    padding: 0.75rem;
    background: rgba(59, 130, 246, 0.08);
    border-radius: 8px;
    border-left: 4px solid #3b82f6;
}

.table-info h5 {
    margin: 0 0 0.5rem 0;
    color: #1e293b;
    font-size: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.table-info p {
    margin: 0;
    font-size: 0.9rem;
    color: #4b5563;
    line-height: 1.4;
}

.table-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
    margin-bottom: 1.75rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #f9fafb, #f3f4f6);
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.stat-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.stat-item i {
    color: #3b82f6;
    width: 18px;
    font-size: 1.1rem;
}

.record-count {
    color: #ef4444;
    font-weight: bold;
}

.null-value {
    color: #9ca3af;
    font-style: italic;
    background: rgba(156, 163, 175, 0.1);
    padding: 2px 4px;
    border-radius: 4px;
}

.boolean-true {
    color: #22c55e;
    font-weight: bold;
    background: rgba(34, 197, 94, 0.1);
    padding: 2px 4px;
    border-radius: 4px;
}

.boolean-false {
    color: #ef4444;
    font-weight: bold;
    background: rgba(239, 68, 68, 0.1);
    padding: 2px 4px;
    border-radius: 4px;
}

.message-content {
    width: 100%;
}

.message-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: #6b7280;
    font-weight: 500;
}

.message-time {
    margin-left: auto;
    font-size: 0.75rem;
    opacity: 0.8;
}

.message-text {
    line-height: 1.6;
}

.message-text p {
    margin: 0.5rem 0;
}

.message-text p:first-child {
    margin-top: 0;
}

.message-text p:last-child {
    margin-bottom: 0;
}

.message-text ul {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
}

.message-text li {
    margin: 0.25rem 0;
}

.typing-indicator {
    opacity: 0.7;
}

.typing-animation {
    display: flex;
    gap: 0.25rem;
    align-items: center;
    padding: 0.75rem 0;
}

.typing-animation span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3b82f6;
    animation: typing 1.4s infinite ease-in-out;
}

.typing-animation span:nth-child(1) {
    animation-delay: 0s;
}

.typing-animation span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-animation span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 80%, 100% {
        transform: scale(0.8);
        opacity: 0.5;
    }
    40% {
        transform: scale(1);
        opacity: 1;
    }
}

.notification {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: 500;
    font-size: 0.95rem;
}

.notification-close {
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 0.25rem;
    border-radius: 4px;
    margin-left: auto;
    opacity: 0.7;
    transition: all 0.2s ease;
}

.notification-close:hover {
    background: rgba(0,0,0,0.1);
    opacity: 1;
}

.text-success {
    color: #22c55e;
}

.btn-loading {
    position: relative;
    color: transparent !important;
    pointer-events: none;
}

.btn-loading::after {
    content: '';
    position: absolute;
    width: 16px;
    height: 16px;
    margin: auto;
    border: 2px solid transparent;
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: white;
}

@keyframes spin {
    0% { transform: translate(-50%, -50%) rotate(0deg); }
    100% { transform: translate(-50%, -50%) rotate(360deg); }
}

/* Enhanced table styling */
#data-table {
    border-collapse: separate;
    border-spacing: 0;
    overflow: hidden;
    border-radius: 12px;
}

#data-table th {
    background: linear-gradient(135deg, #334155, #1e293b);
    color: white;
    padding: 1.25rem;
    text-align: left;
    font-weight: 600;
    position: sticky;
    top: 0;
    z-index: 10;
    border-bottom: 2px solid #1e293b;
}

#data-table th:first-child {
    border-top-left-radius: 12px;
}

#data-table th:last-child {
    border-top-right-radius: 12px;
}

#data-table td {
    padding: 1rem 1.25rem;
    border-bottom: 1px solid #e5e7eb;
    transition: background-color 0.2s ease;
}

#data-table tbody tr:hover {
    background: #f9fafb;
}

#data-table tbody tr:hover td {
    background: inherit;
}

/* Loading overlay enhancements */
.loading-overlay {
    backdrop-filter: blur(4px);
    background: rgba(0,0,0,0.6);
}

.loading-spinner {
    background: white;
    padding: 3rem;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    min-width: 300px;
}

.loading-spinner i {
    font-size: 3rem;
    color: #3b82f6;
    margin-bottom: 1.5rem;
    display: block;
}

.loading-spinner p {
    font-size: 1.2rem;
    color: #1e293b;
    margin: 0;
    font-weight: 500;
}

/* Success/Error message styling */
.success-message, .error-message, .info-message {
    padding: 1.25rem 1.75rem;
    border-radius: 12px;
    margin: 1.25rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: 500;
    animation: slideIn 0.3s ease;
}

.success-message {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #166534;
    border: 1px solid #bbf7d0;
}

.error-message {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    color: #991b1b;
    border: 1px solid #fecaca;
}

.info-message {
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    color: #1e40af;
    border: 1px solid #bfdbfe;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Responsive improvements */
@media (max-width: 768px) {
    .table-stats {
        grid-template-columns: 1fr;
        gap: 0.75rem;
    }

    .stat-item {
        padding: 0.75rem;
        justify-content: center;
        text-align: center;
    }

    .notification {
        max-width: calc(100vw - 40px);
        right: 20px;
    }

    .loading-spinner {
        min-width: 280px;
        margin: 20px;
    }
}
</style>
`;

// Inject enhanced styles
document.head.insertAdjacentHTML('beforeend', enhancedStyles);

console.log('✅ Data Assistant JavaScript loaded successfully');