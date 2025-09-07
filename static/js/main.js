// ===== GLOBAL VARIABLES =====
let currentTwin = 'healthy';
let currentScenario = 'neutral';
let ragEnabled = false;
let isTyping = false;
let chatHistory = [];

// Twin configurations - UPDATED WITH NEW NAMES AND AVATARS
const twinConfigs = {
    healthy: {
        name: 'Eli Carter',
        age: 17,
        grade: '12th',
        avatar: '/static/images/eli-carter-avatar.png',
        description: 'A well-balanced adolescent with good mental health',
        initialMessage: "Hi! I'm Eli Carter. What would you like to talk about?"
    },
    anxiety: {
        name: 'Luna Marquez',
        age: 17,
        grade: '11th',
        avatar: '/static/images/luna-marquez-avatar.png',
        description: 'Experiences anxiety and social challenges',
        initialMessage: "Hey... I'm Luna Marquez. I guess we can talk if you want."
    },
    depression: {
        name: 'Mina Chen',
        age: 18,
        grade: 'Senior',
        avatar: '/static/images/mina-chen-avatar.png',
        description: 'Struggles with feelings of sadness and hopelessness',
        initialMessage: "Hi, I'm Mina Chen. Not sure what to say really..."
    }
};

// ===== UTILITY FUNCTIONS =====
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'error' ? '#e74c3c' : '#7F96B2'};
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        z-index: 1000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after delay
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

function formatTime(date = new Date()) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ===== MAIN PAGE FUNCTIONS =====
function initMainPage() {
    console.log('Initializing main page...');
    
    // Add page load animation
    const cards = document.querySelectorAll('.twin-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(50px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 200);
    });
    
    // Add click handlers
    cards.forEach(card => {
        card.addEventListener('click', function() {
            const twinType = this.classList.contains('healthy') ? 'healthy' :
                           this.classList.contains('anxiety') ? 'anxiety' : 'depression';
            selectTwin(twinType);
        });
    });
}

function selectTwin(twinType) {
    console.log(`Selecting twin: ${twinType}`);
    
    const card = document.querySelector(`.twin-card.${twinType}`);
    if (card) {
        // Add selection animation
        card.style.transform = 'scale(0.95)';
        card.style.transition = 'transform 0.1s ease';
        
        setTimeout(() => {
            // Navigate to chat page
            window.location.href = `/chat/${twinType}`;
        }, 100);
    }
}

// ===== CHAT PAGE FUNCTIONS =====
function initChatPage() {
    console.log('Initializing chat page...');
    
    // Get twin type from URL
    const pathParts = window.location.pathname.split('/');
    currentTwin = pathParts[pathParts.length - 1] || 'healthy';
    
    console.log('Current twin type from URL:', currentTwin);
    
    // Initialize components immediately
    initializeTwinProfile();
    updateChatHeader();
    setupEventListeners();
    
    // Add initial message after a short delay
    setTimeout(() => {
        addInitialMessage();
    }, 500);
}

function initializeTwinProfile() {
    const config = twinConfigs[currentTwin];
    if (!config) {
        console.error('Invalid twin type:', currentTwin);
        return;
    }
    
    console.log('Initializing profile for:', config.name, 'Type:', currentTwin);
    
    // Update profile avatar with image
    const profileAvatar = document.getElementById('profileAvatar');
    if (profileAvatar) {
        profileAvatar.innerHTML = `<img src="${config.avatar}" alt="${config.name}">`;
    }
    
    // Update profile name
    const profileName = document.getElementById('profileName');
    if (profileName) {
        profileName.textContent = config.name;
        console.log('Updated profile name to:', config.name);
    }
    
    // Update profile details
    const profileDetails = document.getElementById('profileDetails');
    if (profileDetails) {
        const typeCapitalized = currentTwin.charAt(0).toUpperCase() + currentTwin.slice(1);
        profileDetails.innerHTML = `
            Age: ${config.age} | Grade: ${config.grade}<br>
            Type: ${typeCapitalized}<br>
            ${config.description}
        `;
    }
}

function updateChatHeader() {
    const config = twinConfigs[currentTwin];
    const ragStatus = ragEnabled ? 'ON' : 'OFF';
    const scenarioText = currentScenario.charAt(0).toUpperCase() + currentScenario.slice(1);
    
    const chatTitle = document.getElementById('chatTitle');
    if (chatTitle) chatTitle.textContent = `Chat with ${config.name}`;
    
    const chatSubtitle = document.getElementById('chatSubtitle');
    if (chatSubtitle) {
        chatSubtitle.textContent = `${scenarioText} Environment • Enhanced Memory: ${ragStatus}`;
    }
}

function setupEventListeners() {
    // Message form submission
    const chatForm = document.querySelector('.chat-input-form');
    if (chatForm) {
        chatForm.addEventListener('submit', sendMessage);
    }
    
    // Auto-resize textarea
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('input', autoResizeTextarea);
        messageInput.addEventListener('keydown', handleKeyDown);
    }
    
    // Back button
    const backButton = document.querySelector('.back-button');
    if (backButton) {
        backButton.addEventListener('click', goBack);
    }
    
    // Scenario options
    const scenarioOptions = document.querySelectorAll('.scenario-option');
    scenarioOptions.forEach(option => {
        option.addEventListener('click', function() {
            const scenario = this.getAttribute('data-scenario');
            selectScenario(scenario);
        });
    });
    
    // RAG toggle
    const ragToggle = document.getElementById('ragToggle');
    if (ragToggle) {
        ragToggle.addEventListener('change', toggleRAG);
    }
}

function addInitialMessage() {
    const config = twinConfigs[currentTwin];
    if (config && config.initialMessage) {
        setTimeout(() => {
            addMessage('twin', config.initialMessage);
        }, 500);
    }
}

function selectScenario(scenario) {
    if (scenario === currentScenario) return;
    
    console.log(`Switching scenario to: ${scenario}`);
    currentScenario = scenario;
    
    // Update UI
    document.querySelectorAll('.scenario-option').forEach(option => {
        option.classList.remove('active');
    });
    
    const selectedOption = document.querySelector(`[data-scenario="${scenario}"]`);
    if (selectedOption) {
        selectedOption.classList.add('active');
    }
    
    updateChatHeader();
    
    // Send scenario change to backend
    fetch('/api/switch_scenario', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            twin_type: currentTwin,
            scenario: scenario
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification(`Switched to ${scenario} environment`);
        }
    })
    .catch(error => {
        console.error('Error switching scenario:', error);
        showNotification('Failed to switch scenario', 'error');
    });
}

function toggleRAG() {
    const ragToggle = document.getElementById('ragToggle');
    ragEnabled = ragToggle ? ragToggle.checked : false;
    
    console.log(`RAG toggled: ${ragEnabled}`);
    updateChatHeader();
    
    // Send RAG toggle to backend
    fetch('/api/toggle_rag', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            twin_type: currentTwin,
            rag_enabled: ragEnabled
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification(`Enhanced Memory ${ragEnabled ? 'enabled' : 'disabled'}`);
        }
    })
    .catch(error => {
        console.error('Error toggling RAG:', error);
        showNotification('Failed to toggle Enhanced Memory', 'error');
    });
}

function sendMessage(event) {
    event.preventDefault();
    
    const messageInput = document.getElementById('messageInput');
    if (!messageInput) return;
    
    const message = messageInput.value.trim();
    
    if (!message || isTyping) return;
    
    console.log('Sending message:', message);
    
    // Add user message to chat
    addMessage('user', message);
    
    // Clear input
    messageInput.value = '';
    autoResizeTextarea({ target: messageInput });
    
    // Show typing indicator
    showTypingIndicator();
    
    // Send message to backend
    fetch('/api/send_message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            twin_type: currentTwin,
            scenario: currentScenario,
            rag_enabled: ragEnabled
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        hideTypingIndicator();
        if (data.response) {
            addMessage('twin', data.response);
        } else {
            throw new Error('No response received');
        }
    })
    .catch(error => {
        console.error('Error sending message:', error);
        hideTypingIndicator();
        addMessage('twin', 'Sorry, I had trouble understanding that. Could you try again?');
        showNotification('Failed to send message', 'error');
    });
}

function addMessage(sender, content, timestamp = new Date()) {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    const config = twinConfigs[currentTwin];
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    // Determine avatar
    let avatarContent;
    if (sender === 'user') {
        avatarContent = '👤';
    } else {
        avatarContent = `<img src="${config.avatar}" alt="${config.name}">`;
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatarContent}</div>
        <div class="message-content">
            ${content}
            <div class="message-time" style="font-size: 0.7rem; opacity: 0.6; margin-top: 5px;">
                ${formatTime(timestamp)}
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    
    // Add to chat history
    chatHistory.push({
        sender: sender,
        content: content,
        timestamp: timestamp
    });
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Add animation
    setTimeout(() => {
        messageDiv.style.opacity = '1';
    }, 50);
}

function showTypingIndicator() {
    isTyping = true;
    const typingIndicator = document.getElementById('typingIndicator');
    const sendButton = document.getElementById('sendButton');
    
    if (typingIndicator) {
        const config = twinConfigs[currentTwin];
        typingIndicator.textContent = `${config.name} is typing...`;
        typingIndicator.style.display = 'block';
    }
    
    if (sendButton) {
        sendButton.disabled = true;
        sendButton.classList.add('loading');
    }
    
    // Scroll to show typing indicator
    const messagesContainer = document.getElementById('chatMessages');
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

function hideTypingIndicator() {
    isTyping = false;
    const typingIndicator = document.getElementById('typingIndicator');
    const sendButton = document.getElementById('sendButton');
    
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }
    
    if (sendButton) {
        sendButton.disabled = false;
        sendButton.classList.remove('loading');
    }
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage(event);
    }
}

function autoResizeTextarea(event) {
    const textarea = event.target;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function goBack() {
    // Add some animation before leaving
    document.body.style.opacity = '0.8';
    setTimeout(() => {
        window.location.href = '/';
    }, 200);
}

// ===== CHAT UTILITIES =====
function clearChat() {
    const messagesContainer = document.getElementById('chatMessages');
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
    }
    chatHistory = [];
    addInitialMessage();
}

function exportChatHistory() {
    const config = twinConfigs[currentTwin];
    const timestamp = new Date().toISOString().split('T')[0];
    const filename = `chat_${config.name}_${timestamp}.json`;
    
    const data = {
        twin: config,
        scenario: currentScenario,
        rag_enabled: ragEnabled,
        messages: chatHistory,
        exported_at: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showNotification('Chat history exported');
}

function copyLastMessage() {
    if (chatHistory.length === 0) return;
    
    const lastMessage = chatHistory[chatHistory.length - 1];
    if (lastMessage.sender === 'twin') {
        navigator.clipboard.writeText(lastMessage.content)
            .then(() => showNotification('Message copied to clipboard'))
            .catch(() => showNotification('Failed to copy message', 'error'));
    }
}

// ===== ERROR HANDLING =====
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showNotification('Something went wrong. Please refresh the page.', 'error');
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showNotification('Network error. Please check your connection.', 'error');
});

// ===== KEYBOARD SHORTCUTS =====
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to send message
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const messageInput = document.getElementById('messageInput');
        if (messageInput && document.activeElement === messageInput) {
            sendMessage(event);
        }
    }
    
    // Escape to clear input
    if (event.key === 'Escape') {
        const messageInput = document.getElementById('messageInput');
        if (messageInput && document.activeElement === messageInput) {
            messageInput.value = '';
            autoResizeTextarea({ target: messageInput });
        }
    }
    
    // Ctrl/Cmd + K to clear chat
    if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        if (confirm('Clear chat history?')) {
            clearChat();
        }
    }
    
    // Ctrl/Cmd + S to export chat
    if ((event.ctrlKey || event.metaKey) && event.key === 's') {
        event.preventDefault();
        exportChatHistory();
    }
});

// ===== RESPONSIVE UTILITIES =====
function isMobile() {
    return window.innerWidth <= 768;
}

function handleResize() {
    if (isMobile()) {
        // Adjust layout for mobile
        const sidebar = document.querySelector('.sidebar');
        const chatContainer = document.querySelector('.chat-container');
        
        if (sidebar && chatContainer) {
            // Mobile-specific adjustments
            sidebar.style.maxHeight = '40vh';
            chatContainer.style.height = '60vh';
        }
    }
}

// Debounced resize handler
const debouncedResize = debounce(handleResize, 250);
window.addEventListener('resize', debouncedResize);

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');
    
    // Check if we're on main page or chat page
    if (window.location.pathname === '/') {
        initMainPage();
    } else if (window.location.pathname.startsWith('/chat/')) {
        initChatPage();
    }
    
    // Handle initial resize
    handleResize();
    
    // Add some easter eggs
    console.log('Digital Twin Platform loaded successfully!');
    console.log('Keyboard shortcuts:');
    console.log('- Ctrl/Cmd + Enter: Send message');
    console.log('- Escape: Clear input');
    console.log('- Ctrl/Cmd + K: Clear chat');
    console.log('- Ctrl/Cmd + S: Export chat');
});

// ===== SERVICE WORKER (OPTIONAL) =====
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/static/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}

// ===== ANALYTICS (PLACEHOLDER) =====
function trackEvent(eventName, properties = {}) {
    // Placeholder for analytics tracking
    console.log('Track event:', eventName, properties);
    
    // You can integrate with Google Analytics, Mixpanel, etc. here
    // Example: gtag('event', eventName, properties);
}

function trackTwinSelection(twinType) {
    trackEvent('twin_selected', { twin_type: twinType });
}

function trackMessageSent(messageLength, twinType, scenario, ragEnabled) {
    trackEvent('message_sent', {
        message_length: messageLength,
        twin_type: twinType,
        scenario: scenario,
        rag_enabled: ragEnabled
    });
}

function trackScenarioChange(fromScenario, toScenario, twinType) {
    trackEvent('scenario_changed', {
        from_scenario: fromScenario,
        to_scenario: toScenario,
        twin_type: twinType
    });
}

function trackRAGToggle(enabled, twinType) {
    trackEvent('rag_toggled', {
        enabled: enabled,
        twin_type: twinType
    });
}

// ===== EXPORT FOR GLOBAL ACCESS =====
window.DigitalTwinApp = {
    selectTwin,
    sendMessage,
    addMessage,
    clearChat,
    exportChatHistory,
    selectScenario,
    toggleRAG,
    currentTwin: () => currentTwin,
    currentScenario: () => currentScenario,
    ragEnabled: () => ragEnabled,
    chatHistory: () => [...chatHistory]
};
