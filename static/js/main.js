// ===== GLOBAL VARIABLES =====
let currentTwin = null;
let currentScenario = 'neutral';
let ragEnabled = false;
let isTyping = false;
let chatHistory = [];

// Twin configurations for all 12 characters
const twinConfigs = {
    // Middle Adolescence - Healthy
    kaiya: {
        name: 'Kaiya',
        age: 15,
        grade: '9th',
        avatar: '/static/images/kaiya-avatar.png',
        description: 'A well-balanced freshman who maintains good physical and mental health',
        mentalHealthType: 'healthy',
        ageGroup: 'middle',
        initialMessage: "Hey! I'm Kaiya. What's up?"
    },
    ethan: {
        name: 'Ethan',
        age: 14,
        grade: '9th', 
        avatar: '/static/images/ethan-avatar.png',
        description: 'An active freshman who prioritizes physical health and sports',
        mentalHealthType: 'healthy',
        ageGroup: 'middle',
        initialMessage: "Hi there! I'm Ethan. How's it going?"
    },
    
    // Middle Adolescence - Anxious
    jaden: {
        name: 'Jaden',
        age: 15,
        grade: '10th',
        avatar: '/static/images/jaden-avatar.png',
        description: 'Experiences anxiety and social challenges',
        mentalHealthType: 'anxiety',
        ageGroup: 'middle',
        initialMessage: "Um, hi... I'm Jaden. I guess we can talk."
    },
    nia: {
        name: 'Nia',
        age: 14,
        grade: '9th',
        avatar: '/static/images/nia-avatar.png',
        description: 'Struggles with social anxiety and academic pressure',
        mentalHealthType: 'anxiety',
        ageGroup: 'middle',
        initialMessage: "Hi... I'm Nia. Not sure what to say really."
    },
    
    // Middle Adolescence - Depression
    diego: {
        name: 'Diego',
        age: 15,
        grade: '10th',
        avatar: '/static/images/diego-avatar.png',
        description: 'Struggles with feelings of sadness and disconnection',
        mentalHealthType: 'depression',
        ageGroup: 'middle',
        initialMessage: "Hey. I'm Diego. I guess we can chat."
    },
    emily: {
        name: 'Emily',
        age: 14,
        grade: '9th',
        avatar: '/static/images/emily-avatar.png',
        description: 'Experiences persistent sadness and struggles with motivation',
        mentalHealthType: 'depression',
        ageGroup: 'middle',
        initialMessage: "Hi, I'm Emily. Don't really know what to talk about..."
    },
    
    // Late Adolescence - Healthy
    lucas: {
        name: 'Lucas',
        age: 17,
        grade: '12th',
        avatar: '/static/images/lucas-avatar.png',
        description: 'A high-achieving senior with excellent academic performance',
        mentalHealthType: 'healthy',
        ageGroup: 'late',
        initialMessage: "Hey! I'm Lucas. What would you like to talk about?"
    },
    maya: {
        name: 'Maya',
        age: 16,
        grade: '11th',
        avatar: '/static/images/maya-avatar.png',
        description: 'A well-balanced junior who demonstrates resilience',
        mentalHealthType: 'healthy',
        ageGroup: 'late',
        initialMessage: "Hi there! I'm Maya. How can I help?"
    },
    
    // Late Adolescence - Anxious
    hana: {
        name: 'Hana',
        age: 17,
        grade: '12th',
        avatar: '/static/images/hana-avatar.png',
        description: 'Experiences high levels of anxiety around academic performance',
        mentalHealthType: 'anxiety',
        ageGroup: 'late',
        initialMessage: "Oh, hi... I'm Hana. I'm kind of nervous but we can talk."
    },
    mateo: {
        name: 'Mateo',
        age: 16,
        grade: '11th',
        avatar: '/static/images/mateo-avatar.png',
        description: 'Struggles with social anxiety and peer acceptance',
        mentalHealthType: 'anxiety',
        ageGroup: 'late',
        initialMessage: "Hey... I'm Mateo. This is kinda awkward but whatever."
    },
    
    // Late Adolescence - Depression
    amara: {
        name: 'Amara',
        age: 18,
        grade: '12th',
        avatar: '/static/images/amara-avatar.png',
        description: 'Experiences persistent feelings of sadness and hopelessness',
        mentalHealthType: 'depression',
        ageGroup: 'late',
        initialMessage: "Hi, I'm Amara. I guess we can talk if you want to."
    },
    tavian: {
        name: 'Tavian',
        age: 17,
        grade: '12th',
        avatar: '/static/images/tavian-avatar.png',
        description: 'Struggles with depression and feelings of disconnection',
        mentalHealthType: 'depression',
        ageGroup: 'late',
        initialMessage: "Hey. I'm Tavian. Not really sure what to say..."
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
    
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
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
        }, index * 100);
    });
    
    // Add click handlers
    cards.forEach(card => {
        card.addEventListener('click', function() {
            const twinId = this.getAttribute('data-twin');
            if (twinId && twinConfigs[twinId]) {
                selectTwin(twinId);
            }
        });
    });
}

function selectTwin(twinId) {
    console.log(`Selecting twin: ${twinId}`);
    
    const card = document.querySelector(`[data-twin="${twinId}"]`);
    if (card) {
        // Add selection animation
        card.style.transform = 'scale(0.95)';
        card.style.transition = 'transform 0.1s ease';
        
        setTimeout(() => {
            // Navigate to chat page
            window.location.href = `/chat/${twinId}`;
        }, 100);
    }
}

// ===== CHAT PAGE FUNCTIONS =====
function initChatPage() {
    console.log('Initializing chat page...');
    
    // Get twin ID from URL
    const pathParts = window.location.pathname.split('/');
    currentTwin = pathParts[pathParts.length - 1];
    
    console.log('Current twin ID from URL:', currentTwin);
    
    if (!twinConfigs[currentTwin]) {
        console.error('Invalid twin ID:', currentTwin);
        window.location.href = '/';
        return;
    }
    
    // Initialize components
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
        console.error('Invalid twin:', currentTwin);
        return;
    }
    
    console.log('Initializing profile for:', config.name);
    
    // Update profile avatar
    const profileAvatar = document.getElementById('profileAvatar');
    if (profileAvatar) {
        profileAvatar.innerHTML = `<img src="${config.avatar}" alt="${config.name}" onerror="this.style.display='none'; this.parentNode.innerHTML='${config.name.charAt(0).toUpperCase()}';">`;
    }
    
    // Update profile name
    const profileName = document.getElementById('profileName');
    if (profileName) {
        profileName.textContent = config.name;
    }
    
    // Update profile details
    const profileDetails = document.getElementById('profileDetails');
    if (profileDetails) {
        const typeCapitalized = config.mentalHealthType.charAt(0).toUpperCase() + config.mentalHealthType.slice(1);
        const ageGroupText = config.ageGroup === 'middle' ? 'Middle Adolescence' : 'Late Adolescence';
        profileDetails.innerHTML = `
            Age: ${config.age} | Grade: ${config.grade}<br>
            ${ageGroupText}<br>
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
            twin_id: currentTwin,
            scenario: scenario
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification(`Switched to ${scenario} environment`);
            // Add a system message to show the scenario change
            addMessage('system', `Environment changed to ${scenario.charAt(0).toUpperCase() + scenario.slice(1)}`);
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
            twin_id: currentTwin,
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
            twin_id: currentTwin,
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
    } else if (sender === 'system') {
        avatarContent = '⚙️';
    } else {
        avatarContent = `<img src="${config.avatar}" alt="${config.name}" onerror="this.style.display='none'; this.parentNode.innerHTML='${config.name.charAt(0).toUpperCase()}';">`;
    }
    
    // Special styling for system messages
    const messageClass = sender === 'system' ? 'system' : sender;
    const systemStyle = sender === 'system' ? 'style="background: #f0f0f0; color: #666; font-style: italic; text-align: center;"' : '';
    
    // 渲染文本（注意区分 user / twin / system）
    let rendered;
    if (sender === 'twin' || sender === 'system') {
      rendered = renderAssistantText(String(content || ''));
    } else {
      // 用户文本：只转义 + 换行（不要做舞台指示替换）
      rendered = preserveLineBreaks(escapeHTML(String(content || '')));
    }
    
    messageDiv.innerHTML = `
      <div class="message-avatar">${avatarContent}</div>
      <div class="message-content" ${systemStyle}>
        ${rendered}
        <div class="message-time" style="font-size:.7rem;opacity:.6;margin-top:5px;">
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
    document.body.style.opacity = '0.8';
    setTimeout(() => {
        window.location.href = '/';
    }, 200);
}
// ===== SAFE RENDER PIPELINE FOR innerHTML =====
const STAGE_MAP = {
  scoffs: "😤",
  sighs: "😮‍💨",
  laughs: "😆",
  chuckles: "😄",
  sobs: "😭",
  cries: "😢",
  gasps: "😯",
  groans: "😖",
  grins: "😏",
  shrugs: "🤷",
  nods: "🫡",
  shakes_head: "🙅",
  whispers: "🤫",
  clears_throat: "🫤",
  coughs: "😷",
  sneezes: "🤧",
};

function escapeHTML(str) {
  return str
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// [verb words] -> emoji 或兜底成斜体（已转义后替换，安全）
function stageDirectionsToInline(html) {
  // 1) 优先按映射替换
  html = html.replace(/\[([a-z\s\-]+)\]/gi, (m, raw) => {
    const key = raw.trim().toLowerCase().replace(/\s+/g, "_");
    const emoji = STAGE_MAP[key];
    if (emoji) return `<span class="stage-emoji" aria-label="${raw}">${emoji}</span>`;
    // 2) 未命中映射：以斜体兜底
    return `<span class="stage-dir">[${raw}]</span>`;
  });
  return html;
}

// Emotion/Emotional tag 单独包裹，并在其前插入一个空行（等价于 \n\n）
function wrapEmotionTag(html) {
  // 找到最后一个 Emotion(al) tag
  const re = /(Emotion(?:al)?\s*tag\s*:\s*[^\n<]+)/ig;
  let lastMatch = null;
  html.replace(re, (m, _1, idx) => { lastMatch = { m, idx }; });
  if (!lastMatch) return html;

  // 在最后一次出现处包 span，并在其前面插入 <br><br>
  const before = html.slice(0, lastMatch.idx).replace(/(<br>\s*)+$/i, ""); // 去末尾多余 <br>
  const after = html.slice(lastMatch.idx + lastMatch.m.length);
  const wrapped = `<span class="emotion-tag">${lastMatch.m}</span>`;
  return `${before}<br><br>${wrapped}${after}`;
}

function linkify(html) {
  // 简单链接识别（可选）
  return html.replace(
    /\b(https?:\/\/[^\s<]+)\b/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
  );
}

function preserveLineBreaks(html) {
  // 把 \n 变成 <br>，让 innerHTML 与 pre-wrap 一致
  return html.replace(/\n/g, "<br>");
}

// 核心：把“模型文本”变成安全的 HTML 片段
function renderAssistantText(plainText) {
  // 1) 先整体转义
  let html = escapeHTML(plainText);

  // 2) 舞台指示 -> emoji 或斜体
  html = stageDirectionsToInline(html);

  // 3) 链接可点（可选）
  html = linkify(html);

  // 4) 换行
  html = preserveLineBreaks(html);

  // 5) Emotion tag 包裹 + 前置空行（必须放在换行之后执行一次，才能在最终 HTML 里插入 <br><br>）
  html = wrapEmotionTag(html);

  return html;
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
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const messageInput = document.getElementById('messageInput');
        if (messageInput && document.activeElement === messageInput) {
            sendMessage(event);
        }
    }
    
    if (event.key === 'Escape') {
        const messageInput = document.getElementById('messageInput');
        if (messageInput && document.activeElement === messageInput) {
            messageInput.value = '';
            autoResizeTextarea({ target: messageInput });
        }
    }
    
    if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        if (confirm('Clear chat history?')) {
            clearChat();
        }
    }
    
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
        const sidebar = document.querySelector('.sidebar');
        const chatContainer = document.querySelector('.chat-container');
        
        if (sidebar && chatContainer) {
            sidebar.style.maxHeight = '40vh';
            chatContainer.style.height = '60vh';
        }
    }
}

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
    
    handleResize();
    
    console.log('Digital Twin Platform loaded successfully!');
});

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
    chatHistory: () => [...chatHistory],
    twinConfigs: twinConfigs
};
