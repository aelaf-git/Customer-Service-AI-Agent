// script.js (Final Production Version for Netlify)
(function() {
    // --- CRITICAL: CONFIGURE THIS WITH YOUR LIVE RENDER URL ---
    const API_BASE_URL = "http://127.0.0.1:8000";

    // Find the script tag and get the businessId
    const scriptTag = document.currentScript;
    if (!scriptTag) return;
    const businessId = scriptTag.dataset.businessId;

    if (!businessId) {
        console.error("Chatbot Error: 'data-business-id' is missing.");
        return;
    }

    document.addEventListener("DOMContentLoaded", function() {
        fetch(`${API_BASE_URL}/config/${businessId}`)
            .then(response => {
                if (!response.ok) throw new Error(`Config fetch failed: ${response.status}`);
                return response.json();
            })
            .then(config => {
                buildChatbotUI(config);
            })
            .catch(error => console.error("Failed to load chatbot configuration:", error));
    });

    function buildChatbotUI(config) {
        const style = document.createElement('style');
        style.innerHTML = `
            :root { --brand-color: ${config.brand_color || '#007bff'}; }
            /* The rest of the CSS is the same */
            #chat-bubble { position: fixed; bottom: 20px; right: 20px; width: 60px; height: 60px; background-color: var(--brand-color); border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.2); z-index: 9998; transition: transform 0.2s; } #chat-bubble:hover { transform: scale(1.1); } #chat-bubble svg { width: 32px; height: 32px; fill: white; } #chat-window { position: fixed; bottom: 90px; right: 20px; width: 350px; height: 500px; background-color: white; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); display: flex; flex-direction: column; overflow: hidden; opacity: 0; transform: translateY(20px); pointer-events: none; transition: opacity 0.3s, transform 0.3s; z-index: 9999; } #chat-window.open { opacity: 1; transform: translateY(0); pointer-events: auto; } #chat-header { background-color: var(--brand-color); color: white; padding: 15px; font-family: sans-serif; font-weight: bold; display: flex; justify-content: space-between; align-items: center; } #close-btn { background: none; border: none; color: white; font-size: 20px; cursor: pointer; } #chat-messages { flex-grow: 1; padding: 15px; overflow-y: auto; font-family: sans-serif; line-height: 1.5; display: flex; flex-direction: column; } .message { margin-bottom: 10px; padding: 10px 15px; border-radius: 20px; max-width: 80%; word-wrap: break-word; } .user-message { background-color: var(--brand-color); color: white; align-self: flex-end; margin-left: auto; } .bot-message { background-color: #f1f1f1; color: #333; align-self: flex-start; } .typing-indicator { color: #999; font-style: italic; } #chat-input-container { border-top: 1px solid #eee; padding: 10px; display: flex; } #chat-input { border: 1px solid #ccc; border-radius: 20px; padding: 10px 15px; flex-grow: 1; margin-right: 10px; font-size: 14px; } #chat-input:focus { outline: none; border-color: var(--brand-color); } #send-btn { background-color: var(--brand-color); border: none; color: white; padding: 10px 15px; border-radius: 20px; cursor: pointer; }
        `;
        document.head.appendChild(style);

        const container = document.getElementById('chatbot-container');
        if (!container) return;
        
        container.innerHTML = `
            <div id="chat-bubble"> <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg> </div> <div id="chat-window"> <div id="chat-header"> <span>${config.agent_name || 'AI Assistant'}</span> <button id="close-btn">&times;</button> </div> <div id="chat-messages"></div> <div id="chat-input-container"> <input type="text" id="chat-input" placeholder="Ask a question..."> <button id="send-btn">Send</button> </div> </div>
        `;

        const chatBubble = document.getElementById('chat-bubble');
        const chatWindow = document.getElementById('chat-window');
        const closeBtn = document.getElementById('close-btn');
        const sendBtn = document.getElementById('send-btn');
        const chatInput = document.getElementById('chat-input');
        
        addMessage(config.welcome_message || 'Hello!', 'bot');
        
        chatBubble.addEventListener('click', () => toggleChatWindow());
        closeBtn.addEventListener('click', () => toggleChatWindow());
        sendBtn.addEventListener('click', () => sendMessage());
        chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
    }

    function toggleChatWindow() {
        document.getElementById('chat-window').classList.toggle('open');
    }

    async function sendMessage() {
        const input = document.getElementById('chat-input');
        const question = input.value.trim();
        if (!question) return;

        addMessage(question, 'user');
        input.value = '';
        const typingIndicator = addMessage('Typing...', 'bot typing-indicator');
        
        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    businessId: businessId,
                }),
            });
            if (!response.ok) throw new Error(`Chat fetch failed: ${response.status}`);
            const data = await response.json();
            typingIndicator.remove();
            addMessage(data.answer, 'bot');
        } catch (error) {
            console.error("Error fetching chat response:", error);
            typingIndicator.remove();
            addMessage("Sorry, I'm having trouble connecting.", 'bot');
        }
    }

    function addMessage(text, type) {
        const container = document.getElementById('chat-messages');
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}-message`;
        msgDiv.textContent = text;
        container.appendChild(msgDiv);
        container.scrollTop = container.scrollHeight;
        return msgDiv;
    }
})();