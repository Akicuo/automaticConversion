/**
 * Ticket Modal System for GGUF Forge
 * Handles conversation threads between users and admins
 */

let currentTicketId = null;

async function openTicketModal(requestId) {
    // First get the ticket for this request
    try {
        const res = await fetch(`/api/tickets/request/${requestId}`);
        const data = await res.json();

        if (!data.ticket) {
            alert('No discussion thread found for this request.');
            return;
        }

        currentTicketId = data.ticket.id;
        showTicketModal(currentTicketId);
    } catch (e) {
        console.error(e);
        alert('Failed to load discussion thread');
    }
}

function showTicketModal(ticketId) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('ticket-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'ticket-modal';
        modal.innerHTML = `
            <div class="ticket-modal-backdrop" onclick="closeTicketModal()" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.8); z-index: 100;"></div>
            <div class="ticket-modal-content glass-panel" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 500px; max-width: 90vw; max-height: 80vh; z-index: 101; padding: 1.5rem; display: flex; flex-direction: column;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 id="ticket-title" style="margin: 0; font-size: 1.1rem;">Discussion Thread</h3>
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        <button id="ticket-reopen-btn" onclick="reopenTicket()" style="display: none; background: transparent; color: #60a5fa; border: 1px solid #60a5fa; padding: 0.3rem 0.6rem; border-radius: 4px; cursor: pointer; font-size: 0.75rem;">Reopen</button>
                        <button onclick="closeTicketModal()" style="background: transparent; color: var(--text-secondary); border: none; cursor: pointer; font-size: 1.2rem;">&times;</button>
                    </div>
                </div>
                <div id="ticket-messages" style="flex: 1; overflow-y: auto; border: 1px solid var(--glass-border); border-radius: 6px; padding: 1rem; margin-bottom: 1rem; background: rgba(0,0,0,0.3); max-height: 300px;">
                    <div style="text-align: center; color: var(--text-secondary);">Loading...</div>
                </div>
                <div id="ticket-reply-section" style="display: flex; gap: 0.5rem;">
                    <input type="text" id="ticket-reply-input" placeholder="Type your message..." style="flex: 1; background: rgba(255,255,255,0.05); border: 1px solid var(--glass-border); padding: 0.75rem; border-radius: 6px; color: var(--text-primary);">
                    <button onclick="sendTicketReply()" class="primary" style="padding: 0.75rem 1.25rem;">Send</button>
                </div>
                <div id="ticket-closed-notice" style="display: none; text-align: center; color: var(--text-secondary); padding: 0.75rem; background: rgba(255,255,255,0.05); border-radius: 6px; font-size: 0.9rem;">
                    This thread is closed. <a href="#" onclick="reopenTicket(); return false;" style="color: #60a5fa;">Reopen to reply</a>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    modal.style.display = 'block';
    loadTicketMessages(ticketId);
}

function closeTicketModal() {
    const modal = document.getElementById('ticket-modal');
    if (modal) {
        modal.style.display = 'none';
    }
    currentTicketId = null;
}

async function loadTicketMessages(ticketId) {
    const messagesContainer = document.getElementById('ticket-messages');

    try {
        const res = await fetch(`/api/tickets/${ticketId}/messages`);
        const data = await res.json();

        // Update title
        document.getElementById('ticket-title').textContent = `Discussion: ${data.request.hf_repo_id}`;

        // Handle closed status
        const replySection = document.getElementById('ticket-reply-section');
        const closedNotice = document.getElementById('ticket-closed-notice');
        const reopenBtn = document.getElementById('ticket-reopen-btn');

        if (data.ticket.status === 'closed') {
            replySection.style.display = 'none';
            closedNotice.style.display = 'block';
            reopenBtn.style.display = 'inline-block';
        } else {
            replySection.style.display = 'flex';
            closedNotice.style.display = 'none';
            reopenBtn.style.display = 'none';
        }

        // Render messages
        if (data.messages.length === 0) {
            messagesContainer.innerHTML = '<div style="text-align: center; color: var(--text-secondary);">No messages yet. Start the conversation!</div>';
            return;
        }

        messagesContainer.innerHTML = '';
        data.messages.forEach(msg => {
            const msgEl = document.createElement('div');
            msgEl.style.cssText = `margin-bottom: 0.75rem; padding: 0.75rem; border-radius: 6px; ${msg.sender_role === 'admin' ? 'background: rgba(96, 165, 250, 0.15); margin-left: 1rem;' : 'background: rgba(255,255,255,0.05); margin-right: 1rem;'}`;

            const time = new Date(msg.created_at).toLocaleString();
            msgEl.innerHTML = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem; font-size: 0.75rem; color: var(--text-secondary);">
                    <span style="font-weight: 500; color: ${msg.sender_role === 'admin' ? '#60a5fa' : 'var(--text-primary)'};">${msg.sender} ${msg.sender_role === 'admin' ? '(Admin)' : ''}</span>
                    <span>${time}</span>
                </div>
                <div style="font-size: 0.9rem;">${escapeHtml(msg.message)}</div>
            `;
            messagesContainer.appendChild(msgEl);
        });

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

    } catch (e) {
        console.error(e);
        messagesContainer.innerHTML = '<div style="text-align: center; color: var(--error);">Failed to load messages</div>';
    }
}

async function sendTicketReply() {
    if (!currentTicketId) return;

    const input = document.getElementById('ticket-reply-input');
    const message = input.value.trim();
    if (!message) return;

    try {
        const res = await fetch(`/api/tickets/${currentTicketId}/reply`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        if (res.ok) {
            input.value = '';
            loadTicketMessages(currentTicketId);
        } else {
            const data = await res.json();
            alert('Failed to send: ' + (data.detail || 'Unknown error'));
        }
    } catch (e) {
        alert('Failed to send message');
    }
}

async function reopenTicket() {
    if (!currentTicketId) return;

    try {
        const res = await fetch(`/api/tickets/${currentTicketId}/reopen`, { method: 'POST' });
        if (res.ok) {
            loadTicketMessages(currentTicketId);
            if (typeof fetchMyRequests === 'function') fetchMyRequests();
            if (typeof fetchTickets === 'function') fetchTickets();
        } else {
            const data = await res.json();
            alert('Failed to reopen: ' + (data.detail || 'Unknown error'));
        }
    } catch (e) {
        alert('Failed to reopen thread');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle Enter key in reply input
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.id === 'ticket-reply-input') {
        sendTicketReply();
    }
});
