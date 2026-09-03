// ========== Action Controls ==========
function setupActionButtons() {
  const area = document.getElementById('button-area');
  area.innerHTML = '';
  for (const [id, meaning] of Object.entries(actionInfo)) {
    const btn = document.createElement('button');
    btn.textContent = `${id}: ${meaning}`;
    btn.onclick = () => sendAction(parseInt(id));
    area.appendChild(btn);
  }
}

function setupKeyMap() {
  keyMap = {};
  const keys = Object.keys(actionInfo);
  // Number keys for first 10 actions
  for (let i = 0; i < Math.min(keys.length, 10); i++) {
    keyMap[String(i)] = parseInt(keys[i]);
  }
  // Arrow keys for common games
  if (keys.length >= 4) {
    // Try to map arrows based on action meanings
    for (const [id, meaning] of Object.entries(actionInfo)) {
      const m = meaning.toLowerCase();
      if (m.includes('up') || m.includes('上') || m.includes('北')) keyMap['ArrowUp'] = parseInt(id);
      if (m.includes('down') || m.includes('下') || m.includes('南')) keyMap['ArrowDown'] = parseInt(id);
      if (m.includes('left') || m.includes('左') || m.includes('西')) keyMap['ArrowLeft'] = parseInt(id);
      if (m.includes('right') || m.includes('右') || m.includes('东')) keyMap['ArrowRight'] = parseInt(id);
      if (m.includes('fire') || m.includes('发射')) keyMap[' '] = parseInt(id);
    }
  }
}

function toggleControlMode() {
  const isButton = document.getElementById('toggle-button-mode').checked;
  document.getElementById('console-input-area').classList.toggle('hidden', isButton);
  document.getElementById('button-area').classList.toggle('hidden', !isButton);
}

function sendConsoleAction() {
  const input = document.getElementById('action-input');
  const val = input.value.trim().toLowerCase();
  if (val === 'r' || val === 'reset') {
    resetGame();
  } else {
    const num = parseInt(val);
    if (!isNaN(num)) sendAction(num);
  }
  input.value = '';
  input.focus();
}

function sendAction(action) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (isReplaying) return;
  isLive = true;
  ws.send(JSON.stringify({ type: 'action', action }));
  document.getElementById('btn-undo').disabled = false;
}

function handleKeyboard(e) {
  // Don't capture when typing in inputs
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (!sessionId || currentMode === 'ai') return;

  // Button mode: use keyMap
  if (document.getElementById('toggle-button-mode').checked) {
    const action = keyMap[e.key];
    if (action !== undefined) {
      e.preventDefault();
      sendAction(action);
    }
  }
}

