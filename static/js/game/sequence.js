// ========== Action Sequence Manager ==========

async function openSeqManager() {
  const gameId = document.getElementById('game-select').value;
  if (!gameId) { alert('Please select a game first'); return; }

  const gameSafe = gameId.replace(/\//g, '_');

  // Fetch all sequences
  let allSeqs;
  try {
    const resp = await fetch('/api/action-sequences');
    allSeqs = await resp.json();
  } catch (err) {
    alert('Failed to fetch sequences: ' + err.message);
    return;
  }

  const sequences = allSeqs[gameId] || [];

  // Render list
  const list = document.getElementById('seq-list');
  while (list.firstChild) list.removeChild(list.firstChild);

  if (sequences.length === 0) {
    document.getElementById('seq-empty-msg').classList.remove('hidden');
  } else {
    document.getElementById('seq-empty-msg').classList.add('hidden');
    for (const seq of sequences) {
      list.appendChild(_buildSeqRow(seq, gameSafe));
    }
  }

  // Save section: only enabled in human mode with actions
  const saveBtn = document.getElementById('seq-save-btn');
  const saveInput = document.getElementById('seq-save-name');
  const hasActions = lastState && lastState.action_history && lastState.action_history.length > 0;
  const canSave = currentMode === 'human' && sessionId && hasActions;
  saveBtn.disabled = !canSave;
  saveInput.disabled = !canSave;
  if (!canSave && currentMode === 'ai') {
    saveBtn.title = 'AI mode cannot save sequences';
  } else if (!canSave) {
    saveBtn.title = 'Start a game and perform actions first';
  } else {
    saveBtn.title = '';
  }

  // Auto-fill default name
  saveInput.value = '';
  saveInput.placeholder = 'Seq ' + _getNextSeqNumber(sequences);

  // Show loaded sequence indicator
  const indicator = document.getElementById('seq-loaded-indicator');
  if (loadedSequence) {
    indicator.textContent = 'Loaded: ' + loadedSequence.name;
    indicator.classList.remove('hidden');
  } else {
    indicator.classList.add('hidden');
  }

  document.getElementById('seq-manager-overlay').classList.remove('hidden');
}

function closeSeqManager() {
  document.getElementById('seq-manager-overlay').classList.add('hidden');
}

async function saveSequence() {
  const gameId = document.getElementById('game-select').value;
  if (!gameId) return;
  if (!lastState || !lastState.action_history || lastState.action_history.length === 0) {
    alert('No actions to save');
    return;
  }

  const input = document.getElementById('seq-save-name');
  const name = input.value.trim() || input.placeholder;

  // 检查同名序列是否已存在
  const gameSafe = gameId.replace(/\//g, '_');
  try {
    const checkResp = await fetch(`/api/action-sequences/${encodeURIComponent(gameSafe)}/${encodeURIComponent(name)}/sequence`);
    if (checkResp.ok) {
      if (!confirm(`Sequence "${name}" already exists. Overwrite?`)) return;
    }
  } catch (e) { /* ignore check errors */ }

  try {
    const resp = await fetch('/api/action-sequences/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game_id: gameId,
        name: name,
        actions: lastState.action_history,
        action_info: actionInfo,
      }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    // Refresh the list
    openSeqManager();
  } catch (err) {
    alert('Failed to save sequence: ' + err.message);
  }
}

async function loadSequence(gameSafe, seqName) {
  if (!sessionId || !ws || ws.readyState !== WebSocket.OPEN) {
    alert('Unable to load! Please start a game first.');
    return;
  }

  // Fetch sequence data
  let seqData;
  try {
    const resp = await fetch(`/api/action-sequences/${encodeURIComponent(gameSafe)}/${encodeURIComponent(seqName)}/sequence`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    seqData = await resp.json();
  } catch (err) {
    alert('Failed to load sequence: ' + err.message);
    return;
  }

  // Close manager and send replay
  closeSeqManager();
  loadedSequence = { game_safe: gameSafe, name: seqName };
  updateSeqBadge();
  ws.send(JSON.stringify({
    type: 'replay_sequence',
    actions: seqData.actions,
    delay_ms: 80,
  }));
}

async function deleteSequence(gameSafe, seqName) {
  if (!confirm('Delete sequence "' + seqName + '"?')) return;

  try {
    const resp = await fetch(`/api/action-sequences/${encodeURIComponent(gameSafe)}/${encodeURIComponent(seqName)}`, {
      method: 'DELETE',
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    // If deleted the loaded sequence, clear it
    if (loadedSequence && loadedSequence.game_safe === gameSafe && loadedSequence.name === seqName) {
      loadedSequence = null;
      updateSeqBadge();
    }
    // Refresh the list
    openSeqManager();
  } catch (err) {
    alert('Failed to delete sequence: ' + err.message);
  }
}

function _buildSeqRow(seq, gameSafe) {
  const row = document.createElement('div');
  row.className = 'seq-item';

  // Info section
  const info = document.createElement('div');
  info.className = 'seq-item-info';

  const nameEl = document.createElement('div');
  nameEl.className = 'seq-item-name';
  nameEl.textContent = seq.name;
  // Highlight if this is the currently loaded sequence
  if (loadedSequence && loadedSequence.game_safe === gameSafe && loadedSequence.name === seq.name) {
    nameEl.style.color = '#4fc3f7';
  }
  info.appendChild(nameEl);

  const descEl = document.createElement('div');
  descEl.className = 'seq-item-desc';
  descEl.textContent = seq.description;
  info.appendChild(descEl);

  if (seq.created_at) {
    const dateEl = document.createElement('div');
    dateEl.className = 'seq-item-date';
    dateEl.textContent = seq.created_at.slice(0, 10);
    info.appendChild(dateEl);
  }

  row.appendChild(info);

  // Action buttons
  const actions = document.createElement('div');
  actions.className = 'seq-item-actions';

  const loadBtn = document.createElement('button');
  loadBtn.textContent = 'Load';
  loadBtn.className = 'primary';
  loadBtn.style.cssText = 'font-size:11px; padding:3px 10px;';
  loadBtn.onclick = (e) => { e.stopPropagation(); loadSequence(gameSafe, seq.name); };
  actions.appendChild(loadBtn);

  const delBtn = document.createElement('button');
  delBtn.textContent = 'Del';
  delBtn.className = 'warning';
  delBtn.style.cssText = 'font-size:11px; padding:3px 8px;';
  delBtn.onclick = (e) => { e.stopPropagation(); deleteSequence(gameSafe, seq.name); };
  actions.appendChild(delBtn);

  row.appendChild(actions);
  return row;
}

function updateSeqBadge() {
  const badge = document.getElementById('seq-loaded-badge');
  if (!badge) return;
  if (loadedSequence) {
    badge.textContent = '▶ ' + loadedSequence.name;
    badge.style.display = '';
  } else {
    badge.textContent = '';
    badge.style.display = 'none';
  }
}

function _getNextSeqNumber(sequences) {
  let max = 0;
  for (const seq of sequences) {
    const m = seq.name.match(/^(?:Seq|序列) (\d+)$/);
    if (m) max = Math.max(max, parseInt(m[1]));
  }
  return max + 1;
}
