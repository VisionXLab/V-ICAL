// ========== Chat View ==========
let chatMsgCount = 0;
let pendingAiMsgDiv = null;

function toggleChatView() {
  const overlay = document.getElementById('chat-overlay');
  const btn = document.getElementById('ai-btn-chat');
  const hidden = overlay.classList.toggle('hidden');
  btn.textContent = hidden ? 'Chat Log' : 'Hide Chat';
  if (!hidden) {
    const msgs = document.getElementById('chat-messages');
    msgs.scrollTop = msgs.scrollHeight;
  }
}

function clearChatView() {
  document.getElementById('chat-messages').innerHTML = '';
  chatMsgCount = 0;
  pendingAiMsgDiv = null;
  updateChatTitle();
}

function updateChatTitle() {
  const turns = Math.floor(chatMsgCount);
  document.getElementById('chat-title').textContent =
    turns > 0 ? `Conversation Log (${turns} turn${turns !== 1 ? 's' : ''})` : 'Conversation Log';
}

function showLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('img-lightbox').classList.remove('hidden');
  event.stopPropagation();
}

function appendChatPrompt(data) {
  const container = document.getElementById('chat-messages');
  const { step, prompt, frame_b64, context_frames, context_videos_b64, video_mode, is_retry } = data;
  const isInit = (step === 0 || step === 'init') && !is_retry;

  const userMsg = document.createElement('div');
  userMsg.className = 'chat-msg chat-msg-user' + (is_retry ? ' chat-msg-retry' : '');

  let mediaHTML = '<div class="chat-media-blocks">';

  if (isInit) {
    if (video_mode && context_videos_b64 && context_videos_b64.length > 0) {
      context_videos_b64.forEach((vid_b64, vi) => {
        const contextVideoId = `chat-context-video-${Date.now()}-${vi}`;
        mediaHTML += `
          <div class="chat-media-block">
            <div class="chat-media-label">📹 Demo video ${context_videos_b64.length > 1 ? '#' + (vi + 1) : ''} (context)</div>
            <video id="${contextVideoId}" controls style="max-width: 100%; border-radius: 4px;">
              <source src="data:video/mp4;base64,${vid_b64}" type="video/mp4">
              Your browser does not support video playback.
            </video>
          </div>
        `;
      });
    } else if (context_frames && context_frames.length > 0) {
      mediaHTML += '<div class="chat-media-block"><div class="chat-media-label">📸 Demo frames (context)</div><div class="chat-frames">';
      context_frames.forEach((f, i) => {
        mediaHTML += `<img class="chat-thumb chat-thumb-ctx" src="data:image/jpeg;base64,${f}"
          title="Context frame ${i + 1}" onclick="showLightbox(this.src)">`;
      });
      mediaHTML += '</div></div>';
    }
  }

  if (frame_b64) {
    const frameLabel = isInit ? '🎯 Initial state (what AI plays)' : '📍 Current state';
    mediaHTML += `
      <div class="chat-media-block">
        <div class="chat-media-label">${frameLabel}</div>
        <img class="chat-thumb" src="data:image/jpeg;base64,${frame_b64}"
          title="Step ${step} frame" onclick="showLightbox(this.src)" style="border: 2px solid #64b5f6;">
      </div>
    `;
  }

  mediaHTML += '</div>';

  const retryLabel = is_retry
    ? `<span style="color:#ef6c00; font-size: 11px;">[Retry ${data.retry_count}/${data.max_retries}] </span>`
    : '';
  userMsg.innerHTML = `<div class="chat-msg-role">User &mdash; Step ${step+1}</div>
${mediaHTML}<div class="chat-msg-text">${retryLabel}${escapeHTML(prompt || '')}</div>`;

  const aiMsg = document.createElement('div');
  aiMsg.className = 'chat-msg chat-msg-ai chat-msg-thinking';
  aiMsg.innerHTML = `<div class="chat-msg-role">AI &rarr; Step ${step+1}</div>
<div class="chat-msg-text" style="color: #aaa; font-style: italic;">Thinking...</div>`;

  const turn = document.createElement('div');
  turn.className = 'chat-turn';
  turn.appendChild(userMsg);
  turn.appendChild(aiMsg);
  container.appendChild(turn);
  container.scrollTop = container.scrollHeight;

  pendingAiMsgDiv = aiMsg;
  chatMsgCount++;
  updateChatTitle();
}

function updateChatResponse(data) {
  if (!pendingAiMsgDiv) return;
  const { step, is_valid, raw_response, parsed_action, action_meaning, usage } = data;
  const actionMeaningStr = action_meaning ? ` - ${action_meaning}` : '';

  pendingAiMsgDiv.className = 'chat-msg chat-msg-ai' + (is_valid ? '' : ' chat-msg-invalid');
  const actionBg = is_valid ? '#f5f5f5' : '#1a0000';
  const actionColor = is_valid ? '#666' : '#ef5350';
  const actionBorder = is_valid ? '#e0e0e0' : '#c62828';

  let usageHTML = '';
  if (usage && (usage.input != null || usage.output != null)) {
    const parts = [];
    if (usage.input != null) parts.push(`Input: ${usage.input.toLocaleString()}`);
    if (usage.output != null) parts.push(`Output: ${usage.output.toLocaleString()}`);
    if (usage.thoughts != null) parts.push(`Thoughts: ${usage.thoughts.toLocaleString()}`);
    if (usage.total != null) parts.push(`Total: ${usage.total.toLocaleString()}`);
    let line = parts.join(' · ') + ' tokens';
    const media = [];
    if (usage.input_image_tokens != null) media.push(`img: ${usage.input_image_tokens.toLocaleString()} tok`);
    else if (usage.input_image_count != null) media.push(`${usage.input_image_count} img`);
    if (usage.input_video_tokens != null) media.push(`video: ${usage.input_video_tokens.toLocaleString()} tok`);
    else if (usage.input_video_count != null) media.push(`${usage.input_video_count} video`);
    if (media.length) line += `<br><span style="color:#8bc34a">${media.join(' · ')}</span>`;
    usageHTML = `<div style="margin-top: 4px; font-size: 0.8em; color: #999;">${line}</div>`;
  }

  const thinkingHTML = (usage && usage.thinking) ? `
<details class="chat-thinking">
  <summary>💭 Thinking (${usage.thinking.length.toLocaleString()} chars)</summary>
  <div class="chat-thinking-body">${escapeHTML(usage.thinking)}</div>
</details>` : '';

  pendingAiMsgDiv.innerHTML = `<div class="chat-msg-role">AI &rarr; Step ${step+1}</div>
${thinkingHTML}<div class="chat-msg-text">${escapeHTML(raw_response || '')}</div>
<div style="margin-top: 8px; padding: 6px 10px; background: ${actionBg}; border-radius: 4px; font-size: 0.9em; color: ${actionColor}; border: 1px solid ${actionBorder};">
  <strong>Parsed action:</strong> ${is_valid ? `${parsed_action}${actionMeaningStr}` : '(invalid)'}
</div>${usageHTML}`;

  pendingAiMsgDiv = null;
  const container = document.getElementById('chat-messages');
  container.scrollTop = container.scrollHeight;
}

function appendChatTurn(data) {
  const container = document.getElementById('chat-messages');
  const { prompt, frame_b64, video_mode, context_videos_b64, context_frames } = data.chat;
  const isInit = data.step === 0 || data.step === 'init';

  // --- User message ---
  const userMsg = document.createElement('div');
  userMsg.className = 'chat-msg chat-msg-user';

  // Media content: 分别显示 demo 和 initial frame
  let mediaHTML = '<div class="chat-frames">';

  // 1. 显示 context (demo)
  if (isInit) {
    if (video_mode && context_videos_b64 && context_videos_b64.length > 0) {
      // Video mode: 显示 demo 视频（可能多段）
      context_videos_b64.forEach((vid_b64, vi) => {
        const contextVideoId = `chat-context-video-${Date.now()}-${vi}`;
        mediaHTML += `
          <video id="${contextVideoId}" controls style="max-width: 400px; max-height: 300px; border-radius: 4px; margin-bottom: 8px;">
            <source src="data:video/mp4;base64,${vid_b64}" type="video/mp4">
            Your browser does not support video playback.
          </video>
          <div style="font-size: 10px; color: #888; margin-bottom: 12px;">
            📹 Demo video ${context_videos_b64.length > 1 ? '#' + (vi + 1) : ''} (context)
          </div>
        `;
      });
    } else if (context_frames && context_frames.length > 0) {
      // Image mode: 显示 demo 图片
      context_frames.forEach((f, i) => {
        mediaHTML += `<img class="chat-thumb chat-thumb-ctx" src="data:image/jpeg;base64,${f}"
          title="Context frame ${i + 1}" onclick="showLightbox(this.src)">`;
      });
      mediaHTML += '<div style="font-size: 10px; color: #888; margin-top: 4px; margin-bottom: 12px;">📸 Demo frames (context)</div>';
    }
  }

  // 2. 显示 initial/current frame
  if (frame_b64) {
    mediaHTML += `<img class="chat-thumb" src="data:image/jpeg;base64,${frame_b64}"
      title="Step ${data.step} frame" onclick="showLightbox(this.src)" style="border: 2px solid #64b5f6;">`;
    mediaHTML += `<div style="font-size: 10px; color: #888; margin-top: 4px;">
      ${isInit ? '🎯 Initial state (what AI plays)' : '📍 Current state'}
    </div>`;
  }

  mediaHTML += '</div>';

  userMsg.innerHTML = `<div class="chat-msg-role">User &mdash; Step ${data.step+1}</div>
${mediaHTML}<div class="chat-msg-text">${escapeHTML(prompt || '')}</div>`;

  // --- AI message ---
  const aiMsg = document.createElement('div');
  aiMsg.className = 'chat-msg chat-msg-ai';
  const actionMeaning = data.action_meaning ? ` - ${data.action_meaning}` : '';
  aiMsg.innerHTML = `<div class="chat-msg-role">AI &rarr; Step ${data.step+1}</div>
<div class="chat-msg-text">${escapeHTML(data.raw_response || '')}</div>
<div style="margin-top: 8px; padding: 6px 10px; background: #f5f5f5; border-radius: 4px; font-size: 0.9em; color: #666;">
  <strong>Parsed action:</strong> ${data.parsed_action}${actionMeaning}
</div>`;

  const turn = document.createElement('div');
  turn.className = 'chat-turn';
  turn.appendChild(userMsg);
  turn.appendChild(aiMsg);
  container.appendChild(turn);
  container.scrollTop = container.scrollHeight;

  chatMsgCount++;
  updateChatTitle();
}

function escapeHTML(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function showRetryDialog(message) {
  appendAIError(message);
  // 在 AI log 区域内嵌入持久化 Retry/Stop 按钮（不用 confirm 弹窗，避免在非活动标签页丢失）
  const container = document.createElement('div');
  container.style.cssText = 'display:flex;gap:8px;padding:8px;background:#2a1a00;border:1px solid #e65100;border-radius:6px;margin:6px 0;align-items:center;';
  container.innerHTML = `
    <span style="flex:1;color:#ffb74d;font-size:12px;">API Error — waiting for your decision</span>
    <button id="btn-ai-retry" style="padding:4px 14px;background:#2e7d32;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:12px;">Retry</button>
    <button id="btn-ai-stop" style="padding:4px 14px;background:#c62828;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:12px;">Stop</button>
  `;
  const logEl = document.getElementById('ai-log');
  logEl.appendChild(container);
  logEl.scrollTop = logEl.scrollHeight;

  container.querySelector('#btn-ai-retry').onclick = () => {
    container.remove();
    ws.send(JSON.stringify({ type: 'ai_retry' }));
  };
  container.querySelector('#btn-ai-stop').onclick = () => {
    container.remove();
    ws.send(JSON.stringify({ type: 'ai_stop' }));
    document.getElementById('ai-btn-start').disabled = false;
  };
}

// ========== Undo / Save / Context Folder ==========
