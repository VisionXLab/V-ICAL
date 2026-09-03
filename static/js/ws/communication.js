// ========== WebSocket ==========
let _wsPingTimer = null;

function connectWS() {
  if (ws) ws.close();
  _clearWsPing();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/${sessionId}`);

  let opened = false;
  const readyPromise = new Promise((resolve, reject) => {
    ws.addEventListener('open', () => {
      opened = true;
      _startWsPing();
      resolve();
    }, { once: true });
    ws.addEventListener('close', () => {
      _clearWsPing();
      if (!opened) reject(new Error('WebSocket closed before opening'));
      // Auto-reconnect if we still have an active session
      if (sessionId && isLive) {
        console.log('[WS] Connection lost, reconnecting in 2s...');
        setTimeout(() => {
          if (sessionId && isLive && (!ws || ws.readyState > WebSocket.OPEN)) {
            connectWS();
          }
        }, 2000);
      } else {
        ws = null;
      }
    }, { once: true });
  });

  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    handleWSMessage(data);
  };
  ws.onerror = (e) => { console.error('WS error', e); };

  return readyPromise;
}

function _startWsPing() {
  _clearWsPing();
  _wsPingTimer = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping' }));
    }
  }, 30000);  // 每 30 秒发一次心跳
}

function _clearWsPing() {
  if (_wsPingTimer) {
    clearInterval(_wsPingTimer);
    _wsPingTimer = null;
  }
}

function handleWSMessage(data) {
  switch (data.type) {
    case 'pong':
      break;  // 心跳响应，忽略

    case 'frame':
      // 保存AI游戏运行时的帧（仅AI模式且非历史帧请求）
      if (currentMode === 'ai' && data.index === undefined) {
        aiSessionFrames.push(data.frame);
      }

      // 历史帧请求（有 index）：总是显示
      // 实时帧（无 index）：仅在 isLive 时显示
      if (data.index !== undefined) {
        showFrame(data.frame);
        updateProgress(data.index, data.total || totalFrames);
      } else {
        if (data.state && data.state.total_frames) totalFrames = data.state.total_frames;
        if (isLive) {
          showFrame(data.frame);
          updateProgress(totalFrames - 1, totalFrames);
        }
      }
      if (data.state) updateState(data.state);
      break;

    case 'ai_prompt':
      appendChatPrompt(data);
      break;

    case 'ai_response':
      updateChatResponse(data);
      appendAILog(data);
      break;

    case 'ai_game_over':
      appendAILog({
        step: data.total_steps,
        raw_response: `Game Over! Final reward: ${data.final_reward}, Steps: ${data.total_steps}`,
        parsed_action: '-',
      });
      document.getElementById('ai-btn-start').disabled = false;
      alert(`Game Over!\nFinal reward: ${data.final_reward}\nTotal steps: ${data.total_steps}`);
      break;

    case 'ai_retry':
      appendAIRetry(data);
      break;

    case 'ai_error_recoverable':
      showRetryDialog(data.message);
      break;

    case 'ai_error':
      appendAIError(data.message);
      break;

    case 'replay_start':
      isReplaying = true;
      document.getElementById('btn-undo').disabled = true;
      document.getElementById('btn-reset').disabled = true;
      document.getElementById('btn-start').disabled = true;
      break;

    case 'replay_done':
      isReplaying = false;
      document.getElementById('btn-undo').disabled = true;  // 序列不可撤销
      document.getElementById('btn-reset').disabled = false;
      document.getElementById('btn-start').disabled = false;
      // 通知等待回放完成的 aiStart()
      if (window._onReplayDone) {
        window._onReplayDone();
        window._onReplayDone = null;
      } else if (data.is_game_over) {
        alert('Sequence replay stopped early: game over at step ' + data.total_steps);
      }
      break;

    case 'error':
      alert('Server error: ' + data.message);
      break;
  }
}
