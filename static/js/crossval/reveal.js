// ========== Cross-Validation Completion ==========
// 后端自动保存；前端仅解除 HUD 遮蔽并显示非阻塞提示。

function crossvalShowReveal(reason, passed, saveError = null) {
  if (typeof clearCrossvalHudMask === 'function') clearCrossvalHudMask();
  if (typeof crossvalMarkEvaluationComplete === 'function') {
    crossvalMarkEvaluationComplete(saveError);
  }
  showEvaluationSaveToast(saveError ? '保存失败' : '已经保存了', !!saveError);
}

async function crossvalEndAndReveal() {
  if (!sessionId) { alert('当前没有进行中的会话'); return; }
  if (!confirm('确定结束本次代入测试? 将立即揭晓 reward。')) return;
  try {
    const resp = await fetch(`/api/session/${sessionId}/save-evaluation`, { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  } catch (e) {
    showEvaluationSaveToast('保存失败', true);
    return;
  }
  crossvalShowReveal('manual_end', false);
}
