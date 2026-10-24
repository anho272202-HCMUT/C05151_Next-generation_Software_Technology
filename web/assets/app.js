/* Critic-Agent Explorer: tabs, trace player, knowledge graph, results, security playground. */
(function () {
  const $ = s => document.querySelector(s);
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const AXIS_VI = {assumptions: "Giả định ngầm", fairness: "Công bằng khi so sánh", data_reliability: "Độ tin cậy dữ liệu",
                   generalizability: "Tính tổng quát", reproducibility: "Tính tái lập"};
  const LEVEL_VI = {"read-only": "chỉ đọc", "reversible-write": "ghi hoàn tác được",
                    "irreversible-write": "ghi không hoàn tác", "code-execution": "thực thi mã"};
  const PRICE_IN = 0.15 / 1e6, PRICE_OUT = 0.6 / 1e6;  // gpt-4o-mini reference pricing
  const data = {};

  async function getJSON(path) {
    try { const r = await fetch(path, {cache: "no-store"}); return r.ok ? await r.json() : null; }
    catch { return null; }
  }

  // ------------------------------------------------------------------ tabs
  function showTab(name) {
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === name));
    document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
    if (name === "graph") renderGraph();
    if (history.replaceState) history.replaceState(null, "", "#" + name);
  }
  document.querySelectorAll("#tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
  document.querySelectorAll("[data-goto]").forEach(b => b.addEventListener("click", () => showTab(b.dataset.goto)));

  // ------------------------------------------------------------------ overview + architecture
  function renderOverview() {
    const axes = data.system?.axes || Arch.AXES.map(id => ({id, guide: ""}));
    $("#axes-cards").innerHTML = axes.map(a => `<div class="card axis-${a.id}"><b>${AXIS_VI[a.id]}</b>
      <div class="muted small">${esc(a.id)}</div><p>${esc(a.guide)}</p></div>`).join("");
  }

  function renderArchitecture() {
    Arch.render($("#arch-svg"), id => {
      document.querySelectorAll("#arch-svg .node").forEach(n => n.classList.toggle("selected", n.dataset.id === id));
      const [title, body, file] = Arch.nodeInfo(id) || [id, "", ""];
      const tool = id.startsWith("tool:") && data.system?.tools.find(t => t.name === id.slice(5));
      $("#arch-info").innerHTML = `<h3>${esc(title)}</h3><p>${esc(body)}</p>` +
        (tool ? `<p><span class="lvl lvl-${tool.level}">${LEVEL_VI[tool.level]}</span> · agent được gọi: <code>${tool.agents.join(", ")}</code>
          ${tool.needs_approval ? " · <b>cần người duyệt</b>" : ""}</p>` : "") +
        `<p class="muted small">Mã nguồn: <code>${esc(file)}</code></p>`;
    });
  }

  // ------------------------------------------------------------------ trace player
  const player = {run: null, idx: -1, playing: false, timer: null, live: false};

  function nodeFor(e) {
    if (e.type === "reflection") return "reflection";
    if (e.type === "tool_call" || e.type === "tool_denied") return "tool:" + (e.tool === "delete_annotation" ? "save_annotation" : e.tool);
    if (e.type === "guardrail_flag") return e.agent === "gateway" ? "gateway" : "ingest";
    if (["ingest", "summarizer", "orchestrator", "evaluator", "synthesizer", "gateway"].includes(e.agent)) return e.agent;
    if (e.agent?.startsWith("critic:")) return e.agent;
    if (e.agent === "baseline") return "summarizer";
    return null;
  }

  function fold(events, upto) {
    const s = {active: new Set(), done: new Set(), calls: 0, tin: 0, tout: 0, tools: 0, denied: 0, flags: 0, axes: {},
               card: null, plan: null, graph: "", ingest: null, end: null, start: null};
    for (let i = 0; i <= upto && i < events.length; i++) {
      const e = events[i], n = nodeFor(e);
      switch (e.type) {
        case "run_start": s.start = e; break;
        case "node_start":
          if (n) s.active.add(n);
          if (e.agent.startsWith("critic:")) {
            const a = e.agent.slice(7);
            s.axes[a] = {...(s.axes[a] || {}), status: "drafting", round: e.round};
          }
          break;
        case "node_end":
          if (n) { s.active.delete(n); s.done.add(n); }
          if (e.agent === "ingest") s.ingest = e;
          if (e.agent === "summarizer") s.card = e.card;
          if (e.agent === "orchestrator") s.plan = e.plan;
          break;
        case "llm_call": s.calls++; s.tin += e.input_tokens; s.tout += e.output_tokens; break;
        case "tool_call": s.tools++; break;
        case "tool_denied": s.denied++; break;
        case "guardrail_flag": s.flags++; break;
        case "critique": {
          const a = e.axis;
          s.axes[a] = {...(s.axes[a] || {}), status: e.no_issue ? "no_issue" : "proposed", round: e.round,
                       weakness: e.weakness, ids: e.evidence_ids, question: e.question, severity: e.severity,
                       fabricated: e.fabricated_ids};
          break;
        }
        case "evaluation":
          s.axes[e.axis] = {...(s.axes[e.axis] || {}), status: e.accepted ? "accepted" : "rejected", score: e.score,
                            feedback: e.feedback, grounded: e.grounded, supported: e.supported};
          break;
        case "reflection": e.axes.forEach(a => { s.axes[a] = {...(s.axes[a] || {}), status: "retry"}; }); break;
        case "run_end": s.end = e; s.active.clear(); break;
      }
    }
    return s;
  }

  function narrate(e) {
    const ag = e.agent || "";
    switch (e.type) {
      case "run_start": return `Bắt đầu phân tích <b>${esc(e.paper)}</b> (seed ${e.seed}${e.mode ? ", chế độ " + e.mode : ""}). Mỗi bước tiếp theo là một sự kiện mà hệ thống đã ghi lại.`;
      case "node_start": {
        if (ag === "ingest") return "<b>Ingest:</b> gọi MCP tool <code>pdf_ingest</code> để đọc PDF (chỉ đọc), tách thành câu có ID, bỏ chữ ẩn, che câu có dạng chỉ thị, rồi lập chỉ mục BM25 kết hợp vector.";
        if (ag === "summarizer") return "<b>Summarizer</b> tìm đoạn mô tả thực nghiệm rồi nhờ LLM lập paper card (benchmark, baseline, model, claim).";
        if (ag === "orchestrator") return "<b>Orchestrator</b> lập kế hoạch truy vấn cho 5 trục, tra đồ thị tri thức và lấy lessons từ bộ nhớ, rồi phát 5 critic chạy song song.";
        if (ag.startsWith("critic:")) return `<b>Critic ${esc(Arch.AXIS_LABEL[ag.slice(7)])}</b> (vòng ${e.round}) tìm bằng chứng cho trục của mình. Các critic chạy song song nên sự kiện của chúng xen kẽ nhau.`;
        if (ag === "evaluator") return `<b>Evaluator</b> (vòng ${e.round}) kiểm tra tự động các ID trích dẫn bằng <code>get_sentences</code>, rồi LLM chấm điểm mức đúng trục và mức được ủng hộ.`;
        if (ag === "synthesizer") return "<b>Synthesizer</b> chọn critique tốt nhất của mỗi trục, xếp hạng câu hỏi tranh luận và ghi lessons vào bộ nhớ.";
        return `Bắt đầu ${esc(ag)}.`;
      }
      case "node_end": {
        if (ag === "ingest") return `Đã đọc <b>${e.pages}</b> trang → <b>${e.sentences}</b> câu, <b>${e.chunks}</b> đoạn chỉ mục. Guardrail cảnh báo: ${e.flags}.`;
        if (ag === "summarizer") return "Paper card đã sẵn sàng (xem khung “Paper card và kế hoạch”).";
        if (ag === "orchestrator") return `Kế hoạch đã xong cho 5 trục. Tra đồ thị: ${e.graph_lookup ? "có dữ liệu" : "không có"} · lessons dùng: ${e.lessons ?? 0}.`;
        if (ag.startsWith("critic:")) return `Critic ${esc(Arch.AXIS_LABEL[ag.slice(7)])} đã gom ${e.evidence} câu bằng chứng (≈${e.evidence_tokens} token, bộ nhớ ngắn hạn).`;
        if (ag === "evaluator") return `Evaluator xong vòng ${e.round}: nhận ${e.accepted}, sẽ thử lại: ${e.retry?.length ? e.retry.join(", ") : "không"}.`;
        if (ag === "synthesizer") return `Tổng hợp xong: <b>${e.accepted}/5</b> critique được chấp nhận sau ${e.rounds} vòng.`;
        return `Kết thúc ${esc(ag)}.`;
      }
      case "llm_call": return `<b>${esc(ag)}</b> gọi LLM: ${e.input_tokens} token vào, ${e.output_tokens} token ra, ${e.latency_s}s.`;
      case "llm_error": return `<b>${esc(ag)}</b>: lời gọi LLM lỗi (${esc(e.error)}). Hệ thống tự thử lại với seed khác.`;
      case "tool_call": return `<b>${esc(ag)}</b> gọi MCP tool <code>${esc(e.tool)}</code> <span class="lvl lvl-${e.level}">${LEVEL_VI[e.level] || e.level}</span> → gateway: <b>${esc(e.decision)}</b> (${e.latency_s}s).`;
      case "tool_denied": return `<b>Gateway CHẶN</b> <code>${esc(e.tool)}</code> mà ${esc(ag)} yêu cầu: ${e.reason === "least-privilege" ? "agent này không có quyền" : "không có người duyệt"}.`;
      case "guardrail_flag": return `<b>Guardrail</b> đã vô hiệu hoá nội dung nghi là chỉ thị (nguồn: ${esc(e.source)}, trang ${e.page ?? "-"}): “${esc(e.snippet)}”`;
      case "critique": return e.no_issue ? `Critic ${esc(Arch.AXIS_LABEL[e.axis])}: không tìm thấy vấn đề.` :
        `Critic <b>${esc(Arch.AXIS_LABEL[e.axis])}</b> đề xuất: “${esc(e.weakness)}” · trích dẫn <code>${(e.evidence_ids || []).join(", ") || "—"}</code>` +
        (e.fabricated_ids?.length ? ` · <span class="bad">ID không có trong bằng chứng: ${e.fabricated_ids.join(", ")}</span>` : "");
      case "evaluation": return `Evaluator chấm <b>${esc(Arch.AXIS_LABEL[e.axis])}</b>: ${e.score}/10 · grounded=${e.grounded} · supported=${e.supported} → <b class="${e.accepted ? "good" : "bad"}">${e.accepted ? "CHẤP NHẬN" : "LOẠI"}</b>${e.feedback ? " · feedback: “" + esc(e.feedback) + "”" : ""}`;
      case "reflection": return `<b>Reflection có mục tiêu</b> (vòng ${e.round}): chỉ chạy lại ${e.axes.map(a => Arch.AXIS_LABEL[a]).join(", ")}, kèm feedback của Evaluator.`;
      case "run_end": return `Hoàn tất sau <b>${e.latency_s}s</b> · ${e.cost.calls} lời gọi LLM · ${e.cost.input_tokens + e.cost.output_tokens} token · chi phí API thực tế $${e.cost.usd} (model local).`;
      default: return esc(e.type);
    }
  }

  const SHORT = {run_start: "bắt đầu", node_start: "▶", node_end: "■", llm_call: "LLM", llm_error: "LLM lỗi", tool_call: "tool",
                 tool_denied: "CHẶN", guardrail_flag: "guardrail", critique: "critique", evaluation: "chấm", reflection: "reflection",
                 run_end: "kết thúc", tool_error: "tool lỗi"};

  function logLine(e) {
    const what = e.tool ? e.tool : e.axis ? Arch.AXIS_LABEL[e.axis] : "";
    return `<span class="t">${e.t.toFixed(1)}s</span><span class="ty ty-${e.type}">${SHORT[e.type] || e.type}</span>${esc(e.agent || "")} ${esc(what)}`;
  }

  function renderAt(i) {
    const run = player.run;
    if (!run) return;
    player.idx = Math.max(-1, Math.min(i, run.events.length - 1));
    const ev = run.events, s = fold(ev, player.idx), cur = ev[player.idx];
    $("#scrub").max = Math.max(0, ev.length - 1);
    $("#scrub").value = Math.max(0, player.idx);
    $("#step-label").textContent = `Bước ${player.idx + 1} / ${ev.length}`;
    const curNode = cur && nodeFor(cur);
    document.querySelectorAll("#demo-svg .node").forEach(g => {
      const id = g.dataset.id;
      g.classList.toggle("active", s.active.has(id));
      g.classList.toggle("done", s.done.has(id) && !s.active.has(id));
      g.classList.toggle("pulse", id === curNode);
      g.classList.toggle("denied", cur?.type === "tool_denied" && id === curNode);
      if (id.startsWith("critic:")) {
        const st = s.axes[id.slice(7)]?.status;
        g.querySelector(".badge").textContent = {accepted: "✓", rejected: "✗", retry: "↻", drafting: "…", proposed: "?", no_issue: "∅"}[st] || "";
        g.dataset.status = st || "";
      }
    });
    document.querySelectorAll("#demo-svg .edge").forEach(l => l.classList.toggle("lit",
      !!curNode && (l.dataset.to === curNode || (l.dataset.from === "orchestrator" && curNode.startsWith("critic:")))));
    $("#narration").innerHTML = cur ? narrate(cur) : "Bấm <b>▶ Phát</b> để bắt đầu, hoặc <b>▶|</b> để đi từng bước.";
    const log = $("#event-log");
    log.innerHTML = ev.slice(0, player.idx + 1).map(e => `<li data-i="${e.seq}" class="${e.seq === player.idx ? "cur" : ""}">${logLine(e)}</li>`).join("");
    log.scrollTop = log.scrollHeight;
    $("#event-detail").innerHTML = cur ? detailHTML(cur) : "";
    renderStats(s);
    renderBoard(s, run);
    renderPlan(s);
    renderFinal(s, run);
  }

  function detailHTML(e) {
    const rows = Object.entries(e).filter(([k]) => !["seq"].includes(k)).map(([k, v]) => {
      const val = typeof v === "object" ? `<pre>${esc(JSON.stringify(v, null, 1))}</pre>` : esc(v);
      return `<tr><th>${esc(k)}</th><td>${val}</td></tr>`;
    });
    return `<table class="kv">${rows.join("")}</table>`;
  }

  function renderStats(s) {
    const usd = (s.tin * PRICE_IN + s.tout * PRICE_OUT).toFixed(4);
    const t = player.idx >= 0 ? player.run.events[player.idx].t.toFixed(1) : "0.0";
    $("#stats").innerHTML = [["Thời gian", t + " s"], ["Lời gọi LLM", s.calls], ["Token", (s.tin + s.tout).toLocaleString()],
      ["Tool call", s.tools], ["Bị chặn", s.denied], ["Guardrail", s.flags], ["Chi phí quy đổi*", "$" + usd]]
      .map(([k, v]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("") +
      `<div class="stat note">*theo giá gpt-4o-mini; chi phí thật là $0 vì chạy model local</div>`;
  }

  function renderBoard(s, run) {
    const finalCrits = Object.fromEntries((s.end && run.final?.critiques || []).map(c => [c.axis, c]));
    $("#axis-board").innerHTML = Arch.AXES.map(a => {
      const x = s.axes[a] || {}, f = finalCrits[a];
      const st = x.status || "pending";
      const label = {pending: "chờ", drafting: "đang viết", proposed: "chờ chấm", accepted: "chấp nhận", rejected: "bị loại",
                     retry: "chạy lại", no_issue: "không có vấn đề"}[st];
      const quotes = f?.quotes?.length ? `<div class="quotes">${f.quotes.map(q => `<div><code>${esc(q.id)}</code> “${esc(q.text)}”</div>`).join("")}</div>` : "";
      return `<div class="axis-card st-${st}"><div class="axis-head"><b>${AXIS_VI[a]}</b><span class="pill">${label}${x.round !== undefined ? " · vòng " + x.round : ""}${x.score !== undefined ? " · " + x.score + "/10" : ""}</span></div>
        <div class="w">${esc(f?.weakness || x.weakness || "")}</div>
        ${x.ids?.length ? `<div class="muted small">trích dẫn: <code>${x.ids.join(", ")}</code></div>` : ""}
        ${x.feedback && st !== "accepted" ? `<div class="fb">Evaluator: ${esc(x.feedback)}</div>` : ""}${quotes}</div>`;
    }).join("");
  }

  function renderPlan(s) {
    let html = "";
    if (s.card) {
      const c = s.card;
      html += `<h4>Paper card (Summarizer)</h4><table class="kv">` + ["title", "task", "benchmarks", "baselines", "models", "claims", "runs_or_seeds"]
        .filter(k => c[k]).map(k => `<tr><th>${k}</th><td>${esc(Array.isArray(c[k]) ? c[k].join("; ") : c[k])}</td></tr>`).join("") + "</table>";
    }
    if (s.plan) {
      html += `<h4>Kế hoạch truy vấn (Orchestrator + bổ sung theo paper card)</h4><table class="kv">` +
        Object.entries(s.plan).map(([a, p]) => `<tr><th>${AXIS_VI[a]}</th><td>${p.queries.map(q => `<span class="q">${esc(q)}</span>`).join(" ")}</td></tr>`).join("") + "</table>";
    }
    const g = player.run?.final?.graph_facts;
    if (s.plan && g) html += `<h4>Tri thức từ đồ thị</h4><pre class="graphfacts">${esc(g)}</pre>`;
    $("#plan-view").innerHTML = html || `<p class="muted">Xuất hiện sau bước Summarizer/Orchestrator.</p>`;
  }

  function renderFinal(s, run) {
    const f = run.final;
    if (!s.end || !f) { $("#final-report").innerHTML = `<p class="muted">Báo cáo hiện ra khi lượt chạy kết thúc.</p>`; return; }
    const qs = (f.debate_questions || []).map(q => `<li>${esc(q)}</li>`).join("");
    $("#final-report").innerHTML = `<h3>${esc(f.title)}</h3>
      <p>${f.mode === "baseline" ? "Baseline 1 prompt (không có Evaluator)" : `Được chấp nhận: <b>${f.accepted}/5</b> · số vòng: ${f.rounds} · lượt thử: ${f.attempts}`}</p>
      ${f.critiques.map(c => `<div class="crit"><b>${AXIS_VI[c.axis]}</b> <span class="pill">${c.accepted ? "chấp nhận" : c.grounded ? "có trích dẫn" : "chưa kiểm chứng"}${c.score != null ? " · " + c.score + "/10" : ""}</span>
        <p>${esc(c.weakness)}</p>${(c.quotes || []).map(q => `<blockquote><code>${esc(q.id)}, tr.${q.page}</code> ${esc(q.text)}</blockquote>`).join("")}
        ${c.question ? `<p class="muted">Câu hỏi tranh luận: ${esc(c.question)}</p>` : ""}</div>`).join("")}
      ${qs ? `<h4>3 câu hỏi tranh luận hàng đầu</h4><ol>${qs}</ol>` : ""}`;
  }

  function delayFor(i) {
    const ev = player.run.events;
    if (i + 1 >= ev.length) return 0;
    const dt = Math.max(0, ev[i + 1].t - ev[i].t);
    return Math.min(1800, Math.max(260, dt * 70)) / Number($("#speed").value);
  }

  function play() {
    if (!player.run) return;
    player.playing = true;
    $("#btn-play").textContent = "⏸ Tạm dừng";
    const tick = () => {
      if (!player.playing) return;
      if (player.idx >= player.run.events.length - 1) { if (!player.live) pause(); return; }
      renderAt(player.idx + 1);
      player.timer = setTimeout(tick, delayFor(player.idx));
    };
    tick();
  }

  function pause() {
    player.playing = false;
    clearTimeout(player.timer);
    $("#btn-play").textContent = "▶ Phát";
  }

  async function loadRun(runId) {
    pause();
    const run = await getJSON(`data/runs/${runId}.json`);
    if (!run) { $("#narration").textContent = "Không tải được trace."; return; }
    player.run = run;
    player.live = false;
    renderAt(-1);
  }

  function setupPlayer() {
    Arch.render($("#demo-svg"));
    const sel = $("#run-select");
    const idx = data.index || [];
    const label = r => `${r.paper} · ${r.mode === "full" ? "Full multi-agent" : "Baseline 1 prompt"} · ${r.latency_s?.toFixed(0)}s · ${r.llm_calls} lời gọi LLM`;
    sel.innerHTML = idx.map(r => `<option value="${r.run_id}">${esc(label(r))}</option>`).join("");
    sel.addEventListener("change", () => loadRun(sel.value));
    $("#btn-play").addEventListener("click", () => player.playing ? pause() : play());
    $("#btn-next").addEventListener("click", () => { pause(); renderAt(player.idx + 1); });
    $("#btn-back").addEventListener("click", () => { pause(); renderAt(player.idx - 1); });
    $("#btn-restart").addEventListener("click", () => { pause(); renderAt(-1); });
    $("#scrub").addEventListener("input", e => { pause(); renderAt(Number(e.target.value)); });
    $("#event-log").addEventListener("click", e => { const li = e.target.closest("li"); if (li) { pause(); renderAt(Number(li.dataset.i)); } });
    $("#model-name").textContent = data.system?.model || "qwen2.5:3b";
    const first = idx.find(r => r.mode === "full") || idx[0];
    if (first) { sel.value = first.run_id; loadRun(first.run_id); }
    else $("#narration").textContent = "Chưa có trace nào được export (chạy scripts/export_web_data.py).";
  }

  // ------------------------------------------------------------------ live mode (local server only)
  async function setupLive() {
    const health = await getJSON("api/health");
    if (!health) return;
    const papers = await getJSON("api/papers") || [];
    $("#live-paper").innerHTML = papers.map(p => `<option>${esc(p)}</option>`).join("");
    $("#live-box").classList.remove("hidden");
    $("#live-start").addEventListener("click", () => {
      pause();
      player.run = {run_id: "live", events: [], final: null};
      player.live = true;
      renderAt(-1);
      $("#live-start").disabled = true;
      $("#narration").innerHTML = "Đang khởi động lượt chạy LIVE trên máy của bạn…";
      const src = new EventSource(`api/run?paper=${encodeURIComponent($("#live-paper").value)}&seed=1`);
      src.onmessage = msg => {
        const e = JSON.parse(msg.data);
        if (e.type === "final") { player.run.final = e.final; renderAt(player.idx); src.close(); $("#live-start").disabled = false; return; }
        if (e.type === "error") { $("#narration").textContent = "Lỗi: " + e.error; src.close(); $("#live-start").disabled = false; return; }
        player.run.events.push(e);
        renderAt(player.run.events.length - 1);
      };
      src.onerror = () => { src.close(); $("#live-start").disabled = false; };
    });
  }

  // ------------------------------------------------------------------ knowledge graph
  let cy = null;
  const EDGE_COLORS = {CITES: "#888780", USES_BENCHMARK: "#1D9E75", COMPARES_WITH: "#378ADD", CONTRADICTS: "#E24B4A"};
  function renderGraph() {
    if (cy || !data.graph || !window.cytoscape) {
      if (!data.graph) $("#graph-info").innerHTML = "<p class='muted'>Chưa có đồ thị (chạy graph_builder rồi export).</p>";
      return;
    }
    const nodes = data.graph.nodes.map(n => ({data: {...n, label: n.type === "paper" ? (n.aliases?.[0] || n.id) : n.title}}));
    const edges = data.graph.edges.map((e, i) => ({data: {id: "e" + i, source: e.source, target: e.target, type: e.type, note: e.note || "",
                                                          verified: e.verified !== false}}));
    const textColor = getComputedStyle(document.documentElement).getPropertyValue("--text").trim() || "#23221f";
    cy = cytoscape({
      container: $("#cy"), elements: [...nodes, ...edges], wheelSensitivity: 0.2,
      // Papers on the inner ring, benchmarks on the outer ring: stable and readable for a small graph.
      layout: {name: "concentric", concentric: n => (n.data("type") === "paper" ? 2 : 1), levelWidth: () => 1,
               minNodeSpacing: 55, padding: 30, animate: false},
      minZoom: 0.4, maxZoom: 2.5,
      style: [
        {selector: "node", style: {label: "data(label)", "font-size": 12, color: textColor, "text-valign": "bottom",
                                   "text-margin-y": 5, "text-outline-width": 0}},
        {selector: "node[type='paper']", style: {"background-color": "#7F77DD", width: 22, height: 22, "font-weight": 600}},
        {selector: "node[type='benchmark']", style: {"background-color": "#5DCAA5", shape: "diamond", width: 13, height: 13,
                                                     "font-size": 10}},
        {selector: "edge", style: {width: 1.6, "curve-style": "bezier", "target-arrow-shape": "triangle", "arrow-scale": 0.8,
                                   "line-color": e => EDGE_COLORS[e.data("type")], "target-arrow-color": e => EDGE_COLORS[e.data("type")]}},
        {selector: "edge[type='CONTRADICTS']", style: {width: 3}},
        {selector: ".hidden", style: {display: "none"}},
        {selector: ":selected", style: {"border-width": 3, "border-color": "#D85A30"}},
      ],
    });
    $("#graph-filters").innerHTML = Object.entries(EDGE_COLORS).map(([t, c]) =>
      `<label><input type="checkbox" data-type="${t}" checked><i class="sw" style="background:${c}"></i>${t}</label>`).join("");
    $("#graph-filters").addEventListener("change", e => {
      const t = e.target.dataset.type;
      cy.edges(`[type='${t}']`).toggleClass("hidden", !e.target.checked);
    });
    cy.on("tap", "node", ev => {
      const d = ev.target.data();
      const nb = ev.target.connectedEdges().map(e => {
        const out = e.data("source") === d.id, other = cy.getElementById(out ? e.data("target") : e.data("source")).data();
        return `<li><span class="rel" style="color:${EDGE_COLORS[e.data("type")]}">${e.data("type")}</span> ${out ? "→" : "←"} ${esc(other.label)}${e.data("note") ? `<div class="muted small">${esc(e.data("note"))}</div>` : ""}</li>`;
      }).join("");
      const p = d.ref_profile;
      const bar = p ? `<div class="refbar">${[["peer_reviewed", "#534AB7"], ["preprint", "#EF9F27"], ["web", "#1D9E75"], ["other", "#B4B2A9"]]
        .map(([k, c]) => `<span style="flex:${p[k]};background:${c}" title="${k}: ${p[k]}"></span>`).join("")}</div>
        <div class="muted small">${p.total} tài liệu tham khảo · ${p.peer_reviewed} peer-reviewed (tím) · ${p.preprint} preprint (cam) · ${p.web} web (xanh) · ${p.other} khác · năm trung vị ${p.median_year}</div>` : "";
      $("#graph-info").innerHTML = `<h3>${esc(d.title)}</h3>${d.venue ? `<p><b>${esc(d.venue)}</b> · ${d.year || ""}</p>` : ""}${bar}<ul class="nb">${nb}</ul>`;
    });
  }

  // ------------------------------------------------------------------ results
  function renderResults() {
    const s = data.summary, el = $("#results-body");
    if (!s || !s.configs?.full) {
      el.innerHTML = `<div class="panel"><p><b>Đánh giá đang chạy.</b> Bộ đánh giá gồm 40 task (8 bài × 5 trục), so sánh Full với baseline 1 prompt qua 3 seed.
        Kết quả sẽ hiện ở đây sau khi chạy <code>python -m critic_agent.evaluation summarize</code> và export lại dữ liệu web.</p>
        <p class="muted">Cách tính: một task được coi là <i>giải được</i> khi critique cho trục đó (1) có trích dẫn kiểm chứng được và (2) khớp với điểm yếu nhóm đã gán nhãn
        (LLM-judge, đối chiếu thêm bằng từ khoá). Độ tin cậy của judge được kiểm tra trên 30 mẫu do người gán nhãn.</p></div>`;
      return;
    }
    const names = {full: "Full multi-agent", baseline: "Baseline 1 prompt"};
    const cfgs = Object.keys(s.configs);
    const kpi = (c, k, scale = 100, unit = "%") => `${(s.configs[c][k].mean * scale).toFixed(1)}${unit} ± ${(s.configs[c][k].std * scale).toFixed(1)}`;
    el.innerHTML = `<div class="cards ${cfgs.length === 2 ? "two" : "one"}">${cfgs.map(c => `<div class="card"><b>${names[c]}</b>
      <div class="big">${kpi(c, "solved_judge")}</div><div class="muted">task giải được (judge) · keyword: ${kpi(c, "solved_keyword")}</div>
      <p>Grounded: ${kpi(c, "grounded")} · Độ phủ: ${kpi(c, "covered")}<br>Độ trễ: ${kpi(c, "latency_s", 1, "s")}/bài · ${kpi(c, "llm_calls", 1, "")} lời gọi LLM/bài</p></div>`).join("")}</div>
      <div class="demo-grid"><div class="panel"><div class="panel-title">So sánh tổng thể (mean qua 3 seed)</div><canvas id="ch-main"></canvas></div>
      <div class="panel"><div class="panel-title">Task giải được theo trục</div><canvas id="ch-axis"></canvas></div></div>
      <div class="panel"><div class="panel-title">Phân tích lỗi (tổng số task × seed)</div><canvas id="ch-err" height="90"></canvas></div>
      <div id="agree"></div>`;
    const colors = {full: "#7F77DD", baseline: "#B4B2A9"};
    new Chart($("#ch-main"), {type: "bar", data: {labels: ["Giải được (judge)", "Giải được (keyword)", "Grounded", "Độ phủ"],
      datasets: cfgs.map(c => ({label: names[c], backgroundColor: colors[c],
        data: ["solved_judge", "solved_keyword", "grounded", "covered"].map(k => +(s.configs[c][k].mean * 100).toFixed(1))}))},
      options: {scales: {y: {max: 100, ticks: {callback: v => v + "%"}}}}});
    new Chart($("#ch-axis"), {type: "bar", data: {labels: Arch.AXES.map(a => AXIS_VI[a]),
      datasets: cfgs.map(c => ({label: names[c], backgroundColor: colors[c], data: Arch.AXES.map(a => +(s.configs[c].per_axis_solved[a] * 100).toFixed(0))}))},
      options: {scales: {y: {max: 100, ticks: {callback: v => v + "%"}}}}});
    const cats = [["solved", "Giải được", "#1D9E75"], ["grounded_no_match", "Có trích dẫn, không khớp nhãn", "#EF9F27"],
                  ["ungrounded", "Trích dẫn không kiểm chứng được", "#E24B4A"], ["no_critique", "Không có critique", "#B4B2A9"]];
    new Chart($("#ch-err"), {type: "bar", data: {labels: cfgs.map(c => names[c]),
      datasets: cats.map(([k, l, col]) => ({label: l, backgroundColor: col, data: cfgs.map(c => s.configs[c].outcomes.judge[k] || 0)}))},
      options: {indexAxis: "y", scales: {x: {stacked: true}, y: {stacked: true}}}});
    const a = data.agreement || s.judge_agreement;
    $("#agree").innerHTML = a ? `<div class="panel"><div class="panel-title">Độ tin cậy của LLM-as-judge (${a.n_labelled} mẫu người gán nhãn)</div>
      <table class="kv"><tr><th>Judge đánh giá ↔ người</th><td>accuracy ${a.eval_judge_vs_human.accuracy} · Cohen's κ ${a.eval_judge_vs_human.kappa}</td></tr>
      <tr><th>Keyword ↔ người</th><td>accuracy ${a.keyword_vs_human.accuracy} · κ ${a.keyword_vs_human.kappa}</td></tr>
      <tr><th>Evaluator runtime ↔ người</th><td>accuracy ${a.runtime_evaluator_vs_human.accuracy} · κ ${a.runtime_evaluator_vs_human.kappa} (n=${a.runtime_evaluator_vs_human.n})</td></tr></table></div>`
      : `<div class="panel muted">Độ tin cậy của judge: đang chờ nhãn người (eval/human_labels.csv).</div>`;
  }

  // ------------------------------------------------------------------ security
  function renderSecurity() {
    const tools = data.system?.tools || [];
    $("#tools-table").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Tool</th><th>Mức quyền</th><th>Agent được gọi</th><th>Người duyệt</th><th>Mô tả / cơ chế chặn</th></tr></thead><tbody>
      ${tools.map(t => `<tr><td><code>${t.name}</code></td><td><span class="lvl lvl-${t.level}">${LEVEL_VI[t.level]}</span></td><td>${t.agents.join(", ")}</td>
      <td>${t.needs_approval ? "bắt buộc + HMAC" : "—"}</td><td>${esc(t.description)}</td></tr>`).join("")}</tbody></table></div>`;
    const patterns = (data.system?.injection_patterns || []).map(p => { try { return new RegExp(p, "i"); } catch { return null; } }).filter(Boolean);
    const run = () => {
      const text = $("#guard-input").value;
      const parts = text.split(/(?<=[.!?])\s+|\n/);
      $("#guard-output").innerHTML = parts.map(s => patterns.some(p => p.test(s)) ?
        `<span class="redacted" title="${esc(s)}">[REDACTED: suspected injected instruction]</span>` : esc(s)).join(" ") +
        `<p class="muted small">${parts.filter(s => patterns.some(p => p.test(s))).length} câu bị che · ${patterns.length} mẫu đang áp dụng</p>`;
    };
    $("#guard-input").addEventListener("input", run);
    run();
  }

  // ------------------------------------------------------------------ boot
  (async function boot() {
    [data.system, data.index, data.graph, data.summary, data.agreement] = await Promise.all(
      ["data/system.json", "data/index.json", "data/graph.json", "data/summary.json", "data/judge_agreement.json"].map(getJSON));
    renderOverview();
    renderArchitecture();
    setupPlayer();
    setupLive();
    renderResults();
    renderSecurity();
    const start = location.hash.slice(1);
    if (start && document.getElementById(start)) showTab(start);
  })();
})();
