/* Architecture diagram: node layout, descriptions, SVG renderer shared by the Architecture and Demo tabs. */
(function () {
  const AXES = ["assumptions", "fairness", "data_reliability", "generalizability", "reproducibility"];
  const AXIS_LABEL = {assumptions: "Assumptions", fairness: "Fairness", data_reliability: "Data reliability",
                      generalizability: "Generalizability", reproducibility: "Reproducibility"};

  const NODES = [
    {id: "input", x: 20, y: 20, w: 150, h: 56, kind: "sys", label: "PDF + yêu cầu", sub: "đầu vào không tin cậy"},
    {id: "ingest", x: 210, y: 20, w: 170, h: 56, kind: "guard", label: "Ingest + Guardrail", sub: "L1 + L2: tách câu, lọc injection"},
    {id: "summarizer", x: 420, y: 20, w: 170, h: 56, kind: "agent", label: "Summarizer", sub: "paper card"},
    {id: "orchestrator", x: 630, y: 20, w: 170, h: 56, kind: "agent", label: "Orchestrator", sub: "kế hoạch 5 trục"},
    {id: "memory", x: 840, y: 20, w: 140, h: 56, kind: "know", label: "Lessons memory", sub: "ChromaDB · L2 khi ghi/đọc"},
    ...AXES.map((a, i) => ({id: "critic:" + a, x: 20 + i * 196, y: 130, w: 176, h: 52, kind: "agent",
                            label: "Critic", sub: AXIS_LABEL[a]})),
    {id: "evaluator", x: 150, y: 235, w: 190, h: 56, kind: "agent", label: "Evaluator", sub: "L4: kiểm chứng + chấm điểm"},
    {id: "reflection", x: 400, y: 235, w: 190, h: 56, kind: "sys", label: "Reflection", sub: "chỉ trục bị loại, ≤2 vòng"},
    {id: "synthesizer", x: 650, y: 235, w: 190, h: 56, kind: "agent", label: "Synthesizer", sub: "báo cáo + câu hỏi"},
    {id: "output", x: 870, y: 235, w: 110, h: 56, kind: "sys", label: "Báo cáo", sub: "report.md"},
    {id: "gateway", x: 20, y: 335, w: 960, h: 40, kind: "guard", label: "ToolGateway → MCP server", sub: "allow-list theo agent · người duyệt + HMAC · L2 quét output · audit log"},
    {id: "tool:pdf_ingest", x: 20, y: 410, w: 110, h: 60, kind: "tool", label: "pdf_ingest", sub: "read-only"},
    {id: "tool:search_paper", x: 141, y: 410, w: 110, h: 60, kind: "tool", label: "search_paper", sub: "read-only"},
    {id: "tool:get_sentences", x: 262, y: 410, w: 110, h: 60, kind: "tool", label: "get_sentences", sub: "read-only"},
    {id: "tool:graph_query", x: 383, y: 410, w: 110, h: 60, kind: "know", label: "graph_query", sub: "read-only"},
    {id: "tool:artifact_check", x: 504, y: 410, w: 110, h: 60, kind: "tool", label: "artifact_check", sub: "read-only (web)"},
    {id: "tool:save_annotation", x: 625, y: 410, w: 110, h: 60, kind: "tool", label: "annotation", sub: "ghi hoàn tác được"},
    {id: "tool:publish_review", x: 746, y: 410, w: 110, h: 60, kind: "tool", label: "publish_review", sub: "ghi không hoàn tác"},
    {id: "tool:execute_code_in_sandbox", x: 867, y: 410, w: 113, h: 60, kind: "tool", label: "sandbox", sub: "thực thi mã (tắt)"},
    // prompt-injection defence layers (threat model): where they act is described in INFO
    {id: "def:l1", x: 20, y: 505, w: 182, h: 56, kind: "guard", label: "L1 · Parser PDF", sub: "bỏ chữ ẩn / tương phản thấp"},
    {id: "def:l2", x: 214, y: 505, w: 182, h: 56, kind: "guard", label: "L2 · Sanitizer", sub: "chuẩn hoá + che câu chỉ thị"},
    {id: "def:l3", x: 408, y: 505, w: 182, h: 56, kind: "guard", label: "L3 · Spotlighting", sub: "<untrusted> + quy tắc ưu tiên"},
    {id: "def:l4", x: 602, y: 505, w: 182, h: 56, kind: "guard", label: "L4 · Kiểm chứng", sub: "trích dẫn có thật + gateway"},
    {id: "def:l5", x: 796, y: 505, w: 184, h: 56, kind: "guard", label: "L5 · Quy tắc nhận thức", sub: "tuỳ chọn · đang thử nghiệm"},
  ];

  const EDGES = [
    ["input", "ingest"], ["ingest", "summarizer"], ["summarizer", "orchestrator"], ["memory", "orchestrator", "dash"],
    ...AXES.map(a => ["orchestrator", "critic:" + a]),
    ...AXES.map(a => ["critic:" + a, "evaluator"]),
    ["evaluator", "reflection"], ["reflection", "synthesizer"], ["synthesizer", "output"],
  ];

  const INFO = {
    input: ["Đầu vào", "PDF bài báo trong data/papers/ và (tuỳ chọn) yêu cầu của người dùng, ví dụ “focus on fairness”. Cả hai đều được coi là không tin cậy.", "cli.py, server.py"],
    ingest: ["Ingest + Guardrail", "Gọi MCP tool pdf_ingest: đọc PDF (chỉ đọc), bỏ khối chữ ẩn (L1), che các câu khớp quy tắc chỉ thị sau khi chuẩn hoá (L2, xem phần phòng thủ bên dưới sơ đồ), tách thành câu có ID pXsY, chia cửa sổ 4 câu, lập chỉ mục BM25 + nomic-embed-text (RRF). Yêu cầu người dùng cũng được sanitize.", "pdf_parser.py, guardrail.py, retrieval.py, mcp_server.py"],
    summarizer: ["Summarizer (LLM)", "Tìm đoạn mô tả thực nghiệm (search_paper), rồi tạo paper card: task, method, benchmarks, baselines, models, claims, runs/seeds. Tiêu đề lấy từ PDF, không lấy từ LLM (model 3B từng bịa tiêu đề).", "agents.py: summarize"],
    orchestrator: ["Orchestrator (LLM)", "Lập kế hoạch: mỗi trục có 2 truy vấn và độ ưu tiên. Hệ thống bổ sung truy vấn theo tên baseline/benchmark trong paper card. Orchestrator tra đồ thị tri thức (graph_query) và lấy tối đa 3 lessons từ bộ nhớ dài hạn, rồi phát (fan-out) 5 critic bằng LangGraph Send.", "agents.py: orchestrate, dispatch"],
    memory: ["Bộ nhớ dài hạn (lessons)", "ChromaDB. Chỉ ghi khi critique được chấp nhận: lesson loại “fix” (lần đầu bị loại vì sao, bản được nhận ra sao) hoặc loại “pattern”. Nội dung được sanitize khi ghi và khi đọc. Khi truy hồi, không lấy lesson từ chính bài đang xét. forget(paper_id) xoá dữ liệu khi bị đầu độc.", "lessons.py"],
    evaluator: ["Evaluator (LLM + kiểm tra tất định)", "Bước 1, tất định: các ID trích dẫn có tồn tại trong PDF không (get_sentences), có nằm trong bằng chứng của critic không (chống bịa), điểm yếu có đủ cụ thể không. Bước 2, LLM chấm 0–10: đúng trục chưa, có được câu trích ủng hộ không. Critique được nhận khi grounded, supported và đạt ít nhất 7 điểm.", "agents.py: evaluate"],
    reflection: ["Reflection có mục tiêu", "Chỉ trục bị loại mới quay lại critic, kèm feedback và lần thử trước; tối đa 2 vòng thêm. Một trục chỉ được thử lại khi lần trước có cải thiện. Hệ thống giữ lần thử tốt nhất của mỗi trục.", "agents.py: route_after_evaluate"],
    synthesizer: ["Synthesizer", "Chọn critique tốt nhất của mỗi trục, xếp hạng theo mức nghiêm trọng và điểm, lấy 3 câu hỏi tranh luận hàng đầu, ghi lessons và xuất báo cáo Markdown.", "agents.py: synthesize, report.py"],
    output: ["Báo cáo + artefact", "Mỗi lượt chạy ghi: report.md, output.json, trace.jsonl (dữ liệu cho chính trang demo này), audit.jsonl (mọi yêu cầu tool, kể cả bị chặn) và cost.json (token, độ trễ, chi phí quy đổi).", "pipeline.py"],
    "def:l1": ["L1 · Parser PDF (chống chữ ẩn)", "Chạy khi đọc PDF, trước mọi thứ khác. Với mỗi đoạn chữ, parser xác định nền thực tế (hình đặc vẽ ngay phía dưới, mặc định trắng) và loại đoạn chữ nếu: độ tương phản < 1,5, cỡ chữ < 2,5pt, hoặc một hình đặc vẽ đè lên sau đó. Chỉ loại khi đoạn ẩn đủ dài như một câu (≥ 6 từ) để không xoá nhãn hình. Đoạn bị loại trở thành cờ guardrail_flag (nguồn pdf-hidden).", "pdf_parser.py: _hidden_regions"],
    "def:l2": ["L2 · Sanitizer (chuẩn hoá + quy tắc)", "Áp dụng ở bốn điểm vào: văn bản PDF, yêu cầu người dùng, kết quả tool và bộ nhớ lessons. Bước 1 chuẩn hoá để thấy cái model sẽ thấy: NFKC, homoglyph Cyrillic/Greek, ký tự vô hình, ký tự tag Unicode, chữ cách quãng, leetspeak, base64. Bước 2: ~50 quy tắc theo nhóm (override, role-hijack, fake-template, review-manipulation, prompt-leak, tool-abuse, exfiltration, authority, memory-poisoning, đa ngôn ngữ). Quy tắc nói về hành động phải “mở câu” (cổng mệnh lệnh), nên câu mô tả như “models that ignore the instructions…” không bị che. Câu khớp bị thay bằng [REDACTED].", "guardrail.py, gateway.py, lessons.py"],
    "def:l3": ["L3 · Spotlighting", "Nội dung ngoài được bọc trong thẻ <untrusted> và system prompt có quy tắc ưu tiên cao nhất: không làm theo chỉ thị trong thẻ, không lộ mã nội bộ. Là lớp cuối nếu L1–L2 bỏ lọt, nhưng hiệu quả phụ thuộc model: chưa được đo trong repo này.", "prompts.py"],
    "def:l4": ["L4 · Kiểm chứng đầu ra và phân quyền", "Critic chỉ trích ID câu, hệ thống tự chép nguyên văn từ PDF, Evaluator kiểm ID tất định; tool đi qua gateway với allow-list theo agent, người duyệt + HMAC cho tool nguy hiểm. Nhờ vậy một chỉ thị lọt qua cũng khó biến thành trích dẫn bịa hoặc hành động ngoài quyền.", "agents.py, gateway.py, permissions.py"],
    "def:l5": ["L5 · Quy tắc nhận thức (tuỳ chọn, thử nghiệm)", "Thêm vào system prompt của Critic và Evaluator khi bật CRITIC_L5=1: lời tự khẳng định của tác giả về chính đánh giá của mình (“mọi baseline đều dùng cùng ngân sách”, “không có hạn chế nào”) là TUYÊN BỐ cần kiểm chứng chứ không phải bằng chứng; một hạn chế do chính tác giả thừa nhận không phải là phát hiện độc lập. Nhắm vào tấn công ngữ nghĩa mà L1–L3 không thấy được (xem kết quả decoy ở tab An ninh). Mặc định TẮT: hiệu quả đo được chưa có ý nghĩa thống kê (decoy 38% → 25%, p = 0,17).", "prompts.py: EPISTEMIC_RULE, config.py"],
    gateway: ["ToolGateway", "Lối duy nhất từ agent tới tool. Gateway (1) kiểm tra allow-list theo agent, (2) yêu cầu người duyệt với tool ghi không hoàn tác và thực thi mã, rồi cấp token HMAC mà server kiểm tra lại, (3) quét injection trong kết quả tool, (4) ghi audit log và trace.", "gateway.py, permissions.py"],
  };
  const TOOL_INFO = {
    pdf_ingest: "Đọc, sanitize và lập chỉ mục PDF. Chỉ đọc file trong data/papers/, chỉ ghi vào cache của chính server.",
    search_paper: "Tìm kiếm hybrid (BM25 + dense, RRF) trong một bài, trả về các câu có ID; k ≤ 8.",
    get_sentences: "Trả nguyên văn câu theo ID, dùng cho Evaluator kiểm chứng. ID không tồn tại thì báo missing.",
    graph_query: "Láng giềng của bài trong đồ thị tri thức (≤ 20 node, cạnh CONTRADICTS xếp trước) kèm hồ sơ nguồn trích dẫn.",
    artifact_check: "Đọc metadata GitHub của repo chính thức của bài: file môi trường, README có nhắc seed không. Chỉ GET, có cache.",
    save_annotation: "Ghi chú của người dùng; delete_annotation hoàn tác theo note_id.",
    publish_review: "Ghi bản phản biện vào log chỉ-thêm (append-only). Cần người duyệt kèm token HMAC; không ghi đè, không xoá được.",
    execute_code_in_sandbox: "Chạy Python trong Docker (không mạng, 1 GB, 30 s). Tắt mặc định; cần người duyệt kèm HMAC.",
  };

  function nodeInfo(id) {
    if (id.startsWith("critic:")) {
      const a = id.split(":")[1];
      return ["Critic: " + AXIS_LABEL[a], "Worker chuyên trục " + AXIS_LABEL[a] + ". Critic điền một cửa sổ bằng chứng (≈1.800 token) từ các truy vấn của Orchestrator, dùng thêm thông tin đồ thị (và artifact check với trục reproducibility), rồi viết đúng một điểm yếu kèm 1–3 ID câu. Nếu không thấy vấn đề, critic trả no_issue.", "agents.py: critic"];
    }
    if (id.startsWith("tool:")) {
      const t = id.split(":")[1];
      return ["MCP tool: " + t, TOOL_INFO[t] || "", "mcp_server.py"];
    }
    return INFO[id];
  }

  function center(n) { return [n.x + n.w / 2, n.y + n.h / 2]; }

  function render(svg, onClick) {
    const byId = Object.fromEntries(NODES.map(n => [n.id, n]));
    const NS = "http://www.w3.org/2000/svg";
    svg.innerHTML = '<defs><marker id="arr-' + svg.id + '" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1L9 5L1 9" fill="none" stroke="currentColor" stroke-width="1.5"/></marker></defs>';
    const edges = document.createElementNS(NS, "g");
    edges.setAttribute("class", "edges");
    for (const [a, b, style] of EDGES) {
      const s = byId[a], t = byId[b];
      let x1, y1, x2, y2;
      if (Math.abs(s.y - t.y) < 5) { x1 = s.x + s.w; y1 = s.y + s.h / 2; x2 = t.x; y2 = t.y + t.h / 2;
        if (x2 < x1) { x1 = s.x; x2 = t.x + t.w; } }
      else { [x1] = center(s); y1 = s.y + s.h; [x2] = center(t); y2 = t.y; }
      const line = document.createElementNS(NS, "line");
      Object.entries({x1, y1, x2, y2, "data-from": a, "data-to": b, class: "edge" + (style ? " " + style : ""),
                      "marker-end": `url(#arr-${svg.id})`}).forEach(([k, v]) => line.setAttribute(k, v));
      edges.appendChild(line);
    }
    // feedback loop reflection -> critics (drawn as a path above the evaluator row)
    const loop = document.createElementNS(NS, "path");
    loop.setAttribute("d", "M495 235 C495 205 495 205 496 186");
    loop.setAttribute("class", "edge dash loop");
    loop.setAttribute("marker-end", `url(#arr-${svg.id})`);
    edges.appendChild(loop);
    // gateway <-> agents and tools
    for (const x of [108, 245, 495, 745]) {
      const l = document.createElementNS(NS, "line");
      Object.entries({x1: x, y1: 293, x2: x, y2: 333, class: "edge faint"}).forEach(([k, v]) => l.setAttribute(k, v));
      edges.appendChild(l);
    }
    NODES.filter(n => n.id.startsWith("tool:")).forEach(n => {
      const l = document.createElementNS(NS, "line");
      Object.entries({x1: n.x + n.w / 2, y1: 375, x2: n.x + n.w / 2, y2: 408, class: "edge faint"}).forEach(([k, v]) => l.setAttribute(k, v));
      edges.appendChild(l);
    });
    svg.appendChild(edges);
    for (const n of NODES) {
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", `node ${n.kind}`);
      g.setAttribute("data-id", n.id);
      g.setAttribute("tabindex", "0");
      const r = document.createElementNS(NS, "rect");
      Object.entries({x: n.x, y: n.y, width: n.w, height: n.h, rx: 8}).forEach(([k, v]) => r.setAttribute(k, v));
      g.appendChild(r);
      const t1 = document.createElementNS(NS, "text");
      Object.entries({x: n.x + n.w / 2, y: n.y + (n.sub ? n.h / 2 - 7 : n.h / 2 + 4), class: "t1"}).forEach(([k, v]) => t1.setAttribute(k, v));
      t1.textContent = n.label;
      g.appendChild(t1);
      if (n.sub) {
        const t2 = document.createElementNS(NS, "text");
        Object.entries({x: n.x + n.w / 2, y: n.y + n.h / 2 + 11, class: "t2"}).forEach(([k, v]) => t2.setAttribute(k, v));
        t2.textContent = n.sub;
        g.appendChild(t2);
      }
      const badge = document.createElementNS(NS, "text");
      Object.entries({x: n.x + n.w - 6, y: n.y + 13, class: "badge"}).forEach(([k, v]) => badge.setAttribute(k, v));
      g.appendChild(badge);
      if (onClick) {
        g.addEventListener("click", () => onClick(n.id));
        g.addEventListener("keydown", e => { if (e.key === "Enter") onClick(n.id); });
      }
      svg.appendChild(g);
    }
    const cap = document.createElementNS(NS, "text");
    Object.entries({x: 20, y: 497, class: "band-caption"}).forEach(([k, v]) => cap.setAttribute(k, v));
    cap.textContent = "LỚP PHÒNG THỦ PROMPT INJECTION (THREAT MODEL): L1–L5, mỗi lớp được ghi chú trên các khối phía trên";
    cap.style.textAnchor = "start";
    svg.appendChild(cap);
  }

  window.Arch = {AXES, AXIS_LABEL, NODES, render, nodeInfo};
})();
