/**
 * script.js
 * ---------
 * Handles:
 *  - Sending text queries to POST /ask
 *  - Rendering markdown-ish responses
 *  - Voice input via Web Speech API (speech-to-text only)
 *  - Source chunk accordion
 *  - Example query chips
 */

// ─── DOM Refs ────────────────────────────────────────────────────────────────

const chatBox      = document.getElementById("chat-box");
const queryInput   = document.getElementById("query-input");
const sendBtn      = document.getElementById("send-btn");
const voiceBtn     = document.getElementById("voice-btn");
const statusBar    = document.getElementById("status-bar");
const exampleChips = document.querySelectorAll(".example-chip");

// ─── State ───────────────────────────────────────────────────────────────────

let isListening   = false;
let recognition   = null;

// ─── Voice Setup (Web Speech API) ────────────────────────────────────────────

const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang        = "en-IN";   // Indian English
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    isListening = true;
    voiceBtn.classList.add("listening");
    voiceBtn.title = "Listening… click to stop";
    setStatus("🎙️ Listening… speak your question");
  };

  recognition.onresult = (event) => {
    const transcript = Array.from(event.results)
      .map((r) => r[0].transcript)
      .join("");
    queryInput.value = transcript;
    // Auto-send when the result is final
    if (event.results[event.results.length - 1].isFinal) {
      setTimeout(sendQuery, 300);
    }
  };

  recognition.onerror = (event) => {
    console.error("Speech error:", event.error);
    setStatus(`Voice error: ${event.error}. Try typing instead.`);
    stopListening();
  };

  recognition.onend = () => {
    stopListening();
  };
} else {
  voiceBtn.disabled = true;
  voiceBtn.title = "Voice not supported in this browser";
}

function startListening() {
  if (!recognition) return;
  queryInput.value = "";
  recognition.start();
}

function stopListening() {
  isListening = false;
  if (voiceBtn) {
    voiceBtn.classList.remove("listening");
    voiceBtn.title = "Voice input";
  }
  setStatus("");
}

voiceBtn.addEventListener("click", () => {
  if (isListening) {
    recognition.stop();
  } else {
    startListening();
  }
});

// ─── Send Query ───────────────────────────────────────────────────────────────

async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query) return;

  queryInput.value = "";
  queryInput.focus();

  appendMessage("user", query);
  const thinkingId = appendThinking();
  setStatus("Searching knowledge base…");

  try {
    const res = await fetch("/ask", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ query }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    removeThinking(thinkingId);
    appendBotMessage(data);
    setStatus(`✓ Answered in ${data.latency_ms} ms | ${data.source_chunks.length} source(s) retrieved`);
  } catch (err) {
    removeThinking(thinkingId);
    appendMessage("bot-error", `⚠️ ${err.message}`);
    setStatus("Error — please try again.");
  }
}

sendBtn.addEventListener("click", sendQuery);

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

// ─── Example Chips ───────────────────────────────────────────────────────────

exampleChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    queryInput.value = chip.dataset.query;
    sendQuery();
  });
});

// ─── DOM Helpers ─────────────────────────────────────────────────────────────

function appendMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  wrap.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatBox.appendChild(wrap);
  scrollBottom();
  return wrap;
}

function appendThinking() {
  const id = "thinking-" + Date.now();
  const wrap = document.createElement("div");
  wrap.className = "message bot thinking";
  wrap.id = id;
  wrap.innerHTML = `
    <div class="bubble thinking-bubble">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </div>`;
  chatBox.appendChild(wrap);
  scrollBottom();
  return id;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendBotMessage(data) {
  const wrap = document.createElement("div");
  wrap.className = "message bot";

  // Format the answer text: convert bullet dashes and numbered lists
  const formatted = formatAnswer(data.answer);

  // Sources accordion
  const sourcesHtml = buildSourcesHtml(data.source_chunks, data.clean_query);

  wrap.innerHTML = `
    <div class="bubble bot-bubble">
      <div class="answer-text">${formatted}</div>
      ${sourcesHtml}
    </div>`;

  chatBox.appendChild(wrap);
  scrollBottom();

  // Bind accordion toggles
  wrap.querySelectorAll(".source-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const panel = btn.nextElementSibling;
      const open  = panel.style.display !== "none";
      panel.style.display = open ? "none" : "block";
      btn.querySelector(".chevron").textContent = open ? "›" : "‹";
    });
  });
}

function buildSourcesHtml(chunks, cleanQuery) {
  // Sources hidden — clean chatbot output only
  return "";
}

function formatAnswer(text) {
  // Escape HTML first
  let html = escapeHtml(text);
  // Bold **text**
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  // Bullet lines starting with - or •
  html = html.replace(/^[-•]\s+(.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");
  // Numbered lists
  html = html.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");
  // Line breaks
  html = html.replace(/\n{2,}/g, "</p><p>");
  html = html.replace(/\n/g, "<br>");
  return `<p>${html}</p>`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setStatus(msg) {
  statusBar.textContent = msg;
}

function scrollBottom() {
  chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: "smooth" });
}
