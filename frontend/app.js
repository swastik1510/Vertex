document.addEventListener("DOMContentLoaded", () => {
  // Core controls
  const dateInput = document.getElementById("dateInput");
  const predictBtn = document.getElementById("predictBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const messageContainer = document.getElementById("messageContainer");
  const resultsSection = document.getElementById("resultsSection");
  const statusText = document.getElementById("statusText");
  const versionTag = document.getElementById("versionTag");
  const niftyBadge = document.getElementById("niftyBadge");
  const niftyReturnValue = document.getElementById("niftyReturnValue");
  const dateResolutionNote = document.getElementById("dateResolutionNote");

  // KPI elements
  const kpiTotalStocks = document.getElementById("kpiTotalStocks");
  const kpiPredictedBeatPct = document.getElementById("kpiPredictedBeatPct");
  const kpiBeatAccuracy = document.getElementById("kpiBeatAccuracy");
  const kpiBenchmarkAccuracy = document.getElementById("kpiBenchmarkAccuracy");

  // Tab controls
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const shortlistBadgeCount = document.getElementById("shortlistBadgeCount");

  // Table bodies & counters
  const signalsTableBody = document.getElementById("signalsTableBody");
  const outcomesTableBody = document.getElementById("outcomesTableBody");
  const shortlistTableBody = document.getElementById("shortlistTableBody");
  const signalsCount = document.getElementById("signalsCount");
  const outcomesCount = document.getElementById("outcomesCount");

  // Shortlist banner elements
  const shortlistBannerCount = document.getElementById("shortlistBannerCount");
  const shortlistBannerAlpha = document.getElementById("shortlistBannerAlpha");

  // Message alert helper
  function showMessage(type, text) {
    messageContainer.className = `alert alert-${type}`;
    messageContainer.innerHTML = `
      <span class="alert-icon">${type === "warning" ? "⚠️" : type === "danger" ? "❌" : "ℹ️"}</span>
      <span class="alert-text">${text}</span>
    `;
    messageContainer.classList.remove("hidden");
  }

  function hideMessage() {
    messageContainer.classList.add("hidden");
  }

  // 1. Tab switching logic
  function setupTabs() {
    tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-tab");

        tabButtons.forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        tabContents.forEach((c) => c.classList.remove("active"));

        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");

        const targetPanel = document.getElementById(targetId);
        if (targetPanel) {
          targetPanel.classList.add("active");
        }
      });
    });
  }

  // 2. Initialize metadata from backend
  async function loadMetadata() {
    try {
      const res = await fetch("/api/meta");
      if (!res.ok) throw new Error("Failed to load metadata");
      const data = await res.json();

      if (data.min_date && data.max_date) {
        dateInput.min = data.min_date;
        dateInput.max = data.max_date;
        dateInput.value = data.default_date || data.max_date;
      }

      if (data.active_model_mode && versionTag) {
        if (data.active_model_mode.startsWith("v2")) {
          versionTag.textContent = "Version 2 (Dual Model)";
          statusText.textContent = `Model + Data Loaded (${data.unique_stocks} Stocks, 2024 Test Set)`;
        } else {
          versionTag.textContent = "Version 1";
          statusText.textContent = `Model + Data Loaded (${data.unique_stocks} Stocks)`;
        }
      }
    } catch (err) {
      console.error("Metadata load error:", err);
      showMessage("danger", "Could not load dataset metadata. Please check the backend server.");
    }
  }

  // 3. Predict handler
  async function handlePredict() {
    const selectedDate = dateInput.value;
    if (!selectedDate) {
      showMessage("warning", "Please select a trading date first.");
      return;
    }

    hideMessage();
    dateResolutionNote.classList.add("hidden");
    resultsSection.classList.add("hidden");
    predictBtn.disabled = true;
    btnSpinner.classList.remove("hidden");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ date: selectedDate })
      });

      const data = await res.json();

      if (!res.ok) {
        if (res.status === 404) {
          showMessage("warning", data.detail || "No data for this date.");
        } else {
          showMessage("danger", `❌ Prediction failed: ${data.detail || "Unknown error"}`);
        }
        return;
      }

      // Check if trading date was adjusted
      if (!data.is_exact_date && data.resolved_date) {
        dateResolutionNote.textContent = `ℹ️ Non-trading day selected. Showing closest available trading day: ${data.resolved_date}`;
        dateResolutionNote.classList.remove("hidden");
      }

      // Populate Nifty Badge
      if (niftyBadge && niftyReturnValue) {
        niftyReturnValue.textContent = data.formatted_nifty_expected || "+0.00%";
        niftyBadge.classList.remove("hidden");
      }

      // Populate KPI cards
      renderKPIs(data.performance_summary);

      // Render Tables
      renderSignals(data.signals || []);
      renderOutcomes(data.outcomes || []);
      renderShortlist(data.shortlist || [], data.num_shortlisted || 0, data.formatted_avg_actual_alpha || "+0.00%");

      resultsSection.classList.remove("hidden");
    } catch (err) {
      console.error("Prediction error:", err);
      showMessage("danger", `❌ Prediction failed: ${err.message}`);
    } finally {
      predictBtn.disabled = false;
      btnSpinner.classList.add("hidden");
    }
  }

  // 4. Render KPIs
  function renderKPIs(summary) {
    if (!summary) return;
    kpiTotalStocks.textContent = summary.total_stocks || "0";
    kpiPredictedBeatPct.textContent = `${summary.predicted_beat_pct}%`;
    kpiBeatAccuracy.textContent = `${summary.beat_accuracy_pct}%`;
    kpiBenchmarkAccuracy.textContent = `${summary.benchmark_adjusted_accuracy_pct}%`;
  }

  // 5. Render Signals Table
  function renderSignals(signals) {
    signalsTableBody.innerHTML = "";
    signalsCount.textContent = `${signals.length} stocks`;

    if (signals.length === 0) {
      signalsTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--text-muted);">No signals available.</td></tr>`;
      return;
    }

    signals.forEach((item) => {
      const tr = document.createElement("tr");
      const rankBadgeClass = item.rank <= 3 ? "rank-badge top-3" : "rank-badge";
      const isYes = item.Beat_Nifty_Signal === "Yes";
      const signalPillClass = isYes ? "signal-pill signal-yes" : "signal-pill signal-no";

      const alphaVal = item.predicted_alpha_vs_nifty;
      const alphaClass = alphaVal > 0 ? "alpha-badge alpha-pos" : alphaVal < 0 ? "alpha-badge alpha-neg" : "alpha-badge";

      tr.innerHTML = `
        <td class="col-rank"><span class="${rankBadgeClass}">${item.rank}</span></td>
        <td class="col-stock">${item.Stock}</td>
        <td class="col-num">${item.formatted_close}</td>
        <td class="col-center"><span class="${signalPillClass}">${item.Beat_Nifty_Signal}</span></td>
        <td class="col-num" style="color:#34d399;font-weight:600;">${item.formatted_prob}</td>
        <td class="col-num">${item.formatted_pred_return}</td>
        <td class="col-num ${alphaClass}">${item.formatted_predicted_alpha}</td>
      `;
      signalsTableBody.appendChild(tr);
    });
  }

  // 6. Render Outcomes Table
  function renderOutcomes(outcomes) {
    outcomesTableBody.innerHTML = "";
    outcomesCount.textContent = `${outcomes.length} stocks`;

    if (outcomes.length === 0) {
      outcomesTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text-muted);">No outcomes available.</td></tr>`;
      return;
    }

    outcomes.forEach((item, idx) => {
      const tr = document.createElement("tr");
      const devVal = item.return_deviation;
      const devClass = devVal > 0 ? "alpha-pos" : devVal < 0 ? "alpha-neg" : "";

      tr.innerHTML = `
        <td class="col-rank">${idx + 1}</td>
        <td class="col-stock">${item.Stock}</td>
        <td class="col-num">${item.formatted_pred_return}</td>
        <td class="col-num">${item.formatted_actual_return}</td>
        <td class="col-num ${devClass}">${item.formatted_deviation}</td>
      `;
      outcomesTableBody.appendChild(tr);
    });
  }

  // 7. Render Shortlist Table
  function renderShortlist(shortlist, count, avgAlpha) {
    shortlistTableBody.innerHTML = "";
    shortlistBadgeCount.textContent = count;
    shortlistBannerCount.textContent = `${count} stocks`;
    shortlistBannerAlpha.textContent = `Average Actual Alpha: ${avgAlpha}`;

    if (shortlist.length === 0) {
      shortlistTableBody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:2rem;">No stocks met the consensus winning criteria on this date.</td></tr>`;
      return;
    }

    shortlist.forEach((item) => {
      const tr = document.createElement("tr");
      const rankBadgeClass = item.rank <= 3 ? "rank-badge top-3" : "rank-badge";

      tr.innerHTML = `
        <td class="col-rank"><span class="${rankBadgeClass}">${item.rank}</span></td>
        <td class="col-stock">${item.Stock}</td>
        <td class="col-num">${item.formatted_close}</td>
        <td class="col-num" style="color:#34d399;font-weight:600;">${item.formatted_prob}</td>
        <td class="col-num">${item.formatted_pred_return}</td>
        <td class="col-num">${item.formatted_actual_return}</td>
        <td class="col-num">${item.formatted_nifty_return}</td>
        <td class="col-num alpha-pos" style="font-weight:700;">${item.formatted_actual_alpha}</td>
      `;
      shortlistTableBody.appendChild(tr);
    });
  }

  // Initialize
  setupTabs();
  loadMetadata();
  predictBtn.addEventListener("click", handlePredict);
});
