class KapslApp {
  constructor() {
    this.apiBaseUrl = window.location.origin;
    this.updateInterval = 2000;
    this.accessRefreshIntervalMs = 15000;
    this.hardwareRefreshIntervalMs = 30000;
    this.extensionsRefreshIntervalMs = 20000;
    this.intervalId = null;
    this.currentModalId = null;
    this.currentScalingPolicy = null;

    this.authToken = localStorage.getItem("kapsl_auth_token") || "";
    this.selectedUserId = null;
    this.lastAccessRefreshAt = 0;
    this.lastHardwareRefreshAt = 0;
    this.lastExtensionsRefreshAt = 0;

    this.view = "dashboard";

    this.healthData = null;
    this.modelsData = [];
    this.hardwareData = null;
    this.systemStats = null;

    this.maxHistoryPoints = 120;
    this.history = {
      rssGiB: [],
      modelMemGiB: [],
      gpuUtilPct: [],
      throughput: [],
      active: [],
      queue: [],
    };

    this.modelHistory = {};

    this.extensionsDeveloperMode =
      localStorage.getItem("kapsl_extensions_dev") === "1";
    this.remotePlaceholderUrl =
      localStorage.getItem("kapsl_remote_placeholder_url") || "";
    this.extensions = [];
    this.marketplaceExtensions = [];
    this.remoteArtifacts = { remote_url: "", repo: "", available_repos: [], models: [] };
    this.remoteArtifactsLoading = false;
    this.currentRemoteArtifactName = null;

    this.coreFetchInFlight = false;
    this.isAuthenticated = false;
    this.loginInFlight = false;
    this.lastAuthError = "";
    this.sessionRole = "";
    this.sessionMode = "locked";
    this.sessionScopes = [];

    this.init();
  }

  init() {
    this.bindLoginControls();
    this.bindNavigation();
    this.bindDashboardControls();
    this.bindExtensionsControls();
    this.bindLegacyTokenControls();
    this.bindAccessControls();
    this.updateTokenStatus();
    this.updateSessionStatus();
    this.applyInitialView();
    this.setSessionLocked("Checking session access...");
    this.bootstrapSession();
  }

  startAutoRefresh() {
    this.stopAutoRefresh();
    this.intervalId = setInterval(async () => {
      if (!this.isAuthenticated) {
        return;
      }
      await this.fetchData();
      const now = Date.now();
      if (now - this.lastAccessRefreshAt >= this.accessRefreshIntervalMs) {
        await this.refreshAccessData({ silent: true });
        await this.refreshLegacyTokens({ silent: true });
      }
      if (now - this.lastHardwareRefreshAt >= this.hardwareRefreshIntervalMs) {
        await this.refreshHardwareData({ silent: true });
      }
      if (
        this.view === "extensions" &&
        now - this.lastExtensionsRefreshAt >= this.extensionsRefreshIntervalMs
      ) {
        await this.refreshExtensionsData({ silent: true });
      }
    }, this.updateInterval);
  }

  stopAutoRefresh() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  buildAuthHeader(token = this.authToken) {
    const rawToken = String(token || "").trim();
    if (!rawToken) {
      return null;
    }
    if (rawToken.toLowerCase().startsWith("bearer ")) {
      return rawToken;
    }
    return `Bearer ${rawToken}`;
  }

  async apiFetch(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    const authHeader = this.buildAuthHeader();
    if (authHeader) {
      headers.Authorization = authHeader;
    }

    return fetch(`${this.apiBaseUrl}${path}`, {
      ...options,
      headers,
    });
  }

  async requestJson(path, options = {}) {
    const response = await this.apiFetch(path, options);
    const raw = await response.text();
    let data = null;

    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch (_) {
        data = { error: raw };
      }
    }

    return {
      ok: response.ok,
      status: response.status,
      data,
    };
  }

  bindLoginControls() {
    document
      .getElementById("login-form")
      .addEventListener("submit", (event) => this.handleLoginSubmit(event));

    document.getElementById("login-clear-btn").addEventListener("click", () => {
      this.authToken = "";
      localStorage.removeItem("kapsl_auth_token");
      document.getElementById("login-token-input").value = "";
      document.getElementById("auth-token-input").value = "";
      this.updateTokenStatus();
      this.setLoginFeedback("", false);
    });

    document.getElementById("login-token-input").value = this.authToken;
  }

  async bootstrapSession() {
    const savedToken = this.authToken;
    if (savedToken) {
      const ok = await this.authenticate(savedToken, {
        persist: true,
        silent: true,
      });
      if (ok) {
        return;
      }
      this.authToken = "";
      localStorage.removeItem("kapsl_auth_token");
      document.getElementById("login-token-input").value = "";
      document.getElementById("auth-token-input").value = "";
      this.updateTokenStatus();
      const localFallbackOk = await this.authenticate("", {
        persist: false,
        silent: true,
      });
      if (localFallbackOk) {
        return;
      }
      this.setSessionLocked(
        "Saved token is no longer valid. Sign in again.",
        true,
      );
      return;
    }

    const localAccessOk = await this.authenticate("", {
      persist: false,
      silent: true,
    });
    if (localAccessOk) {
      return;
    }

    this.setSessionLocked(
      "Sign in with a reader, writer, or admin API key to view metrics.",
      false,
    );
  }

  setLoginFeedback(message, isError = false) {
    const element = document.getElementById("login-feedback");
    element.textContent = message || "";
    element.classList.toggle("error", Boolean(message && isError));
    element.classList.toggle("success", Boolean(message && !isError));
  }

  setSessionLocked(message = "", isError = false) {
    this.isAuthenticated = false;
    this.sessionRole = "";
    this.sessionMode = "locked";
    this.sessionScopes = [];
    this.stopAutoRefresh();
    document.body.classList.add("auth-locked");
    document.getElementById("auth-gate").classList.add("active");
    document.getElementById("app-shell").classList.add("locked");
    this.updateSessionStatus();
    this.updateDeveloperModeUI();
    this.setLoginFeedback(message, isError);
  }

  setSessionUnlocked() {
    this.isAuthenticated = true;
    if (!this.sessionMode || this.sessionMode === "locked") {
      this.sessionMode = this.authToken ? "api-key" : "local-loopback";
    }
    document.body.classList.remove("auth-locked");
    document.getElementById("auth-gate").classList.remove("active");
    document.getElementById("app-shell").classList.remove("locked");
    this.updateSessionStatus();
    this.updateDeveloperModeUI();
  }

  updateSessionStatus() {
    const status = document.getElementById("session-status");
    const detail = document.getElementById("session-detail");
    const setDetail = (value) => {
      if (detail) {
        detail.textContent = value;
      }
    };
    if (!status) {
      return;
    }
    if (!this.isAuthenticated) {
      status.textContent = "Locked";
      setDetail("mode: locked");
      return;
    }

    const normalizedRole = String(this.sessionRole || "")
      .trim()
      .toLowerCase();
    const mode = String(
      this.sessionMode || (this.authToken ? "api-key" : "local-loopback"),
    )
      .trim()
      .toLowerCase();

    if (normalizedRole) {
      status.textContent =
        normalizedRole.charAt(0).toUpperCase() + normalizedRole.slice(1);
    } else if (mode === "local-loopback") {
      status.textContent = "Local";
    } else {
      status.textContent = this.authToken ? "Authenticated" : "Local";
    }

    const scopes = Array.isArray(this.sessionScopes)
      ? this.sessionScopes
          .map((scope) => String(scope || "").trim())
          .filter((scope) => scope.length > 0)
      : [];
    const scopePreview = scopes.length
      ? scopes.slice(0, 2).join(",") + (scopes.length > 2 ? ",..." : "")
      : "all";
    setDetail(`mode: ${mode || "unknown"} | scopes: ${scopePreview}`);
  }

  handleSessionUnauthorized(
    message = "Session expired. Sign in again to view runtime metrics.",
  ) {
    if (this.currentModalId !== null) {
      this.closeModal();
    }
    this.authToken = "";
    localStorage.removeItem("kapsl_auth_token");
    document.getElementById("auth-token-input").value = "";
    document.getElementById("login-token-input").value = "";
    this.updateTokenStatus();
    this.setSessionLocked(message, true);
  }

  async handleLoginSubmit(event) {
    event.preventDefault();
    const token = document.getElementById("login-token-input").value;
    await this.authenticate(token, { persist: true });
  }

  async authenticate(rawToken, { persist = true, silent = false } = {}) {
    if (this.loginInFlight) {
      return false;
    }

    this.loginInFlight = true;
    const token = String(rawToken || "").trim();
    try {
      if (!silent) {
        this.setLoginFeedback("Signing in...", false);
      }

      const verification = await this.verifyReaderAccess(token);
      if (!verification.ok) {
        this.lastAuthError = verification.message || "Authentication failed.";
        if (!silent) {
          this.setLoginFeedback(verification.message, true);
        }
        return false;
      }

      this.lastAuthError = "";
      this.sessionRole = String(verification.session?.role || "")
        .trim()
        .toLowerCase();
      this.sessionMode = String(
        verification.session?.mode || (token ? "api-key" : "local-loopback"),
      )
        .trim()
        .toLowerCase();
      this.sessionScopes = Array.isArray(verification.session?.scopes)
        ? verification.session.scopes
            .map((scope) => String(scope || "").trim())
            .filter((scope) => scope.length > 0)
        : [];
      this.authToken = token;
      if (persist) {
        if (token) {
          localStorage.setItem("kapsl_auth_token", token);
        } else {
          localStorage.removeItem("kapsl_auth_token");
        }
      }

      document.getElementById("login-token-input").value = token;
      document.getElementById("auth-token-input").value = token;
      this.updateTokenStatus();
      this.setSessionUnlocked();
      if (!silent) {
        const roleLabel = this.sessionRole || "reader";
        this.setLoginFeedback(
          `Signed in as ${roleLabel} (${this.sessionMode || "api-key"}).`,
          false,
        );
      }

      await this.fetchData();
      await Promise.all([
        this.refreshHardwareData({ silent: true }),
        this.refreshLegacyTokens({ silent: true }),
        this.refreshAccessData({ silent: true }),
        this.refreshExtensionsData({ silent: true }),
        this.refreshRemoteArtifacts({ silent: true }),
      ]);
      this.startAutoRefresh();
      return true;
    } finally {
      this.loginInFlight = false;
    }
  }

  async verifyReaderAccess(rawToken) {
    const authHeader = this.buildAuthHeader(rawToken);
    const loginHeaders = { "Content-Type": "application/json" };
    if (authHeader) {
      loginHeaders.Authorization = authHeader;
    }

    try {
      const response = await fetch(`${this.apiBaseUrl}/api/auth/login`, {
        method: "POST",
        headers: loginHeaders,
        body: JSON.stringify({}),
      });

      const body = await this.parseResponseBody(response);
      const bodyError = body?.error || body?.detail || "";

      if (response.ok) {
        return { ok: true, session: body || null };
      }

      // Backward compatibility for runtimes that do not implement /api/auth/login.
      if (response.status === 404) {
        return this.verifyReaderAccessLegacyHealth(authHeader);
      }

      if (response.status === 401 || response.status === 403) {
        return {
          ok: false,
          message:
            bodyError ||
            "Access denied. Provide a valid reader/writer/admin API key.",
        };
      }

      return {
        ok: false,
        message:
          bodyError ||
          `Sign in failed while validating access (HTTP ${response.status}).`,
      };
    } catch (error) {
      return {
        ok: false,
        message: `Unable to reach runtime API: ${error.message}`,
      };
    }
  }

  async verifyReaderAccessLegacyHealth(authHeader) {
    const headers = {};
    if (authHeader) {
      headers.Authorization = authHeader;
    }

    try {
      const response = await fetch(`${this.apiBaseUrl}/api/health`, {
        headers,
      });
      const body = await this.parseResponseBody(response);
      const bodyError = body?.error || body?.detail || "";

      if (response.ok) {
        return {
          ok: true,
          session: {
            role: authHeader ? "reader" : "admin",
            mode: authHeader ? "legacy-token" : "local-loopback",
            scopes: [],
          },
        };
      }

      if (response.status === 401 || response.status === 403) {
        return {
          ok: false,
          message:
            bodyError ||
            "Access denied. Provide a valid reader/writer/admin API key.",
        };
      }

      if (response.status === 404) {
        return {
          ok: false,
          message:
            bodyError ||
            "Runtime API endpoint not found (HTTP 404). Use the runtime base URL (for example, http://localhost:8080), then restart with the latest binary.",
        };
      }

      return {
        ok: false,
        message:
          bodyError ||
          `Sign in failed while validating access (HTTP ${response.status}).`,
      };
    } catch (error) {
      return {
        ok: false,
        message: `Unable to reach runtime API: ${error.message}`,
      };
    }
  }

  async parseResponseBody(response) {
    const bodyText = await response.text();
    if (!bodyText) {
      return null;
    }
    try {
      return JSON.parse(bodyText);
    } catch (_) {
      return { error: bodyText };
    }
  }

  bindNavigation() {
    const navItems = document.querySelectorAll(".nav-item[data-view]");
    for (const item of navItems) {
      item.addEventListener("click", () => {
        const view = item.dataset.view;
        if (view) {
          this.setView(view);
        }
      });
    }

    window.addEventListener("hashchange", () => this.applyInitialView());
  }

  applyInitialView() {
    const raw = (window.location.hash || "")
      .replace("#", "")
      .trim()
      .toLowerCase();
    const allowed = new Set(["dashboard", "system", "extensions", "access"]);
    const view = allowed.has(raw) ? raw : "dashboard";
    this.setView(view, { pushHash: false });
  }

  setView(view, { pushHash = true } = {}) {
    const normalized = String(view || "")
      .trim()
      .toLowerCase();
    if (!normalized) {
      return;
    }
    if (this.view === normalized) {
      // Still ensure the UI is consistent on first load.
      this.syncViewUI(normalized);
      return;
    }

    this.view = normalized;
    this.syncViewUI(normalized);

    if (pushHash) {
      window.location.hash = normalized;
    }

    if (normalized === "extensions") {
      this.refreshExtensionsData({ silent: true });
    }
  }

  syncViewUI(view) {
    const sections = document.querySelectorAll(".view[data-view]");
    for (const section of sections) {
      section.classList.toggle("active", section.dataset.view === view);
    }

    const navItems = document.querySelectorAll(".nav-item[data-view]");
    for (const item of navItems) {
      item.classList.toggle("active", item.dataset.view === view);
    }
  }

  bindDashboardControls() {
    const startForm = document.getElementById("start-model-form");
    startForm.addEventListener("submit", (event) =>
      this.handleStartModel(event),
    );

    document
      .getElementById("start-model-clear")
      .addEventListener("click", () => this.clearStartModelForm());

    const remoteUrlInput = document.getElementById("engine-remote-url");
    remoteUrlInput.value = this.remotePlaceholderUrl;
    remoteUrlInput.addEventListener("change", () => {
      this.persistRemotePlaceholderUrl(remoteUrlInput.value);
      this.refreshRemoteArtifacts();
    });

    document
      .getElementById("engine-remote-refresh")
      .addEventListener("click", () => this.refreshRemoteArtifacts());

    document
      .getElementById("engine-remote-grid")
      .addEventListener("click", (event) => this.handleRemoteArtifactsClick(event));

    document
      .getElementById("modal-body")
      .addEventListener("click", (event) => this.handleModalActionClick(event));
  }

  bindExtensionsControls() {
    document
      .getElementById("extensions-refresh-btn")
      .addEventListener("click", () => this.refreshExtensionsData());

    document
      .getElementById("marketplace-search-form")
      .addEventListener("submit", (event) =>
        this.handleMarketplaceSearch(event),
      );

    const devToggle = document.getElementById("extensions-dev-toggle");
    devToggle.checked = this.extensionsDeveloperMode;
    devToggle.addEventListener("change", () => {
      this.extensionsDeveloperMode = Boolean(devToggle.checked);
      localStorage.setItem(
        "kapsl_extensions_dev",
        this.extensionsDeveloperMode ? "1" : "0",
      );
      this.updateDeveloperModeUI();
    });
    this.updateDeveloperModeUI();

    document
      .getElementById("local-install-form")
      .addEventListener("submit", (event) =>
        this.handleLocalExtensionInstall(event),
      );

    document
      .getElementById("marketplace-results")
      .addEventListener("click", (event) => this.handleMarketplaceClick(event));

    document
      .getElementById("extensions-list")
      .addEventListener("click", (event) =>
        this.handleExtensionsListClick(event),
      );
  }

  bindLegacyTokenControls() {
    document
      .getElementById("legacy-tokens-form")
      .addEventListener("submit", (event) =>
        this.handleLegacyTokensSave(event),
      );
    document
      .getElementById("legacy-refresh-btn")
      .addEventListener("click", () => this.refreshLegacyTokens());
  }

  bindAccessControls() {
    document
      .getElementById("save-auth-token-btn")
      .addEventListener("click", () => this.saveAuthToken());
    document
      .getElementById("clear-auth-token-btn")
      .addEventListener("click", () => this.clearAuthToken());
    document
      .getElementById("refresh-access-btn")
      .addEventListener("click", () => this.refreshAccessData());

    document
      .getElementById("create-user-form")
      .addEventListener("submit", (event) => this.handleCreateUser(event));
    document
      .getElementById("create-key-form")
      .addEventListener("submit", (event) => this.handleCreateKey(event));

    document
      .getElementById("access-users-body")
      .addEventListener("click", (event) => this.handleUsersTableClick(event));
    document
      .getElementById("access-keys-body")
      .addEventListener("click", (event) => this.handleKeysTableClick(event));
    document
      .getElementById("new-key-once")
      .addEventListener("click", (event) => this.handleNewKeyActions(event));

    document.getElementById("auth-token-input").value = this.authToken;
  }

  async saveAuthToken() {
    const input = document.getElementById("auth-token-input");
    const ok = await this.authenticate(input.value, {
      persist: true,
      silent: true,
    });
    if (ok) {
      this.updateTokenStatus();
      return;
    }
    const status = document.getElementById("auth-token-status");
    status.textContent =
      this.lastAuthError ||
      "Access denied. Provide a valid reader/writer/admin API key.";
    status.classList.add("error");
    status.classList.remove("success");
  }

  clearAuthToken() {
    this.authToken = "";
    localStorage.removeItem("kapsl_auth_token");
    document.getElementById("auth-token-input").value = "";
    document.getElementById("login-token-input").value = "";
    this.updateTokenStatus();
    if (this.currentModalId !== null) {
      this.closeModal();
    }
    this.setSessionLocked("Signed out. Sign in to continue.", false);
  }

  updateTokenStatus() {
    const status = document.getElementById("auth-token-status");
    if (this.authToken) {
      status.textContent = "Token saved in browser storage.";
      status.classList.remove("error");
      status.classList.add("success");
      return;
    }

    if (this.isAuthenticated) {
      status.textContent =
        "Using local loopback session (auth token not required).";
      status.classList.remove("error");
      status.classList.add("success");
      return;
    }

    status.textContent =
      "No token configured. Admin APIs require a valid admin API key when auth is enabled.";
    status.classList.remove("success");
    status.classList.remove("error");
  }

  setAccessFeedback(targetId, message, isError = false) {
    const element = document.getElementById(targetId);
    element.textContent = message || "";
    element.classList.toggle("error", Boolean(message && isError));
    element.classList.toggle("success", Boolean(message && !isError));
  }

  async fetchData() {
    if (!this.isAuthenticated) {
      return;
    }
    if (this.coreFetchInFlight) {
      return;
    }
    this.coreFetchInFlight = true;
    try {
      const [healthResponse, modelsResponse] = await Promise.all([
        this.apiFetch("/api/health"),
        this.apiFetch("/api/models"),
      ]);

      // System stats are optional; keep the UI usable if the endpoint is missing
      // or requires different auth.
      let systemStatsResponse = null;
      try {
        systemStatsResponse = await this.apiFetch("/api/system/stats");
      } catch (_) {
        systemStatsResponse = null;
      }

      if (!healthResponse.ok || !modelsResponse.ok) {
        const status = !healthResponse.ok
          ? healthResponse.status
          : modelsResponse.status;
        if (status === 401 || status === 403) {
          this.handleSessionUnauthorized(
            "Access denied. Sign in with a valid API key.",
          );
          return;
        }
        throw new Error(`Failed to fetch API data (HTTP ${status})`);
      }

      const healthData = await healthResponse.json();
      const modelsData = await modelsResponse.json();
      let systemStats = null;
      if (systemStatsResponse && systemStatsResponse.ok) {
        try {
          systemStats = await systemStatsResponse.json();
        } catch (_) {
          systemStats = null;
        }
      }

      this.healthData = healthData;
      this.modelsData = Array.isArray(modelsData) ? modelsData : [];
      this.systemStats =
        systemStats && typeof systemStats === "object" ? systemStats : null;

      this.updateUI();
      this.hideError();
    } catch (error) {
      console.error("Error fetching data:", error);
      this.showError(error.message);
    } finally {
      this.coreFetchInFlight = false;
    }
  }

  async refreshHardwareData({ silent = false } = {}) {
    if (!this.isAuthenticated) {
      return;
    }
    try {
      const result = await this.requestJson("/api/hardware");
      if (!result.ok) {
        if (result.status === 401 || result.status === 403) {
          this.handleSessionUnauthorized(
            "Session no longer has reader access. Sign in again.",
          );
          return;
        }
        throw new Error(
          result.data?.error || `Hardware API error (HTTP ${result.status})`,
        );
      }

      this.hardwareData = result.data;
      this.lastHardwareRefreshAt = Date.now();
      this.updateHardwareUI();
      if (!silent) {
        this.setAccessFeedback("start-model-feedback", "", false);
      }
    } catch (error) {
      console.error("Error refreshing hardware:", error);
      if (!silent) {
        this.setAccessFeedback("start-model-feedback", error.message, true);
      }
    }
  }

  updateHardwareUI() {
    const hw = this.hardwareData;
    const cpuEl = document.getElementById("hw-cpu");
    const memEl = document.getElementById("hw-mem");
    const gpusEl = document.getElementById("hw-gpus");
    const providersEl = document.getElementById("hw-providers");
    const detailEl = document.getElementById("hardware-detail");

    if (!hw || typeof hw !== "object") {
      cpuEl.textContent = "-";
      memEl.textContent = "-";
      gpusEl.textContent = "-";
      providersEl.textContent = "-";
      detailEl.textContent = "";
      return;
    }

    const cpuCores = hw.cpu_cores ?? "-";
    cpuEl.textContent =
      typeof cpuCores === "number" ? `${cpuCores} cores` : String(cpuCores);

    const totalMemKiB = hw.total_memory ?? 0;
    memEl.textContent = this.formatBytes(Number(totalMemKiB) * 1024);

    const devices = Array.isArray(hw.devices) ? hw.devices : [];
    const gpus = devices.filter(
      (d) => String(d.backend || "").toLowerCase() !== "cpu",
    );
    if (!gpus.length) {
      gpusEl.textContent = "none";
    } else {
      gpusEl.textContent = `${gpus.length} detected`;
    }

    const providers = [];
    if (hw.has_cuda) providers.push("cuda");
    if (hw.has_metal) providers.push("metal");
    if (hw.has_rocm) providers.push("rocm");
    if (hw.has_directml) providers.push("directml");
    const seen = new Set(providers);
    for (const dev of devices) {
      const key = String(dev.backend || "").toLowerCase();
      if (key && !seen.has(key) && key !== "cpu") {
        providers.push(key);
        seen.add(key);
      }
    }
    providersEl.textContent = providers.length ? providers.join(", ") : "cpu";

    const lines = [];
    for (const dev of devices) {
      const backend = String(dev.backend || "").toLowerCase() || "unknown";
      const name = String(dev.name || "device");
      const mem = dev.memory_mb ? `${dev.memory_mb} MB` : "unknown mem";
      const util =
        dev.utilization_gpu_pct !== undefined &&
        dev.utilization_gpu_pct !== null
          ? ` util=${dev.utilization_gpu_pct}%`
          : "";
      const temp =
        dev.temperature_c !== undefined && dev.temperature_c !== null
          ? ` temp=${dev.temperature_c}C`
          : "";
      lines.push(`${backend}: ${name} (${mem})${util}${temp}`);
    }
    detailEl.textContent = lines.join("\n");
  }

  updateSystemView(modelsData) {
    const stats = this.systemStats;
    const feedbackId = "system-feedback";

    const primaryModels = modelsData.filter((m) => (m.replica_id || 0) === 0);

    const modelMemBytes = primaryModels.reduce(
      (sum, m) => sum + Number(m.memory_usage || 0),
      0,
    );
    const totalThroughput = primaryModels.reduce(
      (sum, m) => sum + Number(m.throughput || 0),
      0,
    );
    const totalActive = primaryModels.reduce(
      (sum, m) => sum + Number(m.active_inferences || 0),
      0,
    );
    const totalQueue = primaryModels.reduce((sum, m) => {
      const q = Array.isArray(m.queue_depth) ? m.queue_depth : [0, 0];
      return sum + Number(q[0] || 0) + Number(q[1] || 0);
    }, 0);

    const rssBytes =
      stats?.process_memory_bytes ??
      primaryModels.reduce((max, m) => {
        const mem = Number(m.memory_usage || 0);
        return mem > max ? mem : max;
      }, 0);

    let gpuUtilFrac = Number(stats?.gpu_utilization || 0);
    if (!(gpuUtilFrac > 0)) {
      gpuUtilFrac = primaryModels.length
        ? primaryModels.reduce(
            (sum, m) => sum + Number(m.gpu_utilization || 0),
            0,
          ) / primaryModels.length
        : 0;
    }
    const gpuUtilPct = gpuUtilFrac * 100;

    document.getElementById("sys-rss").textContent = this.formatBytes(rssBytes);
    document.getElementById("sys-model-mem").textContent =
      this.formatBytes(modelMemBytes);
    document.getElementById("sys-gpu-util").textContent =
      this.formatPercent(gpuUtilPct);
    document.getElementById("sys-throughput").textContent =
      `${totalThroughput.toFixed(2)} req/s`;
    document.getElementById("sys-active").textContent = String(totalActive);
    document.getElementById("sys-queue").textContent = String(totalQueue);

    document.getElementById("sys-pid").textContent =
      stats?.pid !== undefined && stats?.pid !== null ? String(stats.pid) : "-";
    const gpuMemBytes = stats?.gpu_memory_bytes ?? null;
    document.getElementById("sys-gpu-mem").textContent = gpuMemBytes
      ? this.formatBytes(Number(gpuMemBytes))
      : "n/a";
    document.getElementById("sys-collected").textContent =
      stats?.collected_at_ms
        ? new Date(Number(stats.collected_at_ms)).toLocaleString()
        : "-";

    const rssGiB = rssBytes / 1024 ** 3;
    const modelMemGiB = modelMemBytes / 1024 ** 3;

    this.pushHistory(this.history.rssGiB, rssGiB);
    this.pushHistory(this.history.modelMemGiB, modelMemGiB);
    this.pushHistory(this.history.gpuUtilPct, gpuUtilPct);
    this.pushHistory(this.history.throughput, totalThroughput);
    this.pushHistory(this.history.active, totalActive);
    this.pushHistory(this.history.queue, totalQueue);

    this.drawSparkline("chart-rss", this.history.rssGiB, "#0b6bcb");
    this.drawSparkline("chart-model-mem", this.history.modelMemGiB, "#ef5b25");
    this.drawSparkline(
      "chart-gpu-util",
      this.history.gpuUtilPct,
      "#0a9b7c",
      100,
    );
    this.drawSparkline("chart-throughput", this.history.throughput, "#ef5b25");
    this.drawSparkline("chart-active", this.history.active, "#0b6bcb");
    this.drawSparkline("chart-queue", this.history.queue, "#b26a00");

    if (!stats) {
      this.setAccessFeedback(
        feedbackId,
        "System stats unavailable. Ensure /api/system/stats is reachable.",
        true,
      );
    } else {
      this.setAccessFeedback(feedbackId, "", false);
    }
  }

  updateModelHistory(modelsData) {
    const primaryModels = (Array.isArray(modelsData) ? modelsData : []).filter(
      (m) => (m.replica_id || 0) === 0,
    );

    const seen = new Set();
    for (const model of primaryModels) {
      const id = model.id;
      if (id === undefined || id === null) {
        continue;
      }
      const key = String(id);
      seen.add(key);

      if (!this.modelHistory[key]) {
        this.modelHistory[key] = {
          totalInferences: [],
          queuePending: [],
          queueProcessing: [],
          throughput: [],
          gpuUtilPct: [],
        };
      }

      const entry = this.modelHistory[key];
      this.pushHistory(
        entry.totalInferences,
        Number(model.total_inferences || 0),
      );

      const q = Array.isArray(model.queue_depth) ? model.queue_depth : [0, 0];
      this.pushHistory(entry.queuePending, Number(q[0] || 0));
      this.pushHistory(entry.queueProcessing, Number(q[1] || 0));

      this.pushHistory(entry.throughput, Number(model.throughput || 0));
      this.pushHistory(
        entry.gpuUtilPct,
        Number(model.gpu_utilization || 0) * 100,
      );
    }

    for (const key of Object.keys(this.modelHistory)) {
      if (!seen.has(key)) {
        delete this.modelHistory[key];
      }
    }
  }

  pushHistory(series, value) {
    if (!Array.isArray(series)) {
      return;
    }
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return;
    }
    series.push(num);
    const extra = series.length - this.maxHistoryPoints;
    if (extra > 0) {
      series.splice(0, extra);
    }
  }

  drawSparkline(canvasId, series, color, fixedMax = null) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !canvas.getContext) {
      return;
    }
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Background grid
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(18,18,18,0.10)";
    for (let i = 1; i <= 3; i += 1) {
      const y = Math.round((height * i) / 4);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (!Array.isArray(series) || series.length < 2) {
      return;
    }

    let min = Infinity;
    let max = -Infinity;
    for (const v of series) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (fixedMax !== null && Number.isFinite(fixedMax)) {
      max = Math.max(max, fixedMax);
      min = Math.min(min, 0);
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return;
    }
    if (max === min) {
      max += 1;
      min -= 1;
    }

    const pad = 8;
    const usableH = height - pad * 2;
    const stepX = (width - pad * 2) / (series.length - 1);

    const points = series.map((v, idx) => {
      const x = pad + stepX * idx;
      const t = (v - min) / (max - min);
      const y = pad + usableH * (1 - t);
      return [x, y];
    });

    // Area fill
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(points[0][0], height - pad);
    for (const [x, y] of points) {
      ctx.lineTo(x, y);
    }
    ctx.lineTo(points[points.length - 1][0], height - pad);
    ctx.closePath();
    ctx.fill();

    // Line
    ctx.globalAlpha = 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.stroke();

    // Latest dot
    const [lx, ly] = points[points.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lx, ly, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  drawMultiSparkline(canvasId, seriesList, colors, fixedMax = null) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !canvas.getContext) {
      return;
    }
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Background grid
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(18,18,18,0.10)";
    for (let i = 1; i <= 3; i += 1) {
      const y = Math.round((height * i) / 4);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const normalized = (Array.isArray(seriesList) ? seriesList : []).filter(
      (series) => Array.isArray(series) && series.length > 0,
    );
    if (!normalized.length) {
      return;
    }

    let min = Infinity;
    let max = -Infinity;
    let maxLen = 0;
    for (const series of normalized) {
      maxLen = Math.max(maxLen, series.length);
      for (const v of series) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }

    if (fixedMax !== null && Number.isFinite(fixedMax)) {
      max = Math.max(max, fixedMax);
      min = Math.min(min, 0);
    }
    if (!Number.isFinite(min) || !Number.isFinite(max) || maxLen < 2) {
      return;
    }
    if (max === min) {
      max += 1;
      min -= 1;
    }

    const pad = 8;
    const usableH = height - pad * 2;
    const stepX = (width - pad * 2) / (maxLen - 1);

    normalized.forEach((series, idx) => {
      if (series.length < 2) {
        return;
      }

      const offset = maxLen - series.length;
      const points = series.map((v, j) => {
        const x = pad + stepX * (offset + j);
        const t = (v - min) / (max - min);
        const y = pad + usableH * (1 - t);
        return [x, y];
      });

      const color = Array.isArray(colors) ? colors[idx] : null;
      ctx.globalAlpha = 0.95;
      ctx.strokeStyle = color || "rgba(18,18,18,0.6)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i += 1) {
        ctx.lineTo(points[i][0], points[i][1]);
      }
      ctx.stroke();

      const [lx, ly] = points[points.length - 1];
      ctx.fillStyle = color || "rgba(18,18,18,0.6)";
      ctx.beginPath();
      ctx.arc(lx, ly, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  updateModalCharts(modelId) {
    const key = String(modelId);
    const entry = this.modelHistory[key] || null;
    if (!entry) {
      this.drawSparkline("modal-chart-inferences", [], "#0b6bcb");
      this.drawSparkline("modal-chart-throughput", [], "#ef5b25");
      this.drawMultiSparkline("modal-chart-queue", [], []);
      return;
    }

    this.drawSparkline(
      "modal-chart-inferences",
      entry.totalInferences,
      "#0b6bcb",
    );
    this.drawSparkline("modal-chart-throughput", entry.throughput, "#ef5b25");
    this.drawMultiSparkline(
      "modal-chart-queue",
      [entry.queuePending, entry.queueProcessing],
      ["#0b6bcb", "#b26a00"],
    );
  }

  updateDeveloperModeUI() {
    const localPanel = document.getElementById("local-install-panel");
    const disabledNote = document.getElementById("local-install-disabled-note");
    const notice = document.getElementById("local-install-notice");
    const input = document.getElementById("local-extension-path");
    const submit = document.querySelector(
      "#local-install-form button[type='submit']",
    );
    const devToggleLabel = document.getElementById(
      "extensions-dev-toggle",
    ).parentElement;

    const isAdmin = this.sessionRole === "admin";
    const enabled = isAdmin && Boolean(this.extensionsDeveloperMode);

    devToggleLabel.hidden = !isAdmin;
    localPanel.hidden = !enabled;
    input.disabled = !enabled;
    submit.disabled = !enabled;

    notice.textContent = enabled
      ? ""
      : "Local extension install is available only in developer mode.";
    disabledNote.hidden = !isAdmin;
    disabledNote.textContent = isAdmin && !enabled
      ? "Enable Developer Features to reveal local extension install."
      : "";

    if (!enabled) {
      this.setAccessFeedback("local-install-feedback", "", false);
    }
  }

  async refreshExtensionsData({ silent = false } = {}) {
    if (!this.isAuthenticated) {
      return;
    }
    try {
      const result = await this.requestJson("/api/extensions");
      if (!result.ok) {
        if (result.status === 401 || result.status === 403) {
          this.extensions = [];
          this.renderExtensionsUnauthorized();
          if (!silent) {
            this.setAccessFeedback(
              "extensions-feedback",
              "Writer access required. Save a writer/admin API key.",
              true,
            );
          }
          return;
        }
        throw new Error(
          result.data?.error || `Extensions API error (HTTP ${result.status})`,
        );
      }

      this.extensions = Array.isArray(result.data) ? result.data : [];
      this.lastExtensionsRefreshAt = Date.now();
      this.renderExtensionsList();
      if (!silent) {
        this.setAccessFeedback("extensions-feedback", "", false);
      }
    } catch (error) {
      console.error("Error refreshing extensions:", error);
      if (!silent) {
        this.setAccessFeedback("extensions-feedback", error.message, true);
      }
    }
  }

  renderExtensionsUnauthorized() {
    const container = document.getElementById("extensions-list");
    container.innerHTML =
      '<div class="empty-state">Writer access is required to view installed extensions.</div>';
  }

  renderExtensionsList() {
    const container = document.getElementById("extensions-list");
    if (!this.extensions.length) {
      container.innerHTML =
        '<div class="empty-state">No extensions installed.</div>';
      return;
    }

    container.innerHTML = this.extensions
      .map((ext) => {
        const manifest = ext.manifest || {};
        const name = manifest.name || manifest.id || "Extension";
        const id = manifest.id || "";
        const version = manifest.version || "";
        const runtime = manifest.runtime || "";
        const caps = Array.isArray(manifest.capabilities)
          ? manifest.capabilities
          : [];
        const auth = Array.isArray(manifest.auth) ? manifest.auth : [];
        const desc = manifest.description || "";
        const configSchema = manifest.config_schema
          ? JSON.stringify(manifest.config_schema, null, 2)
          : "";

        return `
          <div class="ext-card" data-extension-id="${this.escapeHtml(id)}">
            <div class="ext-head">
              <div>
                <div class="ext-name">${this.escapeHtml(name)}</div>
                <div class="ext-id">${this.escapeHtml(id)}${version ? ` @ ${this.escapeHtml(version)}` : ""}</div>
              </div>
              <div class="chips">
                ${runtime ? `<span class="chip">${this.escapeHtml(runtime)}</span>` : ""}
                ${caps.map((c) => `<span class="chip">${this.escapeHtml(c)}</span>`).join("")}
                ${auth.map((a) => `<span class="chip">${this.escapeHtml(a)}</span>`).join("")}
              </div>
            </div>
            ${desc ? `<div class="ext-desc">${this.escapeHtml(desc)}</div>` : ""}
            <div class="ext-actions">
              <button class="btn btn-secondary" type="button" data-action="ext-launch">Launch</button>
              <button class="btn btn-secondary" type="button" data-action="ext-sync">Sync</button>
              <button class="btn btn-secondary" type="button" data-action="ext-load-config">Load Config</button>
              <button class="btn btn-secondary" type="button" data-action="ext-save-config">Save Config</button>
              <button class="btn btn-danger" type="button" data-action="ext-uninstall">Uninstall</button>
            </div>
            <textarea class="mono-textarea" data-field="ext-config" placeholder="Workspace config JSON (Load Config to populate)"></textarea>
            ${
              configSchema
                ? `<div class="mono-block">${this.escapeHtml(configSchema)}</div>`
                : ""
            }
          </div>
        `;
      })
      .join("");
  }

  async handleMarketplaceSearch(event) {
    event.preventDefault();
    const query = document.getElementById("marketplace-query").value.trim();
    const urlOverride = document.getElementById("marketplace-url").value.trim();

    const params = new URLSearchParams();
    if (query) params.set("q", query);
    if (urlOverride) params.set("marketplace_url", urlOverride);

    const path = `/api/extensions/marketplace?${params.toString()}`;
    this.setAccessFeedback("marketplace-feedback", "Searching...", false);

    try {
      const result = await this.requestJson(path);
      if (!result.ok) {
        throw new Error(
          result.data?.error ||
            `Marketplace request failed (HTTP ${result.status})`,
        );
      }

      this.marketplaceExtensions = Array.isArray(result.data)
        ? result.data
        : [];
      this.renderMarketplaceResults();
      this.setAccessFeedback("marketplace-feedback", "", false);
    } catch (error) {
      console.error("Marketplace search error:", error);
      this.setAccessFeedback("marketplace-feedback", error.message, true);
    }
  }

  renderMarketplaceResults() {
    const container = document.getElementById("marketplace-results");
    const list = Array.isArray(this.marketplaceExtensions)
      ? this.marketplaceExtensions
      : [];

    if (!list.length) {
      container.innerHTML =
        '<div class="empty-state">No marketplace extensions found.</div>';
      return;
    }

    container.innerHTML = list
      .map((ext) => {
        const manifest = ext.manifest || {};
        const name = manifest.name || manifest.id || "Extension";
        const id = manifest.id || "";
        const version = manifest.version || "";
        const desc = manifest.description || "";
        const runtime = manifest.runtime || "";
        const caps = Array.isArray(manifest.capabilities)
          ? manifest.capabilities
          : [];

        return `
          <div class="ext-card" data-extension-id="${this.escapeHtml(id)}">
            <div class="ext-head">
              <div>
                <div class="ext-name">${this.escapeHtml(name)}</div>
                <div class="ext-id">${this.escapeHtml(id)}${version ? ` @ ${this.escapeHtml(version)}` : ""}</div>
              </div>
              <div class="chips">
                ${runtime ? `<span class="chip">${this.escapeHtml(runtime)}</span>` : ""}
                ${caps.map((c) => `<span class="chip">${this.escapeHtml(c)}</span>`).join("")}
              </div>
            </div>
            ${desc ? `<div class="ext-desc">${this.escapeHtml(desc)}</div>` : ""}
            <div class="ext-actions">
              <button class="btn btn-primary" type="button" data-action="market-install">Install</button>
            </div>
          </div>
        `;
      })
      .join("");
  }

  handleMarketplaceClick(event) {
    const button = event.target.closest("button[data-action]");
    if (!button) {
      return;
    }

    const card = button.closest(".ext-card");
    const extensionId = card?.dataset.extensionId;
    if (!extensionId) {
      return;
    }

    const action = button.dataset.action;
    if (action === "market-install") {
      this.installMarketplaceExtension(extensionId);
    }
  }

  async installMarketplaceExtension(extensionId) {
    const urlOverride = document.getElementById("marketplace-url").value.trim();
    const payload = {
      extension_id: extensionId,
      marketplace_url: urlOverride || null,
    };

    try {
      const result = await this.requestJson("/api/extensions/install", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!result.ok) {
        throw new Error(
          result.data?.error || `Install failed (HTTP ${result.status})`,
        );
      }

      this.setAccessFeedback(
        "marketplace-feedback",
        `Installed ${extensionId}.`,
        false,
      );
      await this.refreshExtensionsData({ silent: true });
    } catch (error) {
      this.setAccessFeedback("marketplace-feedback", error.message, true);
    }
  }

  async handleLocalExtensionInstall(event) {
    event.preventDefault();
    if (!this.extensionsDeveloperMode) {
      this.setAccessFeedback(
        "local-install-feedback",
        "Enable Developer Features to install a local extension.",
        true,
      );
      return;
    }

    const path = document.getElementById("local-extension-path").value.trim();
    if (!path) {
      this.setAccessFeedback(
        "local-install-feedback",
        "Extension path is required.",
        true,
      );
      return;
    }

    try {
      const result = await this.requestJson("/api/extensions/install", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });

      if (!result.ok) {
        throw new Error(
          result.data?.error || `Local install failed (HTTP ${result.status})`,
        );
      }

      this.setAccessFeedback(
        "local-install-feedback",
        "Extension installed.",
        false,
      );
      await this.refreshExtensionsData({ silent: true });
    } catch (error) {
      this.setAccessFeedback("local-install-feedback", error.message, true);
    }
  }

  handleExtensionsListClick(event) {
    const button = event.target.closest("button[data-action]");
    if (!button) {
      return;
    }

    const card = button.closest(".ext-card");
    const extensionId = card?.dataset.extensionId;
    if (!extensionId) {
      return;
    }

    const action = button.dataset.action;
    if (action === "ext-uninstall") {
      this.uninstallExtension(extensionId);
      return;
    }
    if (action === "ext-launch") {
      this.launchExtension(extensionId);
      return;
    }
    if (action === "ext-sync") {
      this.syncExtension(extensionId);
      return;
    }
    if (action === "ext-load-config") {
      this.loadExtensionConfig(extensionId);
      return;
    }
    if (action === "ext-save-config") {
      this.saveExtensionConfig(extensionId);
    }
  }

  getWorkspaceInputs() {
    const workspaceId = document
      .getElementById("extensions-workspace-id")
      .value.trim();
    const tenantId = document
      .getElementById("extensions-tenant-id")
      .value.trim();
    const sourceId = document
      .getElementById("extensions-source-id")
      .value.trim();
    return {
      workspaceId,
      tenantId: tenantId || null,
      sourceId: sourceId || null,
    };
  }

  async uninstallExtension(extensionId) {
    if (!confirm(`Uninstall extension ${extensionId}?`)) {
      return;
    }

    try {
      const result = await this.requestJson(
        `/api/extensions/${encodeURIComponent(extensionId)}/uninstall`,
        { method: "POST" },
      );
      if (!result.ok) {
        throw new Error(
          result.data?.error || `Uninstall failed (HTTP ${result.status})`,
        );
      }

      this.setAccessFeedback(
        "extensions-feedback",
        `Uninstalled ${extensionId}.`,
        false,
      );
      await this.refreshExtensionsData({ silent: true });
    } catch (error) {
      this.setAccessFeedback("extensions-feedback", error.message, true);
    }
  }

  async launchExtension(extensionId) {
    const { workspaceId } = this.getWorkspaceInputs();
    if (!workspaceId) {
      this.setAccessFeedback(
        "extensions-feedback",
        "workspace_id is required to launch a connector.",
        true,
      );
      return;
    }

    try {
      const result = await this.requestJson(
        `/api/extensions/${encodeURIComponent(extensionId)}/launch`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workspace_id: workspaceId }),
        },
      );
      if (!result.ok) {
        throw new Error(
          result.data?.error || `Launch failed (HTTP ${result.status})`,
        );
      }

      this.setAccessFeedback(
        "extensions-feedback",
        `Launched ${extensionId} for workspace ${workspaceId}.`,
        false,
      );
    } catch (error) {
      this.setAccessFeedback("extensions-feedback", error.message, true);
    }
  }

  async syncExtension(extensionId) {
    const { workspaceId, tenantId, sourceId } = this.getWorkspaceInputs();
    if (!workspaceId) {
      this.setAccessFeedback(
        "extensions-feedback",
        "workspace_id is required to sync.",
        true,
      );
      return;
    }

    const payload = {
      workspace_id: workspaceId,
      tenant_id: tenantId,
      source_id: sourceId,
      cursor: null,
    };

    try {
      const result = await this.requestJson(
        `/api/extensions/${encodeURIComponent(extensionId)}/sync`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );
      if (!result.ok) {
        throw new Error(
          result.data?.error || `Sync failed (HTTP ${result.status})`,
        );
      }

      const count = result.data?.result?.chunks_written ?? null;
      const suffix = count !== null ? ` (chunks_written=${count})` : "";
      this.setAccessFeedback(
        "extensions-feedback",
        `Synced ${extensionId}${suffix}.`,
        false,
      );
    } catch (error) {
      this.setAccessFeedback("extensions-feedback", error.message, true);
    }
  }

  findExtensionConfigTextarea(extensionId) {
    const card = document.querySelector(
      `.ext-card[data-extension-id="${CSS.escape(extensionId)}"]`,
    );
    return card?.querySelector("textarea[data-field='ext-config']") || null;
  }

  async loadExtensionConfig(extensionId) {
    const { workspaceId } = this.getWorkspaceInputs();
    if (!workspaceId) {
      this.setAccessFeedback(
        "extensions-feedback",
        "workspace_id is required to load config.",
        true,
      );
      return;
    }

    try {
      const result = await this.requestJson(
        `/api/extensions/${encodeURIComponent(extensionId)}/config?workspace_id=${encodeURIComponent(workspaceId)}`,
      );
      if (!result.ok) {
        throw new Error(
          result.data?.error || `Load config failed (HTTP ${result.status})`,
        );
      }

      const textarea = this.findExtensionConfigTextarea(extensionId);
      if (textarea) {
        textarea.value = JSON.stringify(result.data?.config ?? {}, null, 2);
      }
      this.setAccessFeedback(
        "extensions-feedback",
        `Loaded config for ${extensionId}.`,
        false,
      );
    } catch (error) {
      this.setAccessFeedback("extensions-feedback", error.message, true);
    }
  }

  async saveExtensionConfig(extensionId) {
    const { workspaceId } = this.getWorkspaceInputs();
    if (!workspaceId) {
      this.setAccessFeedback(
        "extensions-feedback",
        "workspace_id is required to save config.",
        true,
      );
      return;
    }

    const textarea = this.findExtensionConfigTextarea(extensionId);
    if (!textarea) {
      return;
    }

    let config = {};
    const raw = textarea.value.trim();
    if (raw) {
      try {
        config = JSON.parse(raw);
      } catch (error) {
        this.setAccessFeedback(
          "extensions-feedback",
          `Config JSON parse error: ${error.message}`,
          true,
        );
        return;
      }
    }

    try {
      const result = await this.requestJson(
        `/api/extensions/${encodeURIComponent(extensionId)}/config`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            workspace_id: workspaceId,
            config,
          }),
        },
      );
      if (!result.ok) {
        throw new Error(
          result.data?.error || `Save config failed (HTTP ${result.status})`,
        );
      }

      this.setAccessFeedback(
        "extensions-feedback",
        `Saved config for ${extensionId}.`,
        false,
      );
    } catch (error) {
      this.setAccessFeedback("extensions-feedback", error.message, true);
    }
  }

  async refreshLegacyTokens({ silent = false } = {}) {
    if (!this.isAuthenticated) {
      return;
    }
    try {
      const result = await this.requestJson("/api/auth/roles");
      if (!result.ok) {
        if (result.status === 401 || result.status === 403) {
          document.getElementById("legacy-reader-token").value = "";
          document.getElementById("legacy-writer-token").value = "";
          document.getElementById("legacy-admin-token").value = "";
          if (!silent) {
            this.setAccessFeedback(
              "legacy-feedback",
              "Admin access required to view legacy role tokens.",
              true,
            );
          }
          return;
        }
        throw new Error(
          result.data?.error ||
            `Legacy role tokens API error (HTTP ${result.status})`,
        );
      }

      const cfg = result.data || {};
      document.getElementById("legacy-reader-token").value =
        cfg.reader_token || "";
      document.getElementById("legacy-writer-token").value =
        cfg.writer_token || "";
      document.getElementById("legacy-admin-token").value =
        cfg.admin_token || "";

      if (!silent) {
        this.setAccessFeedback(
          "legacy-feedback",
          "Loaded legacy tokens.",
          false,
        );
      }
    } catch (error) {
      console.error("Error refreshing legacy tokens:", error);
      if (!silent) {
        this.setAccessFeedback("legacy-feedback", error.message, true);
      }
    }
  }

  async handleLegacyTokensSave(event) {
    event.preventDefault();

    const payload = {
      reader_token:
        document.getElementById("legacy-reader-token").value.trim() || null,
      writer_token:
        document.getElementById("legacy-writer-token").value.trim() || null,
      admin_token:
        document.getElementById("legacy-admin-token").value.trim() || null,
    };

    try {
      const result = await this.requestJson("/api/auth/roles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!result.ok) {
        throw new Error(
          result.data?.error ||
            `Save legacy tokens failed (HTTP ${result.status})`,
        );
      }

      this.setAccessFeedback(
        "legacy-feedback",
        "Legacy tokens updated.",
        false,
      );
      await this.refreshLegacyTokens({ silent: true });
    } catch (error) {
      this.setAccessFeedback("legacy-feedback", error.message, true);
    }
  }

  async refreshAccessData({ silent = false } = {}) {
    if (!this.isAuthenticated) {
      return;
    }
    try {
      const [statusResult, rolesResult, usersResult] = await Promise.all([
        this.requestJson("/api/auth/access/status"),
        this.requestJson("/api/auth/access/roles"),
        this.requestJson("/api/auth/access/users"),
      ]);

      if (!statusResult.ok || !rolesResult.ok || !usersResult.ok) {
        const failing = [statusResult, rolesResult, usersResult].find(
          (result) => !result.ok,
        );
        if (failing && (failing.status === 401 || failing.status === 403)) {
          this.renderAccessUnauthorized();
          if (!silent) {
            this.setAccessFeedback(
              "users-feedback",
              "Admin access required. Save a valid admin API key.",
              true,
            );
          }
          return;
        }
        const message =
          failing?.data?.error ||
          `Access API error (HTTP ${failing?.status || "?"})`;
        throw new Error(message);
      }

      this.renderAccessStatus(statusResult.data);
      this.renderRoleSummary(rolesResult.data || []);
      this.renderUsers(usersResult.data || []);

      if (this.selectedUserId) {
        await this.refreshKeysForSelectedUser({ silent: true });
      }

      this.lastAccessRefreshAt = Date.now();
      if (!silent) {
        this.setAccessFeedback("users-feedback", "", false);
      }
    } catch (error) {
      console.error("Error refreshing access data:", error);
      if (!silent) {
        this.setAccessFeedback("users-feedback", error.message, true);
      }
    }
  }

  renderAccessUnauthorized() {
    document.getElementById("access-auth-enabled").textContent = "unknown";
    document.getElementById("access-user-count").textContent = "-";
    document.getElementById("access-active-key-count").textContent = "-";
    document.getElementById("access-active-admin-key-count").textContent = "-";
    document.getElementById("access-roles").innerHTML =
      '<div class="empty-table">Admin access is required to view role summary.</div>';
    document.getElementById("access-users-body").innerHTML =
      '<tr><td colspan="6" class="empty-table">Admin access is required to view users.</td></tr>';
    document.getElementById("access-keys-body").innerHTML =
      '<tr><td colspan="6" class="empty-table">Select a user after authentication.</td></tr>';
  }

  renderAccessStatus(status) {
    document.getElementById("access-auth-enabled").textContent =
      status.auth_enabled ? "yes" : "no";
    document.getElementById("access-user-count").textContent =
      status.user_count ?? 0;
    document.getElementById("access-active-key-count").textContent =
      status.active_key_count ?? 0;
    document.getElementById("access-active-admin-key-count").textContent =
      status.active_admin_key_count ?? 0;
  }

  renderRoleSummary(roles) {
    const container = document.getElementById("access-roles");
    if (!roles.length) {
      container.innerHTML =
        '<div class="empty-table">No role data available.</div>';
      return;
    }

    container.innerHTML = roles
      .map(
        (role) => `
          <div class="role-card">
            <div class="role-title">${this.escapeHtml(role.role)}</div>
            <div class="role-description">${this.escapeHtml(role.description || "")}</div>
            <div class="role-meta">users: ${role.user_count} · active keys: ${role.active_key_count}</div>
          </div>
        `,
      )
      .join("");
  }

  renderUsers(users) {
    const body = document.getElementById("access-users-body");
    if (!users.length) {
      body.innerHTML =
        '<tr><td colspan="6" class="empty-table">No users found.</td></tr>';
      return;
    }

    body.innerHTML = users
      .map(
        (user) => `
          <tr class="${this.selectedUserId === user.id ? "selected-row" : ""}">
            <td>${this.escapeHtml(user.username)}</td>
            <td>
              <input
                class="access-input compact"
                type="text"
                data-user-id="${this.escapeHtml(user.id)}"
                data-field="display-name"
                value="${this.escapeHtml(user.display_name || "")}" />
            </td>
            <td>
              <select class="access-select compact user-role-select" data-user-id="${this.escapeHtml(user.id)}">
                ${this.renderRoleOptions(user.role)}
              </select>
            </td>
            <td>
              <select class="access-select compact user-status-select" data-user-id="${this.escapeHtml(user.id)}">
                <option value="active" ${user.status === "active" ? "selected" : ""}>active</option>
                <option value="suspended" ${user.status === "suspended" ? "selected" : ""}>suspended</option>
              </select>
            </td>
            <td>${user.active_key_count}/${user.key_count}</td>
            <td>
              <button class="btn-table save-user-btn" data-user-id="${this.escapeHtml(user.id)}" type="button">Save</button>
              <button class="btn-table select-user-btn" data-user-id="${this.escapeHtml(user.id)}" type="button">Keys</button>
            </td>
          </tr>
        `,
      )
      .join("");
  }

  renderRoleOptions(selectedRole) {
    const roles = ["admin", "writer", "reader"];
    return roles
      .map(
        (role) =>
          `<option value="${role}" ${selectedRole === role ? "selected" : ""}>${role}</option>`,
      )
      .join("");
  }

  async handleCreateUser(event) {
    event.preventDefault();

    const payload = {
      username: document.getElementById("new-user-username").value.trim(),
      display_name:
        document.getElementById("new-user-display-name").value.trim() || null,
      role: document.getElementById("new-user-role").value,
      status: document.getElementById("new-user-status").value,
    };

    try {
      const result = await this.requestJson("/api/auth/access/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!result.ok) {
        throw new Error(
          result.data?.error || `Create user failed (HTTP ${result.status})`,
        );
      }

      document.getElementById("new-user-username").value = "";
      document.getElementById("new-user-display-name").value = "";
      this.setAccessFeedback("users-feedback", "User created.", false);
      await this.refreshAccessData({ silent: true });
    } catch (error) {
      this.setAccessFeedback("users-feedback", error.message, true);
    }
  }

  handleUsersTableClick(event) {
    const target = event.target;
    const userId = target.dataset.userId;
    if (!userId) {
      return;
    }

    if (target.classList.contains("save-user-btn")) {
      this.handleUpdateUser(userId);
      return;
    }

    if (target.classList.contains("select-user-btn")) {
      this.selectUser(userId);
    }
  }

  async handleUpdateUser(userId) {
    const roleSelect = document.querySelector(
      `.user-role-select[data-user-id="${CSS.escape(userId)}"]`,
    );
    const statusSelect = document.querySelector(
      `.user-status-select[data-user-id="${CSS.escape(userId)}"]`,
    );
    const displayInput = document.querySelector(
      `input[data-user-id="${CSS.escape(userId)}"][data-field="display-name"]`,
    );

    if (!roleSelect || !statusSelect || !displayInput) {
      return;
    }

    const payload = {
      role: roleSelect.value,
      status: statusSelect.value,
      display_name: displayInput.value.trim() || null,
    };

    try {
      const result = await this.requestJson(
        `/api/auth/access/users/${encodeURIComponent(userId)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );

      if (!result.ok) {
        throw new Error(
          result.data?.error || `Update user failed (HTTP ${result.status})`,
        );
      }

      this.setAccessFeedback("users-feedback", "User updated.", false);
      await this.refreshAccessData({ silent: true });
    } catch (error) {
      this.setAccessFeedback("users-feedback", error.message, true);
    }
  }

  selectUser(userId) {
    this.selectedUserId = userId;
    this.renderSelectedUserLabel();
    this.refreshAccessData({ silent: true });
  }

  renderSelectedUserLabel() {
    const label = document.getElementById("selected-user-label");
    if (!this.selectedUserId) {
      label.textContent = "Select a user to manage API keys.";
      return;
    }

    const row = document.querySelector(
      `.select-user-btn[data-user-id="${CSS.escape(this.selectedUserId)}"]`,
    );
    if (!row) {
      label.textContent = `Selected user id: ${this.selectedUserId}`;
      return;
    }

    const usernameCell = row.closest("tr")?.children?.[0];
    const username = usernameCell
      ? usernameCell.textContent
      : this.selectedUserId;
    label.textContent = `Selected user: ${username}`;
  }

  async refreshKeysForSelectedUser({ silent = false } = {}) {
    if (!this.isAuthenticated) {
      return;
    }
    const keysBody = document.getElementById("access-keys-body");
    if (!this.selectedUserId) {
      keysBody.innerHTML =
        '<tr><td colspan="6" class="empty-table">Select a user to view API keys.</td></tr>';
      return;
    }

    const result = await this.requestJson(
      `/api/auth/access/keys?user_id=${encodeURIComponent(this.selectedUserId)}`,
    );
    if (!result.ok) {
      if (!silent) {
        this.setAccessFeedback(
          "keys-feedback",
          result.data?.error || `Key list failed (HTTP ${result.status})`,
          true,
        );
      }
      return;
    }

    this.renderKeys(result.data || []);
    if (!silent) {
      this.setAccessFeedback("keys-feedback", "", false);
    }
  }

  renderKeys(keys) {
    const body = document.getElementById("access-keys-body");
    if (!keys.length) {
      body.innerHTML =
        '<tr><td colspan="6" class="empty-table">No API keys for this user.</td></tr>';
      return;
    }

    body.innerHTML = keys
      .map(
        (key) => `
          <tr>
            <td>${this.escapeHtml(key.name)}</td>
            <td><code>${this.escapeHtml(key.key_prefix)}</code></td>
            <td>${this.escapeHtml(key.user_role)}</td>
            <td>${key.active ? "active" : "inactive"}</td>
            <td>${this.formatTimestamp(key.expires_at)}</td>
            <td>
              <button class="btn-table danger revoke-key-btn" data-key-id="${this.escapeHtml(key.id)}" type="button" ${key.revoked_at ? "disabled" : ""}>
                ${key.revoked_at ? "Revoked" : "Revoke"}
              </button>
            </td>
          </tr>
        `,
      )
      .join("");
  }

  async handleCreateKey(event) {
    event.preventDefault();
    if (!this.selectedUserId) {
      this.setAccessFeedback("keys-feedback", "Select a user first.", true);
      return;
    }

    const keyNameInput = document.getElementById("new-key-name");
    const scopesInput = document.getElementById("new-key-scopes");
    const expiryInput = document.getElementById("new-key-expiry");

    const scopes = scopesInput.value
      .split(",")
      .map((scope) => scope.trim())
      .filter((scope) => scope.length > 0);

    const expiresInDays = expiryInput.value.trim();
    const payload = {
      name: keyNameInput.value.trim(),
      scopes,
      expires_in_days: expiresInDays ? Number(expiresInDays) : null,
    };

    try {
      const result = await this.requestJson(
        `/api/auth/access/users/${encodeURIComponent(this.selectedUserId)}/keys`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );

      if (!result.ok) {
        throw new Error(
          result.data?.error || `Create key failed (HTTP ${result.status})`,
        );
      }

      keyNameInput.value = "";
      scopesInput.value = "";
      expiryInput.value = "";

      this.showNewKey(result.data?.raw_key || "");
      this.setAccessFeedback("keys-feedback", "API key created.", false);
      await this.refreshAccessData({ silent: true });
    } catch (error) {
      this.setAccessFeedback("keys-feedback", error.message, true);
    }
  }

  showNewKey(rawKey) {
    const container = document.getElementById("new-key-once");
    if (!rawKey) {
      container.style.display = "none";
      container.innerHTML = "";
      return;
    }

    container.style.display = "block";
    container.dataset.rawKey = rawKey;
    container.innerHTML = `
      <div class="new-key-title">Copy this API key now. It will not be shown again.</div>
      <code class="new-key-code">${this.escapeHtml(rawKey)}</code>
      <div class="new-key-actions">
        <button class="btn-table" type="button" data-action="use-key">Use This Key Now</button>
        <button class="btn-table" type="button" data-action="copy-key">Copy</button>
      </div>
    `;
  }

  async handleNewKeyActions(event) {
    const action = event.target.dataset.action;
    if (!action) {
      return;
    }

    const rawKey = document.getElementById("new-key-once").dataset.rawKey || "";
    if (!rawKey) {
      return;
    }

    if (action === "use-key") {
      await this.authenticate(rawKey, { persist: true });
      return;
    }

    if (action === "copy-key") {
      try {
        await navigator.clipboard.writeText(rawKey);
        this.setAccessFeedback("keys-feedback", "API key copied.", false);
      } catch (_) {
        this.setAccessFeedback(
          "keys-feedback",
          "Failed to copy API key.",
          true,
        );
      }
    }
  }

  handleKeysTableClick(event) {
    const target = event.target;
    if (!target.classList.contains("revoke-key-btn")) {
      return;
    }

    const keyId = target.dataset.keyId;
    if (!keyId) {
      return;
    }

    this.revokeKey(keyId);
  }

  async revokeKey(keyId) {
    const confirmed = confirm("Revoke this API key?");
    if (!confirmed) {
      return;
    }

    try {
      const result = await this.requestJson(
        `/api/auth/access/keys/${encodeURIComponent(keyId)}/revoke`,
        {
          method: "POST",
        },
      );

      if (!result.ok) {
        throw new Error(
          result.data?.error || `Revoke key failed (HTTP ${result.status})`,
        );
      }

      this.setAccessFeedback("keys-feedback", "API key revoked.", false);
      await this.refreshAccessData({ silent: true });
    } catch (error) {
      this.setAccessFeedback("keys-feedback", error.message, true);
    }
  }

  updateUI() {
    if (!this.isAuthenticated) {
      return;
    }
    const healthData = this.healthData || { status: "unknown" };
    const modelsData = Array.isArray(this.modelsData) ? this.modelsData : [];
    this.updateSystemStatus(healthData);
    this.updateHeaderStats(healthData, modelsData);
    this.updateModelHistory(modelsData);
    this.updateModelsGrid(modelsData);
    this.updateSystemView(modelsData);
    this.updateHardwareUI();

    if (this.currentModalId !== null) {
      this.updateModalData(this.currentModalId);
    }
  }

  updateSystemStatus(healthData) {
    const statusElement = document.getElementById("system-status");
    const { status } = healthData;

    const normalized = String(status || "").toLowerCase();
    let statusClass = "status-loading";
    let statusText = "Loading";

    if (normalized === "healthy") {
      statusClass = "status-healthy";
      statusText = "Healthy";
    } else if (normalized === "degraded") {
      statusClass = "status-degraded";
      statusText = "Degraded";
    } else if (normalized === "error") {
      statusClass = "status-error";
      statusText = "Error";
    } else if (normalized) {
      statusText = normalized;
    }

    statusElement.innerHTML = `
      <span class="status-dot ${statusClass}"></span>
      <span>${statusText}</span>
    `;
  }

  updateHeaderStats(_healthData, modelsData) {
    const primaryModels = modelsData.filter((m) => (m.replica_id || 0) === 0);
    document.getElementById("total-models").textContent = primaryModels.length;

    const totalActive = primaryModels.reduce(
      (sum, model) => sum + (model.active_inferences || 0),
      0,
    );
    document.getElementById("total-active").textContent = totalActive;
  }

  updateModelsGrid(modelsData) {
    const loadingState = document.getElementById("loading-state");
    const emptyState = document.getElementById("empty-state");
    const modelsGrid = document.getElementById("models-grid");

    loadingState.style.display = "none";

    const primaryModels = modelsData.filter((m) => (m.replica_id || 0) === 0);

    if (!primaryModels.length) {
      emptyState.style.display = "block";
      modelsGrid.style.display = "none";
      return;
    }

    emptyState.style.display = "none";
    modelsGrid.style.display = "grid";

    modelsGrid.innerHTML = primaryModels
      .map((model) => {
        const baseId =
          model.base_model_id !== undefined && model.base_model_id !== null
            ? model.base_model_id
            : model.id;
        const replicas = modelsData.filter((r) => {
          const rid =
            r.base_model_id !== undefined && r.base_model_id !== null
              ? r.base_model_id
              : r.id;
          return rid === baseId;
        });
        return this.createModelCard(model, replicas);
      })
      .join("");
  }

  createModelCard(model, replicas) {
    const status = String(model.status || "").toLowerCase();
    const isHealthy = Boolean(model.healthy);

    let statusText = "Unknown";
    let statusDotClass = "status-degraded";
    if (status === "active") {
      statusText = isHealthy ? "Healthy" : "Unhealthy";
      statusDotClass = isHealthy ? "status-healthy" : "status-error";
    } else if (status === "inactive") {
      statusText = "Stopped";
      statusDotClass = "status-degraded";
    } else if (status === "starting") {
      statusText = "Starting";
      statusDotClass = "status-starting";
    } else if (status === "loading") {
      statusText = "Loading";
      statusDotClass = "status-starting";
    } else if (status === "stopping") {
      statusText = "Stopping";
      statusDotClass = "status-stopping";
    }

    const queueDepth = Array.isArray(model.queue_depth)
      ? model.queue_depth
      : [0, 0];
    const queuePending = Number(queueDepth[0] || 0);
    const queueProcessing = Number(queueDepth[1] || 0);
    const replicaList = Array.isArray(replicas) ? replicas : [];
    const totalReplicas = replicaList.length || 1;
    const activeReplicas = replicaList.filter(
      (r) => String(r.status || "").toLowerCase() !== "inactive",
    ).length;

    const throughput = Number(model.throughput || 0);
    const queueLabel = `${queuePending}/${queueProcessing}`;

    let startStopButton = "";
    if (status === "active") {
      startStopButton = `<button class="btn btn-danger" type="button" onclick="event.stopPropagation(); app.stopModel(${model.id})">Stop</button>`;
    } else if (status === "inactive") {
      startStopButton = `<button class="btn btn-primary" type="button" onclick="event.stopPropagation(); app.startModel(${model.id})">Start</button>`;
    } else {
      startStopButton = `<button class="btn btn-disabled" type="button" disabled>${this.escapeHtml(statusText)}</button>`;
    }

    return `
      <div class="model-card" onclick="app.showModelDetail(${model.id})">
        <div class="model-header">
          <div class="model-info">
            <h3>${this.escapeHtml(model.name)}</h3>
            <div class="model-id">ID: ${model.id}</div>
          </div>
          <div class="model-status">
            <span class="status-dot ${statusDotClass}"></span>
            ${this.escapeHtml(statusText)}
          </div>
        </div>

        <div class="model-metrics">
          <div class="metric">
            <div class="metric-label">Active</div>
            <div class="metric-value">${model.active_inferences}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Total</div>
            <div class="metric-value">${this.formatNumber(model.total_inferences)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Queue</div>
            <div class="metric-value">${queueLabel}</div>
          </div>
          <div class="metric">
            <div class="metric-label">RPS</div>
            <div class="metric-value">${throughput.toFixed(1)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Replicas</div>
            <div class="metric-value">${activeReplicas}/${totalReplicas}</div>
          </div>
        </div>

        <div class="model-details">
          <div class="detail-row">
            <span class="detail-label">Framework:</span>
            <span class="detail-value">${this.escapeHtml(model.framework)}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Version:</span>
            <span class="detail-value">${this.escapeHtml(model.version)}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Device:</span>
            <span class="detail-value">${this.escapeHtml(model.device)}</span>
          </div>
        </div>

        <div class="model-actions">
          <button class="btn btn-secondary" type="button" onclick="event.stopPropagation(); app.showModelDetail(${model.id})">Details</button>
          ${startStopButton}
          <button class="btn btn-danger" type="button" onclick="event.stopPropagation(); app.removeModel(${model.id})">Remove</button>
        </div>
      </div>
    `;
  }

  clearStartModelForm() {
    document.getElementById("start-model-path").value = "";
    document.getElementById("start-model-id").value = "";
    document.getElementById("start-model-topology").value = "data-parallel";
    document.getElementById("start-model-tp-degree").value = "1";
    this.setAccessFeedback("start-model-feedback", "", false);
  }

  persistRemotePlaceholderUrl(value) {
    const trimmed = String(value || "").trim();
    this.remotePlaceholderUrl = trimmed;
    if (trimmed) {
      localStorage.setItem("kapsl_remote_placeholder_url", trimmed);
    } else {
      localStorage.removeItem("kapsl_remote_placeholder_url");
    }
  }

  currentRemotePlaceholderUrl() {
    const input = document.getElementById("engine-remote-url");
    const value = input ? input.value : this.remotePlaceholderUrl;
    const trimmed = String(value || "").trim();
    this.persistRemotePlaceholderUrl(trimmed);
    return trimmed;
  }

  async refreshRemoteArtifacts({ silent = false } = {}) {
    if (!this.isAuthenticated) {
      return;
    }

    const remoteUrl = this.currentRemotePlaceholderUrl();
    const params = new URLSearchParams();
    if (remoteUrl) {
      params.set("remote_url", remoteUrl);
    }

    this.remoteArtifactsLoading = true;
    this.renderRemoteArtifacts();
    if (!silent) {
      this.setAccessFeedback(
        "engine-remote-feedback",
        "Loading remote repository models...",
        false,
      );
    }

    try {
      const path = params.toString()
        ? `/api/engine/remote-artifacts?${params.toString()}`
        : "/api/engine/remote-artifacts";
      const result = await this.requestJson(path);
      if (!result.ok) {
        throw new Error(
          result.data?.error ||
            `Remote artifact request failed (HTTP ${result.status})`,
        );
      }

      const payload = result.data || {};
      this.remoteArtifacts = {
        remote_url: payload.remote_url || remoteUrl,
        repo: payload.repo || "",
        available_repos: Array.isArray(payload.available_repos)
          ? payload.available_repos
          : [],
        models: Array.isArray(payload.models) ? payload.models : [],
      };
      this.renderRemoteArtifacts();
      if (!silent) {
        this.setAccessFeedback("engine-remote-feedback", "", false);
      }
    } catch (error) {
      console.error("Remote artifacts refresh error:", error);
      this.remoteArtifacts = {
        remote_url: remoteUrl,
        repo: "",
        available_repos: [],
        models: [],
      };
      this.renderRemoteArtifacts();
      this.setAccessFeedback("engine-remote-feedback", error.message, true);
    } finally {
      this.remoteArtifactsLoading = false;
      this.renderRemoteArtifacts();
    }
  }

  renderRemoteArtifacts() {
    const repoEl = document.getElementById("engine-remote-repo");
    const countEl = document.getElementById("engine-remote-count");
    const grid = document.getElementById("engine-remote-grid");
    const empty = document.getElementById("engine-remote-empty");
    const models = Array.isArray(this.remoteArtifacts.models)
      ? this.remoteArtifacts.models
      : [];

    repoEl.textContent = this.remoteArtifacts.repo || "-";
    countEl.textContent = String(models.length);

    if (this.remoteArtifactsLoading) {
      empty.style.display = "block";
      empty.querySelector("h3").textContent = "Loading Remote Models";
      empty.querySelector("p").textContent =
        "Fetching your current repository and published models...";
      grid.style.display = "none";
      grid.innerHTML = "";
      return;
    }

    if (!models.length) {
      empty.style.display = "block";
      empty.querySelector("h3").textContent = this.remoteArtifacts.repo
        ? "No Remote Models"
        : "Remote Repository Unavailable";
      empty.querySelector("p").textContent = this.remoteArtifacts.repo
        ? "This repository does not have any published models yet."
        : "Configure a reachable remote URL and authenticate with the remote backend.";
      grid.style.display = "none";
      grid.innerHTML = "";
      return;
    }

    empty.style.display = "none";
    grid.style.display = "grid";
    grid.innerHTML = models.map((model) => this.createRemoteArtifactCard(model)).join("");
  }

  createRemoteArtifactCard(model) {
    const labels = Array.isArray(model.labels) ? model.labels : [];
    const latest = labels[0] || null;
    const latestReference =
      latest?.reference || model.latest_reference || "";
    const latestLabel = latest?.label || model.latest_label || "-";
    const latestSize = latest ? this.formatBytes(latest.size_bytes || 0) : "-";
    const updatedAt = latest?.updated_at
      ? this.formatDateTime(latest.updated_at)
      : "-";

    return `
      <div class="model-card remote-card" data-model-name="${this.escapeHtml(model.name || "")}">
        <div class="model-header">
          <div class="model-info">
            <h3>${this.escapeHtml(model.name || "Unnamed Model")}</h3>
            <div class="model-id">${this.escapeHtml(this.remoteArtifacts.repo || "-")}</div>
          </div>
          <div class="model-status">
            <span class="status-dot status-healthy"></span>
            ${this.escapeHtml(latestLabel)}
          </div>
        </div>

        <div class="model-metrics">
          <div class="metric">
            <div class="metric-label">Artifacts</div>
            <div class="metric-value">${Number(model.artifact_count || labels.length || 0)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Latest Size</div>
            <div class="metric-value">${this.escapeHtml(latestSize)}</div>
          </div>
        </div>

        <div class="model-details">
          <div class="detail-row">
            <span class="detail-label">Target:</span>
            <span class="detail-value">${this.escapeHtml(latestReference || "-")}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Updated:</span>
            <span class="detail-value">${this.escapeHtml(updatedAt)}</span>
          </div>
        </div>

        <div class="model-actions">
          <button class="btn btn-secondary" type="button" data-action="remote-details">Details</button>
          <button class="btn btn-primary" type="button" data-action="remote-pull" data-reference="${this.escapeHtml(latestReference)}" ${latestReference ? "" : "disabled"}>
            Pull
          </button>
        </div>
      </div>
    `;
  }

  handleRemoteArtifactsClick(event) {
    const button = event.target.closest("button[data-action]");
    const card = event.target.closest(".remote-card");
    if (!card) {
      return;
    }

    const modelName = card.dataset.modelName;
    if (!modelName) {
      return;
    }

    if (button) {
      const action = button.dataset.action;
      if (action === "remote-pull") {
        event.stopPropagation();
        const reference = button.dataset.reference;
        if (reference) {
          this.pullRemoteArtifact(reference);
        }
        return;
      }
      if (action === "remote-details") {
        event.stopPropagation();
        this.showRemoteArtifactDetail(modelName);
        return;
      }
    }

    this.showRemoteArtifactDetail(modelName);
  }

  handleModalActionClick(event) {
    const button = event.target.closest("button[data-action]");
    if (!button) {
      return;
    }

    const action = button.dataset.action;
    if (action === "remote-modal-pull" || action === "remote-pull") {
      const reference = button.dataset.reference;
      if (reference) {
        this.pullRemoteArtifact(reference);
      }
    }
  }

  findRemoteArtifactModel(modelName) {
    const models = Array.isArray(this.remoteArtifacts.models)
      ? this.remoteArtifacts.models
      : [];
    return models.find((model) => String(model.name || "") === String(modelName || "")) || null;
  }

  showRemoteArtifactDetail(modelName) {
    const model = this.findRemoteArtifactModel(modelName);
    if (!model) {
      return;
    }

    this.currentRemoteArtifactName = modelName;
    this.currentModalId = null;

    const modalTitle = document.getElementById("modal-title");
    const modalBody = document.getElementById("modal-body");
    const labels = Array.isArray(model.labels) ? model.labels : [];
    const latest = labels[0] || null;

    modalTitle.textContent = `${model.name} (${this.remoteArtifacts.repo || "-"})`;
    modalBody.innerHTML = `
      <div class="modal-section">
        <h3>Repository Artifact</h3>
        <div class="modal-actions-row">
          <div class="model-status">
            <span class="status-dot status-healthy"></span>
            ${this.escapeHtml(latest?.label || model.latest_label || "latest")}
          </div>
          <div class="modal-actions-buttons">
            <button
              class="btn btn-primary"
              type="button"
              data-action="remote-modal-pull"
              data-reference="${this.escapeHtml(latest?.reference || model.latest_reference || "")}"
              ${(latest?.reference || model.latest_reference) ? "" : "disabled"}
            >
              Pull Latest
            </button>
          </div>
        </div>
      </div>

      <div class="modal-section">
        <h3>Model Information</h3>
        <div class="info-grid">
          <div class="info-row"><div class="info-label">Repository</div><div class="info-value">${this.escapeHtml(this.remoteArtifacts.repo || "-")}</div></div>
          <div class="info-row"><div class="info-label">Model</div><div class="info-value">${this.escapeHtml(model.name || "-")}</div></div>
          <div class="info-row"><div class="info-label">Artifacts</div><div class="info-value">${Number(model.artifact_count || labels.length || 0)}</div></div>
          <div class="info-row"><div class="info-label">Remote URL</div><div class="info-value">${this.escapeHtml(this.remoteArtifacts.remote_url || this.currentRemotePlaceholderUrl() || "-")}</div></div>
        </div>
      </div>

      <div class="modal-section">
        <h3>Available Labels</h3>
        <div class="info-grid">
          ${labels
            .map(
              (label) => `
                <div class="info-row artifact-label-row">
                  <div class="artifact-label-copy">
                    <div class="info-label">${this.escapeHtml(label.label || "-")}</div>
                    <div class="info-value artifact-reference">${this.escapeHtml(label.reference || "-")}</div>
                    <div class="artifact-meta">${this.escapeHtml(this.formatBytes(label.size_bytes || 0))} • ${this.escapeHtml(this.formatDateTime(label.updated_at || ""))}</div>
                  </div>
                  <button class="btn btn-secondary" type="button" data-action="remote-modal-pull" data-reference="${this.escapeHtml(label.reference || "")}">
                    Pull
                  </button>
                </div>
              `,
            )
            .join("")}
        </div>
      </div>
    `;

    document.getElementById("model-modal").classList.add("active");
  }

  updateEngineRemoteResult(payload) {
    const block = document.getElementById("engine-remote-result");
    if (!payload) {
      block.textContent = "";
      return;
    }
    block.textContent = JSON.stringify(payload, null, 2);
  }

  async pullRemoteArtifact(reference) {
    const target = String(reference || "").trim();
    if (!target) {
      return;
    }

    const remoteUrl = this.currentRemotePlaceholderUrl();
    const destinationDir = document
      .getElementById("engine-pull-destination")
      .value.trim();
    const payload = { target };
    if (destinationDir) {
      payload.destination_dir = destinationDir;
    }
    if (remoteUrl) {
      payload.remote_url = remoteUrl;
    }

    this.setAccessFeedback("engine-remote-feedback", `Pulling ${target}...`, false);
    this.updateEngineRemoteResult(null);

    try {
      const result = await this.requestJson("/api/engine/pull", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!result.ok) {
        throw new Error(
          result.data?.error || `Pull failed (HTTP ${result.status})`,
        );
      }

      const downloadedBytes = Number(result.data?.bytes_downloaded || 0);
      const kapslPath = result.data?.kapsl_path || "local path unavailable";
      this.setAccessFeedback(
        "engine-remote-feedback",
        `Pulled ${target} (${downloadedBytes.toLocaleString()} bytes) to ${kapslPath}.`,
        false,
      );
      this.updateEngineRemoteResult(result.data);
    } catch (error) {
      this.setAccessFeedback("engine-remote-feedback", error.message, true);
    }
  }

  async handlePushKapsl(event) {
    event.preventDefault();

    const target = document.getElementById("engine-push-target").value.trim();
    const kapslPath = document.getElementById("engine-push-path").value.trim();
    if (!target) {
      this.setAccessFeedback(
        "engine-remote-feedback",
        "Push requires a target in the form repo/model:label.",
        true,
      );
      return;
    }
    if (!kapslPath) {
      this.setAccessFeedback(
        "engine-remote-feedback",
        "Push requires a .aimod file path.",
        true,
      );
      return;
    }

    const remoteUrl = this.currentRemotePlaceholderUrl();
    const payload = { kapsl_path: kapslPath, target };
    if (remoteUrl) {
      payload.remote_url = remoteUrl;
    }

    this.setAccessFeedback(
      "engine-remote-feedback",
      "Pushing package...",
      false,
    );
    this.updateEngineRemoteResult(null);

    try {
      const result = await this.requestJson("/api/engine/push", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!result.ok) {
        throw new Error(
          result.data?.error || `Push failed (HTTP ${result.status})`,
        );
      }

      const uploadedBytes = Number(result.data?.bytes_uploaded || 0);
      this.setAccessFeedback(
        "engine-remote-feedback",
        `Push completed (${uploadedBytes.toLocaleString()} bytes).`,
        false,
      );
      this.updateEngineRemoteResult(result.data);
    } catch (error) {
      this.setAccessFeedback("engine-remote-feedback", error.message, true);
    }
  }

  async handlePullKapsl(event) {
    event.preventDefault();

    const target = document.getElementById("engine-pull-target").value.trim();
    const destinationDir = document
      .getElementById("engine-pull-destination")
      .value.trim();
    const reference = document
      .getElementById("engine-pull-reference")
      .value.trim();

    if (!target) {
      this.setAccessFeedback(
        "engine-remote-feedback",
        "Pull requires a target in the form repo/model:label.",
        true,
      );
      return;
    }

    const remoteUrl = this.currentRemotePlaceholderUrl();
    const payload = { target };
    if (destinationDir) {
      payload.destination_dir = destinationDir;
    }
    if (reference) {
      payload.reference = reference;
    }
    if (remoteUrl) {
      payload.remote_url = remoteUrl;
    }

    this.setAccessFeedback(
      "engine-remote-feedback",
      "Pulling package...",
      false,
    );
    this.updateEngineRemoteResult(null);

    try {
      const result = await this.requestJson("/api/engine/pull", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!result.ok) {
        throw new Error(
          result.data?.error || `Pull failed (HTTP ${result.status})`,
        );
      }

      const downloadedBytes = Number(result.data?.bytes_downloaded || 0);
      this.setAccessFeedback(
        "engine-remote-feedback",
        `Pull completed (${downloadedBytes.toLocaleString()} bytes).`,
        false,
      );
      this.updateEngineRemoteResult(result.data);
    } catch (error) {
      this.setAccessFeedback("engine-remote-feedback", error.message, true);
    }
  }

  async handleStartModel(event) {
    event.preventDefault();

    const modelPath = document.getElementById("start-model-path").value.trim();
    const modelIdRaw = document.getElementById("start-model-id").value.trim();
    const topology =
      document.getElementById("start-model-topology").value || "data-parallel";
    const tpDegreeRaw = document
      .getElementById("start-model-tp-degree")
      .value.trim();
    const tpDegree = Number.parseInt(tpDegreeRaw || "1", 10);

    if (!modelPath) {
      this.setAccessFeedback(
        "start-model-feedback",
        "Model path is required.",
        true,
      );
      return;
    }
    if (!Number.isFinite(tpDegree) || tpDegree < 1) {
      this.setAccessFeedback(
        "start-model-feedback",
        "TP degree must be >= 1.",
        true,
      );
      return;
    }

    const payload = {
      model_path: modelPath,
      topology,
      tp_degree: tpDegree,
    };

    if (modelIdRaw) {
      const modelId = Number.parseInt(modelIdRaw, 10);
      if (!Number.isFinite(modelId) || modelId < 1) {
        this.setAccessFeedback(
          "start-model-feedback",
          "Model ID must be >= 1.",
          true,
        );
        return;
      }
      payload.model_id = modelId;
    }

    try {
      const result = await this.requestJson("/api/models/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!result.ok) {
        throw new Error(
          result.data?.error || `Failed to start model (HTTP ${result.status})`,
        );
      }

      const startedId = result.data?.model_id;
      const idText =
        startedId !== undefined && startedId !== null ? startedId : "unknown";
      this.setAccessFeedback(
        "start-model-feedback",
        `Model load started (id=${idText}).`,
        false,
      );

      // The model loads asynchronously; refresh quickly.
      this.fetchData();
    } catch (error) {
      this.setAccessFeedback("start-model-feedback", error.message, true);
    }
  }

  async startModel(modelId) {
    try {
      const numericId = Number(modelId);
      const cached = (
        Array.isArray(this.modelsData) ? this.modelsData : []
      ).find((m) => Number(m.id) === numericId);
      let modelPath = cached?.model_path || cached?.modelPath || null;

      if (!modelPath) {
        const modelResult = await this.requestJson(
          `/api/models/${encodeURIComponent(modelId)}`,
        );
        if (!modelResult.ok) {
          throw new Error(
            modelResult.data?.error ||
              `Failed to load model details (HTTP ${modelResult.status})`,
          );
        }
        modelPath = modelResult.data?.model_path || null;
      }

      if (!modelPath) {
        throw new Error("Model path unavailable.");
      }

      const result = await this.requestJson("/api/models/start", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_id: modelId,
          model_path: modelPath,
        }),
      });

      if (!result.ok) {
        throw new Error(result.data?.error || "Failed to start model");
      }

      this.fetchData();
    } catch (error) {
      console.error("Error starting model:", error);
      alert(`Failed to start model: ${error.message}`);
    }
  }

  async stopModel(modelId) {
    if (!confirm(`Stop model ${modelId}?`)) {
      return;
    }

    try {
      const result = await this.requestJson(`/api/models/${modelId}/stop`, {
        method: "POST",
      });

      if (!result.ok) {
        throw new Error(result.data?.error || "Failed to stop model");
      }

      this.fetchData();
    } catch (error) {
      console.error("Error stopping model:", error);
      alert(`Failed to stop model: ${error.message}`);
    }
  }

  async removeModel(modelId) {
    if (
      !confirm(
        `Remove model ${modelId}? This unregisters the model and all replicas.`,
      )
    ) {
      return;
    }

    try {
      const result = await this.requestJson(`/api/models/${modelId}/remove`, {
        method: "POST",
      });

      if (!result.ok) {
        throw new Error(result.data?.error || "Failed to remove model");
      }

      if (this.currentModalId !== null) {
        this.closeModal();
      }
      this.fetchData();
    } catch (error) {
      console.error("Error removing model:", error);
      alert(`Failed to remove model: ${error.message}`);
    }
  }

  async showModelDetail(modelId) {
    try {
      const [modelResponse, scalingPolicy] = await Promise.all([
        this.apiFetch(`/api/models/${modelId}`),
        this.fetchScalingPolicy(modelId),
      ]);

      if (!modelResponse.ok) throw new Error("Failed to fetch model details");
      const modelData = await modelResponse.json();

      if (modelData.error) {
        alert(modelData.error);
        return;
      }

      this.currentModalId = modelId;
      this.renderModalContent(modelData, scalingPolicy);

      const modal = document.getElementById("model-modal");
      modal.classList.add("active");
    } catch (error) {
      console.error("Error fetching model details:", error);
      alert("Failed to load model details");
    }
  }

  renderModalContent(model, scalingPolicy) {
    const modalTitle = document.getElementById("modal-title");
    const modalBody = document.getElementById("modal-body");

    modalTitle.textContent = `${model.name} (id=${model.id})`;
    const policy = scalingPolicy || this.currentScalingPolicy;
    this.currentScalingPolicy = policy || null;

    const status = String(model.status || "").toLowerCase();
    const isHealthy = Boolean(model.healthy);

    let statusText = "Unknown";
    let statusDotClass = "status-degraded";
    if (status === "active") {
      statusText = isHealthy ? "Healthy" : "Unhealthy";
      statusDotClass = isHealthy ? "status-healthy" : "status-error";
    } else if (status === "inactive") {
      statusText = "Stopped";
      statusDotClass = "status-degraded";
    } else if (status === "starting") {
      statusText = "Starting";
      statusDotClass = "status-starting";
    } else if (status === "loading") {
      statusText = "Loading";
      statusDotClass = "status-starting";
    } else if (status === "stopping") {
      statusText = "Stopping";
      statusDotClass = "status-stopping";
    }

    const queueDepth = Array.isArray(model.queue_depth)
      ? model.queue_depth
      : [0, 0];
    const queuePending = Number(queueDepth[0] || 0);
    const queueProcessing = Number(queueDepth[1] || 0);
    const successRate =
      model.total_inferences > 0
        ? (
            (model.successful_inferences / model.total_inferences) *
            100
          ).toFixed(2)
        : "0.00";

    const throughput = Number(model.throughput || 0);
    const gpuUtilPct = Number(model.gpu_utilization || 0) * 100;
    const memoryUsageText = this.formatBytes(Number(model.memory_usage || 0));

    const loadedAt =
      model.loaded_at !== undefined && model.loaded_at !== null
        ? Number(model.loaded_at)
        : null;
    const loadedAtText =
      loadedAt && Number.isFinite(loadedAt)
        ? new Date(loadedAt * 1000).toLocaleString()
        : "-";
    const loadedSinceText =
      loadedAt && Number.isFinite(loadedAt) ? this.formatSince(loadedAt) : "-";

    const baseId =
      model.base_model_id !== undefined && model.base_model_id !== null
        ? model.base_model_id
        : model.id;
    const replicaList = (
      Array.isArray(this.modelsData) ? this.modelsData : []
    ).filter((r) => {
      const rid =
        r.base_model_id !== undefined && r.base_model_id !== null
          ? r.base_model_id
          : r.id;
      return rid === baseId;
    });
    const totalReplicas = replicaList.length || 1;
    const activeReplicas = replicaList.filter(
      (r) => String(r.status || "").toLowerCase() !== "inactive",
    ).length;

    let startStopButton = "";
    if (status === "active") {
      startStopButton = `<button class="btn btn-danger" type="button" onclick="app.stopModel(${model.id})">Stop</button>`;
    } else if (status === "inactive") {
      startStopButton = `<button class="btn btn-primary" type="button" onclick="app.startModel(${model.id})">Start</button>`;
    } else {
      startStopButton = `<button class="btn btn-disabled" type="button" disabled>${this.escapeHtml(statusText)}</button>`;
    }

    modalBody.innerHTML = `
      <div class="modal-section">
        <h3>Status & Actions</h3>
        <div class="modal-actions-row">
          <div class="model-status">
            <span class="status-dot ${statusDotClass}"></span>
            ${this.escapeHtml(statusText)}
          </div>
          <div class="modal-actions-buttons">
            ${startStopButton}
            <button class="btn btn-danger" type="button" onclick="app.removeModel(${model.id})">Remove</button>
          </div>
        </div>
      </div>

      <div class="modal-section">
        <h3>Runtime Metrics</h3>
        <div class="modal-metrics">
          <div class="metric">
            <div class="metric-label">Active</div>
            <div class="metric-value">${model.active_inferences}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Total</div>
            <div class="metric-value">${this.formatNumber(model.total_inferences)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value">${successRate}%</div>
          </div>
          <div class="metric">
            <div class="metric-label">Queue (P)</div>
            <div class="metric-value">${queuePending}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Queue (Proc)</div>
            <div class="metric-value">${queueProcessing}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Throughput</div>
            <div class="metric-value">${throughput.toFixed(2)} req/s</div>
          </div>
          <div class="metric">
            <div class="metric-label">GPU Util</div>
            <div class="metric-value">${this.formatPercent(gpuUtilPct)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Engine Memory</div>
            <div class="metric-value">${this.escapeHtml(memoryUsageText)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Replicas</div>
            <div class="metric-value">${activeReplicas}/${totalReplicas}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Successful</div>
            <div class="metric-value">${this.formatNumber(model.successful_inferences)}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Failed</div>
            <div class="metric-value">${this.formatNumber(model.failed_inferences)}</div>
          </div>
        </div>
      </div>

      <div class="modal-section">
        <h3>Trends</h3>
        <div class="modal-charts">
          <div class="metric-card">
            <div class="metric-label">Total Inferences</div>
            <div class="metric-value">${this.formatNumber(model.total_inferences)}</div>
            <canvas id="modal-chart-inferences" class="spark" width="360" height="120"></canvas>
          </div>
          <div class="metric-card">
            <div class="metric-label">Throughput</div>
            <div class="metric-value">${throughput.toFixed(2)} req/s</div>
            <canvas id="modal-chart-throughput" class="spark" width="360" height="120"></canvas>
          </div>
          <div class="metric-card">
            <div class="metric-label">Queue (Pending/Processing)</div>
            <div class="metric-value">${queuePending + queueProcessing}</div>
            <canvas id="modal-chart-queue" class="spark" width="360" height="120"></canvas>
          </div>
        </div>
      </div>

      <div class="modal-section">
        <h3>Model Information</h3>
        <div class="info-grid">
          <div class="info-row"><div class="info-label">Model ID</div><div class="info-value">${model.id}</div></div>
          <div class="info-row"><div class="info-label">Name</div><div class="info-value">${this.escapeHtml(model.name)}</div></div>
          <div class="info-row"><div class="info-label">Framework</div><div class="info-value">${this.escapeHtml(model.framework)}</div></div>
          <div class="info-row"><div class="info-label">Version</div><div class="info-value">${this.escapeHtml(model.version)}</div></div>
          <div class="info-row"><div class="info-label">Device</div><div class="info-value">${this.escapeHtml(model.device)}</div></div>
          <div class="info-row"><div class="info-label">Optimization</div><div class="info-value">${this.escapeHtml(model.optimization_level)}</div></div>
          <div class="info-row"><div class="info-label">Loaded</div><div class="info-value">${this.escapeHtml(loadedAtText)} (${this.escapeHtml(loadedSinceText)})</div></div>
          <div class="info-row"><div class="info-label">Model Path</div><div class="info-value">${this.escapeHtml(model.model_path)}</div></div>
        </div>
      </div>

      <div class="modal-section">
        <h3>Auto-scaling Configuration</h3>
        <div class="scaling-controls" data-model-id="${model.id}">
          <div class="scaling-field">
            <label for="min-replicas">Min Replicas</label>
            <input class="scaling-input" id="min-replicas" type="number" min="1" step="1" value="${policy?.min_replicas ?? ""}">
          </div>
          <div class="scaling-field">
            <label for="max-replicas">Max Replicas</label>
            <input class="scaling-input" id="max-replicas" type="number" min="1" step="1" value="${policy?.max_replicas ?? ""}">
          </div>
          <div class="scaling-field">
            <label for="target-queue-depth">Target Queue</label>
            <input class="scaling-input" id="target-queue-depth" type="number" min="1" step="1" value="${policy?.target_queue_depth ?? ""}">
          </div>
          <div class="scaling-field">
            <label for="scale-down-threshold">Scale-down Threshold</label>
            <input class="scaling-input" id="scale-down-threshold" type="number" min="0" step="1" value="${policy?.scale_down_threshold ?? ""}">
          </div>
          <div class="scaling-field">
            <label for="cooldown-seconds">Cooldown (s)</label>
            <input class="scaling-input" id="cooldown-seconds" type="number" min="1" step="1" value="${policy?.cooldown_seconds ?? ""}">
          </div>
          <button class="btn btn-primary" type="button" onclick="app.saveScalingPolicy(${model.id})">Save Policy</button>
        </div>
        <div class="scaling-status" id="scaling-status"></div>
      </div>
    `;

    this.updateModalCharts(model.id);
  }

  async updateModalData(modelId) {
    if (!document.getElementById("model-modal").classList.contains("active")) {
      return;
    }

    const activeElement = document.activeElement;
    if (activeElement && activeElement.classList.contains("scaling-input")) {
      return;
    }

    try {
      const [modelResponse, scalingPolicy] = await Promise.all([
        this.apiFetch(`/api/models/${modelId}`),
        this.fetchScalingPolicy(modelId),
      ]);

      if (!modelResponse.ok) return;
      const modelData = await modelResponse.json();

      if (!modelData.error) {
        this.renderModalContent(modelData, scalingPolicy);
      }
    } catch (error) {
      console.error("Error updating modal:", error);
    }
  }

  async fetchScalingPolicy(modelId) {
    try {
      const response = await this.apiFetch(`/api/models/${modelId}/scaling`);
      if (!response.ok) {
        return null;
      }
      return await response.json();
    } catch (error) {
      console.error("Error fetching scaling policy:", error);
      return null;
    }
  }

  setScalingStatus(message, isError = false) {
    const statusElement = document.getElementById("scaling-status");
    if (!statusElement) return;
    statusElement.textContent = message;
    statusElement.classList.toggle("error", isError);
    statusElement.classList.toggle("success", !isError);
  }

  async saveScalingPolicy(modelId) {
    const minInput = document.getElementById("min-replicas");
    const maxInput = document.getElementById("max-replicas");
    const targetInput = document.getElementById("target-queue-depth");
    const scaleDownInput = document.getElementById("scale-down-threshold");
    const cooldownInput = document.getElementById("cooldown-seconds");

    if (
      !minInput ||
      !maxInput ||
      !targetInput ||
      !scaleDownInput ||
      !cooldownInput
    ) {
      this.setScalingStatus("Scaling form is not available.", true);
      return;
    }

    const minValue = parseInt(minInput.value, 10);
    const maxValue = parseInt(maxInput.value, 10);
    const targetValue = parseInt(targetInput.value, 10);
    const scaleDownValue = parseInt(scaleDownInput.value, 10);
    const cooldownValue = parseInt(cooldownInput.value, 10);

    if (
      Number.isNaN(minValue) ||
      Number.isNaN(maxValue) ||
      Number.isNaN(targetValue) ||
      Number.isNaN(scaleDownValue) ||
      Number.isNaN(cooldownValue)
    ) {
      this.setScalingStatus(
        "Please enter valid numbers for all scaling fields.",
        true,
      );
      return;
    }

    if (minValue < 1 || maxValue < 1) {
      this.setScalingStatus("Replica counts must be at least 1.", true);
      return;
    }

    if (minValue > maxValue) {
      this.setScalingStatus(
        "Min replicas cannot be greater than max replicas.",
        true,
      );
      return;
    }

    if (targetValue < 1) {
      this.setScalingStatus("Target queue must be at least 1.", true);
      return;
    }

    if (scaleDownValue < 0) {
      this.setScalingStatus("Scale-down threshold cannot be negative.", true);
      return;
    }

    if (cooldownValue < 1) {
      this.setScalingStatus("Cooldown seconds must be at least 1.", true);
      return;
    }

    try {
      let policy = this.currentScalingPolicy;
      if (!policy) {
        policy = await this.fetchScalingPolicy(modelId);
      }
      if (!policy) {
        throw new Error("Scaling policy unavailable.");
      }

      const updatedPolicy = {
        ...policy,
        min_replicas: minValue,
        max_replicas: maxValue,
        target_queue_depth: targetValue,
        scale_down_threshold: scaleDownValue,
        cooldown_seconds: cooldownValue,
      };

      const result = await this.requestJson(`/api/models/${modelId}/scaling`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(updatedPolicy),
      });

      if (!result.ok) {
        throw new Error(
          result.data?.error || "Failed to update scaling policy",
        );
      }

      this.currentScalingPolicy = updatedPolicy;
      this.setScalingStatus("Scaling policy updated.", false);
    } catch (error) {
      console.error("Error updating scaling policy:", error);
      this.setScalingStatus(`Update failed: ${error.message}`, true);
    }
  }

  closeModal() {
    const modal = document.getElementById("model-modal");
    modal.classList.remove("active");
    this.currentModalId = null;
    this.currentRemoteArtifactName = null;
  }

  showError(message) {
    document.getElementById("loading-state").style.display = "none";
    document.getElementById("empty-state").style.display = "none";
    document.getElementById("models-grid").style.display = "none";

    const errorState = document.getElementById("error-state");
    errorState.style.display = "block";
    document.getElementById("error-message").textContent = message;
  }

  hideError() {
    document.getElementById("error-state").style.display = "none";
  }

  formatBytes(bytes) {
    const num = Number(bytes);
    if (!Number.isFinite(num)) {
      return "-";
    }

    const sign = num < 0 ? "-" : "";
    let value = Math.abs(num);
    const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let idx = 0;
    while (value >= 1024 && idx < units.length - 1) {
      value /= 1024;
      idx += 1;
    }

    const decimals = idx === 0 ? 0 : value >= 100 ? 0 : value >= 10 ? 1 : 2;
    return `${sign}${value.toFixed(decimals)} ${units[idx]}`;
  }

  formatPercent(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return "-";
    }
    let pct = num;
    if (pct >= 0 && pct <= 1) {
      pct *= 100;
    }
    return `${pct.toFixed(1)}%`;
  }

  formatSince(unixSeconds) {
    const sec = Number(unixSeconds);
    if (!Number.isFinite(sec) || sec <= 0) {
      return "-";
    }

    const nowSec = Date.now() / 1000;
    let delta = Math.floor(nowSec - sec);
    if (!Number.isFinite(delta) || delta < 0) {
      delta = 0;
    }

    const days = Math.floor(delta / 86400);
    delta -= days * 86400;
    const hours = Math.floor(delta / 3600);
    delta -= hours * 3600;
    const minutes = Math.floor(delta / 60);
    const seconds = delta - minutes * 60;

    if (days > 0) {
      return `${days}d ${hours}h ago`;
    }
    if (hours > 0) {
      return `${hours}h ${minutes}m ago`;
    }
    if (minutes > 0) {
      return `${minutes}m ${seconds}s ago`;
    }
    return `${seconds}s ago`;
  }

  formatTimestamp(value) {
    if (!value) {
      return "never";
    }
    try {
      const date = new Date(value * 1000);
      return date.toLocaleString();
    } catch (_) {
      return "invalid";
    }
  }

  formatDateTime(value) {
    const raw = String(value || "").trim();
    if (!raw) {
      return "-";
    }

    const date = new Date(raw);
    if (Number.isNaN(date.getTime())) {
      return raw;
    }
    return date.toLocaleString();
  }

  escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  formatNumber(num) {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + "M";
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + "K";
    }
    return String(num || 0);
  }
}

let app;
document.addEventListener("DOMContentLoaded", () => {
  app = new KapslApp();
});

window.addEventListener("beforeunload", () => {
  if (app) {
    app.stopAutoRefresh();
  }
});
