import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("./gui.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const settle = () => new Promise((resolve) => setImmediate(resolve));

// Run the embedded script with controlled DOM, network responses, and polling.
class Element {
  value = "";
  textContent = "";
  checked = false;
  disabled = false;
  style = {};
  children = [];
  selectors = new Map();
  append(child) { this.children.push(child); }
  querySelector(selector) {
    if (!this.selectors.has(selector)) this.selectors.set(selector, new Element());
    return this.selectors.get(selector);
  }
  querySelectorAll() { return []; }
}

async function fixture() {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  let state = {
    enrolled: true, running: true, busy: "", paused: false,
    earning_mode: "earning", schedule_status: "Schedule allows work",
    connected_workers: 0, total_workers: 2,
    schedule_enabled: false, schedule_timezone: "UTC",
    schedule_windows: [{ start_hour: 18, end_hour: 23, days: [] }],
  };
  let poll;
  const requests = [];
  let respond = async () => ({ ok: true, json: async () => structuredClone(state) });
  const context = vm.createContext({
    document: { getElementById: element, createElement: () => new Element() },
    window: { location: { search: "?token=test-only" }, confirm: () => true },
    Headers, URLSearchParams,
    setInterval: (fn) => { poll = fn; },
    fetch: async (path, options) => {
      requests.push({ path, body: options.body && JSON.parse(options.body) });
      return respond(path, options);
    },
  });
  vm.runInContext(script, context);
  await settle();
  return {
    element, requests,
    setState: (patch) => { state = { ...state, ...patch }; },
    respond: (fn) => { respond = fn; },
    poll: async () => { poll(); await settle(); },
    refresh: () => vm.runInContext("refresh()", context),
    response: () => ({ ok: true, json: async () => structuredClone(state) }),
  };
}

test("polling preserves checkbox, timezone, and window drafts", async () => {
  const f = await fixture();
  f.element("sched-enabled").checked = true;
  f.element("sched-enabled").onchange();
  f.element("sched-tz").value = "America/New_York";
  f.element("sched-tz").oninput();
  f.element("add-window").onclick();
  await f.poll();
  assert.equal(f.element("sched-enabled").checked, true);
  assert.equal(f.element("sched-tz").value, "America/New_York");
  await f.element("save-schedule").onclick();
  const saved = f.requests.find((r) => r.path === "/api/schedule").body;
  assert.equal(saved.enabled, true);
  assert.equal(saved.timezone, "America/New_York");
  assert.equal(saved.windows.length, 2);
});

test("failed save preserves draft and permits retry", async () => {
  const f = await fixture();
  f.element("sched-tz").value = "America/New_York";
  f.element("sched-tz").oninput();
  f.respond(async (path) => path === "/api/schedule"
    ? { ok: false, json: async () => ({ error: "cannot save config" }) }
    : f.response());
  await f.element("save-schedule").onclick();
  await f.poll();
  assert.equal(f.element("sched-tz").value, "America/New_York");
  assert.equal(f.element("error").textContent, "cannot save config");
  assert.equal(f.element("save-schedule").disabled, false);
});

test("edits made during save survive its response and subsequent polls", async () => {
  const f = await fixture();
  let finishSave;
  f.respond((path) => path === "/api/schedule"
    ? new Promise((resolve) => { finishSave = resolve; })
    : f.response());
  const saving = f.element("save-schedule").onclick();
  assert.equal(f.element("save-schedule").disabled, true);
  f.element("sched-tz").value = "America/New_York";
  f.element("sched-tz").oninput();
  await f.poll();
  finishSave(f.response());
  await saving;
  await f.poll();
  assert.equal(f.element("sched-tz").value, "America/New_York");
});

test("successful save adopts server state and ignores older in-flight polling", async () => {
  const f = await fixture();
  let finishPoll;
  f.respond((path) => path === "/api/state"
    ? new Promise((resolve) => { finishPoll = resolve; })
    : f.response());
  const stale = f.refresh();
  f.element("sched-enabled").checked = true;
  f.element("sched-enabled").onchange();
  f.setState({ schedule_enabled: true });
  await f.element("save-schedule").onclick();
  finishPoll({ ok: true, json: async () => ({ schedule_enabled: false }) });
  await stale;
  assert.equal(f.element("sched-enabled").checked, true);
  f.respond(async () => f.response());
  f.setState({ schedule_timezone: "Europe/London" });
  await f.poll();
  assert.equal(f.element("sched-tz").value, "Europe/London");
});

test("connection status distinguishes connecting, partial, connected, pause, and stop", async () => {
  const f = await fixture();
  const headline = () => f.element("summary").querySelector("strong").textContent;
  assert.equal(headline(), "Connecting to pool");
  for (const connected of [1, 2]) {
    f.setState({ connected_workers: connected });
    await f.poll();
    assert.equal(headline(), `${connected} of 2 GPU workers connected`);
    assert.doesNotMatch(headline(), /earning/i);
  }
  f.setState({ connected_workers: 0, paused: true });
  await f.poll();
  assert.equal(headline(), "Paused");
  f.setState({ paused: false, earning_mode: "scheduled_pause" });
  await f.poll();
  assert.equal(headline(), "Outside scheduled hours");
  f.setState({ running: false });
  await f.poll();
  assert.equal(headline(), "Enrolled — click Start to connect");
});

test("failed polling cannot leave stale connected status on screen", async () => {
  const f = await fixture();
  f.setState({ connected_workers: 2 });
  await f.poll();
  f.respond(async () => { throw new Error("network unavailable"); });
  await f.poll();
  assert.equal(f.element("summary").querySelector("strong").textContent, "Agent status unavailable");
  assert.equal(f.element("error").textContent, "network unavailable");
  f.respond(async () => f.response());
  await f.poll();
  assert.equal(f.element("summary").querySelector("strong").textContent, "2 of 2 GPU workers connected");
  assert.equal(f.element("error").textContent, "");
});
