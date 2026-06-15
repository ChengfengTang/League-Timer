/* League Timer frontend: connects to /ws, renders champion cards, sends
   add/remove/trigger/reset commands, and counts timers down smoothly between
   the server's periodic state broadcasts. */

(() => {
  "use strict";

  const connEl = document.getElementById("conn");
  const cardsEl = document.getElementById("cards");
  const emptyEl = document.getElementById("empty");
  const form = document.getElementById("add-form");
  const input = document.getElementById("add-input");

  let ws = null;
  let champions = [];
  let recvAt = perfSec();
  const cardEls = new Map();

  function perfSec() {
    return performance.now() / 1000;
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => {
      connEl.textContent = "connected";
      connEl.className = "conn conn--on";
    };
    ws.onclose = () => {
      connEl.textContent = "reconnecting…";
      connEl.className = "conn conn--off";
      setTimeout(connect, 1000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "state") {
        champions = msg.champions || [];
        recvAt = perfSec();
        reconcile();
      }
    };
  }

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const name = input.value.trim();
    if (!name) return;
    send({ type: "add_champion", name });
    input.value = "";
  });

  function statInput(label, min, max, onChange) {
    const wrap = document.createElement("label");
    wrap.className = "stat";
    const lbl = document.createElement("span");
    lbl.className = "stat__label";
    lbl.textContent = label;
    const inp = document.createElement("input");
    inp.type = "number";
    inp.min = String(min);
    if (max != null) inp.max = String(max);
    inp.addEventListener("change", () => {
      const v = Number(inp.value);
      if (!Number.isFinite(v)) return;
      onChange(v);
    });
    wrap.append(lbl, inp);
    return { wrap, inp };
  }

  function reconcile() {
    emptyEl.style.display = champions.length ? "none" : "block";

    const seen = new Set();
    for (const champ of champions) {
      seen.add(champ.id);
      if (!cardEls.has(champ.id)) buildCard(champ);
    }
    for (const [id, entry] of cardEls) {
      if (!seen.has(id)) {
        entry.root.remove();
        cardEls.delete(id);
      }
    }
    update();
  }

  function buildCard(champ) {
    const root = document.createElement("div");
    root.className = "card";

    const head = document.createElement("div");
    head.className = "card__head";

    const name = document.createElement("span");
    name.className = "card__name";
    name.textContent = champ.name;

    const badge = document.createElement("span");
    badge.className = "badge";

    const status = document.createElement("span");
    status.className = "card__status";

    const remove = document.createElement("button");
    remove.className = "remove";
    remove.title = "Remove champion";
    remove.textContent = "\u00d7";
    remove.addEventListener("click", () =>
      send({ type: "remove_champion", id: champ.id })
    );

    head.append(name, badge, status, remove);
    root.appendChild(head);

    const stats = document.createElement("div");
    stats.className = "stats";

    const level = statInput("Lvl", 1, 18, (v) =>
      send({ type: "set_level", id: champ.id, level: v })
    );
    const ah = statInput("AH", 0, null, (v) =>
      send({ type: "set_ability_haste", id: champ.id, haste: v })
    );
    const ssh = statInput("SSH", 0, null, (v) =>
      send({ type: "set_summoner_haste", id: champ.id, haste: v })
    );
    stats.append(level.wrap, ah.wrap, ssh.wrap);
    root.appendChild(stats);

    const slots = new Map();

    const addGroup = (label, items, isSummoner) => {
      if (!items.length) return;
      const lbl = document.createElement("div");
      lbl.className = "row-label";
      lbl.textContent = label;
      root.appendChild(lbl);

      const wrap = document.createElement("div");
      wrap.className = "slots";
      for (const slot of items) {
        const cell = document.createElement("div");
        cell.className = "slot-cell";

        const el = document.createElement("div");
        el.className = "slot" + (isSummoner ? " slot--summoner" : "");
        el.title = `${slot.label} — ${slot.effective_cd}s CD · click to start, right-click to reset`;

        const key = document.createElement("div");
        key.className = "slot__key";
        key.textContent = slot.key;

        const time = document.createElement("div");
        time.className = "slot__time";

        const fill = document.createElement("div");
        fill.className = "slot__fill";

        el.append(key, time, fill);
        el.addEventListener("click", () =>
          send({ type: "trigger", id: champ.id, key: slot.key })
        );
        el.addEventListener("contextmenu", (e) => {
          e.preventDefault();
          send({ type: "reset", id: champ.id, key: slot.key });
        });
        cell.appendChild(el);

        let rankSel = null;
        if (!isSummoner) {
          rankSel = document.createElement("select");
          rankSel.className = "rank";
          rankSel.title = "Ability rank";
          for (let r = 1; r <= 5; r++) {
            const opt = document.createElement("option");
            opt.value = String(r);
            opt.textContent = `R${r}`;
            rankSel.appendChild(opt);
          }
          rankSel.addEventListener("change", () =>
            send({
              type: "set_ability_rank",
              id: champ.id,
              key: slot.key,
              rank: Number(rankSel.value),
            })
          );
          cell.appendChild(rankSel);
        }

        wrap.appendChild(cell);
        slots.set(slot.key, { el, timeEl: time, fillEl: fill, rankSel });
      }
      root.appendChild(wrap);
    };

    addGroup("Abilities", champ.abilities, false);
    addGroup("Summoners", champ.summoners, true);

    cardsEl.appendChild(root);
    cardEls.set(champ.id, {
      root,
      statusEl: status,
      badgeEl: badge,
      slots,
      levelIn: level.inp,
      ahIn: ah.inp,
      sshIn: ssh.inp,
    });
  }

  function syncStat(inp, value) {
    if (document.activeElement === inp) return;
    inp.value = String(value);
  }

  function fmt(remaining) {
    const s = Math.ceil(remaining);
    if (s >= 60) {
      const m = Math.floor(s / 60);
      const r = s % 60;
      return `${m}:${String(r).padStart(2, "0")}`;
    }
    return `${s}`;
  }

  function applyBadge(entry, champ) {
    const b = entry.badgeEl;
    const st = champ.detector_status || {};
    if (champ.auto && st.error) {
      b.className = "badge badge--error";
      b.textContent = "model error";
    } else if (champ.auto && st.loading) {
      b.className = "badge badge--loading";
      b.textContent = "loading…";
    } else if (champ.auto) {
      b.className = "badge badge--auto";
      b.textContent = "auto";
    } else {
      b.className = "badge badge--manual";
      b.textContent = "manual";
    }

    let line = "";
    if (champ.auto && st.capture_fps != null && !st.error) {
      line = `${st.capture_fps.toFixed(0)} fps`;
      if (st.top1) line += ` · ${st.top1} ${(st.top1_score ?? 0).toFixed(2)}`;
    } else if (st.error) {
      line = st.error;
    }
    entry.statusEl.textContent = line;
  }

  function update() {
    const elapsed = perfSec() - recvAt;
    for (const champ of champions) {
      const entry = cardEls.get(champ.id);
      if (!entry) continue;

      syncStat(entry.levelIn, champ.level ?? 1);
      syncStat(entry.ahIn, champ.ability_haste ?? 0);
      syncStat(entry.sshIn, champ.summoner_haste ?? 0);
      applyBadge(entry, champ);

      const all = [...champ.abilities, ...champ.summoners];
      for (const slot of all) {
        const ref = entry.slots.get(slot.key);
        if (!ref) continue;

        if (ref.rankSel && document.activeElement !== ref.rankSel) {
          ref.rankSel.value = String(slot.rank ?? 1);
        }

        const remaining = Math.max(0, (slot.remaining || 0) - elapsed);
        const ticking = remaining > 0.05;
        ref.el.classList.toggle("slot--ticking", ticking);
        ref.el.title = `${slot.label} — ${slot.effective_cd}s CD · click to start, right-click to reset`;
        ref.timeEl.textContent = ticking ? fmt(remaining) : "";
        const pct = ticking && slot.total ? (remaining / slot.total) * 100 : 0;
        ref.fillEl.style.width = `${pct}%`;
      }
    }
  }

  setInterval(update, 100);
  connect();
})();
