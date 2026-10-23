/**
 * Crop Yield Prediction Portal — frontend behaviour.
 *
 * Modules, in order:
 *   1. Tabs      — accessible tablist with keyboard support, deep links, sliding ink
 *   2. Motion    — scroll reveals, count-ups and chart/diagram triggers
 *   3. Sliders   — two-way sync between range inputs and number fields
 *   4. Prediction— calls POST /predict, with a labelled offline fallback
 *
 * Everything degrades safely: without JavaScript all panels are visible, the
 * chart bars render at full height and the stat figures show their real values.
 */

(() => {
  'use strict';

  const prefersReducedMotion = () =>
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ======================================================================
     1. Tabs
     ====================================================================== */
  const TabController = {
    tabs: [],
    panels: [],
    ink: null,
    storageKey: 'cropYield.activeTab',

    init() {
      const tablist = document.querySelector('[role="tablist"]');
      if (!tablist) return;

      this.tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
      this.panels = this.tabs.map((tab) =>
        document.getElementById(tab.getAttribute('aria-controls'))
      );
      this.ink = tablist.querySelector('.tab-ink');
      if (!this.tabs.length) return;

      this.tabs.forEach((tab, index) => {
        tab.addEventListener('click', () => this.select(index));
        tab.addEventListener('keydown', (event) => this.onKeydown(event, index));
      });

      // Deep-link support: #panel-predict, plus the remembered tab.
      const requested = this.fromHash() ?? this.fromStorage() ?? 0;
      this.select(requested, { animate: false, focus: false });

      window.addEventListener('hashchange', () => {
        const index = this.fromHash();
        if (index !== null) this.select(index, { focus: false });
      });

      window.addEventListener('resize', () => this.moveInk());
    },

    fromHash() {
      const id = window.location.hash.replace('#', '');
      if (!id) return null;
      const index = this.panels.findIndex((panel) => panel && panel.id === id);
      return index === -1 ? null : index;
    },

    fromStorage() {
      try {
        const raw = window.localStorage.getItem(this.storageKey);
        if (raw === null) return null;
        const index = Number.parseInt(raw, 10);
        return Number.isInteger(index) && index >= 0 && index < this.tabs.length ? index : null;
      } catch {
        return null; // private mode / storage disabled
      }
    },

    remember(index) {
      try {
        window.localStorage.setItem(this.storageKey, String(index));
      } catch {
        /* non-fatal */
      }
    },

    onKeydown(event, index) {
      const last = this.tabs.length - 1;
      let next = null;

      switch (event.key) {
        case 'ArrowRight': next = index === last ? 0 : index + 1; break;
        case 'ArrowLeft': next = index === 0 ? last : index - 1; break;
        case 'Home': next = 0; break;
        case 'End': next = last; break;
        default: return;
      }

      event.preventDefault();
      this.select(next, { focus: true });
    },

    select(index, { animate = true, focus = true } = {}) {
      if (index === null || index === undefined) return;
      index = Math.max(0, Math.min(index, this.tabs.length - 1));

      this.tabs.forEach((tab, i) => {
        const active = i === index;
        tab.setAttribute('aria-selected', String(active));
        tab.tabIndex = active ? 0 : -1;
        tab.classList.toggle('is-active', active);
        if (this.panels[i]) this.panels[i].hidden = !active;
      });

      const panel = this.panels[index];
      if (panel) {
        panel.classList.remove('is-entering');
        if (animate && !prefersReducedMotion()) {
          void panel.offsetWidth; // restart the animation
          panel.classList.add('is-entering');
        }
        Motion.revealWithin(panel);
      }

      if (focus) this.tabs[index].focus();
      this.moveInk();
      this.remember(index);

      // Keep the URL shareable without stacking history entries.
      const target = `#${this.panels[index].id}`;
      if (window.location.hash !== target) {
        window.history.replaceState(null, '', target);
      }

      // Charts/statistics animate the first time their panel appears.
      Motion.triggerPanelEffects(panel);
    },

    moveInk() {
      if (!this.ink) return;
      const active = this.tabs.find((tab) => tab.getAttribute('aria-selected') === 'true');
      if (!active) return;
      this.ink.style.width = `${active.offsetWidth}px`;
      this.ink.style.transform = `translateX(${active.offsetLeft}px)`;
    },
  };

  /* ======================================================================
     2. Motion
     ====================================================================== */
  const Motion = {
    triggered: new WeakSet(),

    init() {
      const targets = document.querySelectorAll('.reveal');

      if (prefersReducedMotion() || !('IntersectionObserver' in window)) {
        targets.forEach((el) => el.classList.add('is-visible'));
        document.querySelectorAll('.chart, .diagram').forEach((el) => this.animate(el));
        return;
      }

      // Stagger items that sit side by side so groups cascade rather than snap.
      this.applyStagger();

      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          });
        },
        { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
      );

      targets.forEach((el) => observer.observe(el));

      const bandObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) this.animate(entry.target);
          });
        },
        { threshold: 0.3 }
      );

      document.querySelectorAll('.chart, .diagram').forEach((el) => bandObserver.observe(el));
    },

    applyStagger() {
      document.querySelectorAll('.grid-3, .steps, .roadmap').forEach((group) => {
        Array.from(group.children).forEach((child, index) => {
          child.style.setProperty('--delay', `${Math.min(index * 0.07, 0.42)}s`);
        });
      });
    },

    animate(el) {
      if (!el || this.triggered.has(el)) return;
      this.triggered.add(el);

      // The hero artwork animates immediately; chart and diagram on first view.
      el.classList.add(el.classList.contains('chart') ? 'is-animated' : 'is-animated');
    },

    revealWithin(panel) {
      if (!panel) return;
      panel.querySelectorAll('.reveal').forEach((el) => el.classList.add('is-visible'));
    },

    triggerPanelEffects(panel) {
      if (!panel) return;
      panel.querySelectorAll('.chart, .diagram').forEach((el) => this.animate(el));
      Counters.runWithin(panel);
      Counters.runWithin(document.querySelector('.hero-stats'));
    },
  };

  /* ======================================================================
     Count-ups
     ====================================================================== */
  const Counters = {
    done: false,

    runWithin(scope) {
      if (!scope) return;
      const nodes = scope.querySelectorAll('[data-count-to]');
      if (!nodes.length) return;

      nodes.forEach((node) => {
        const target = Number.parseFloat(node.dataset.countTo);
        const decimals = Number.parseInt(node.dataset.decimals ?? '0', 10);
        if (!Number.isFinite(target)) return;

        if (prefersReducedMotion() || !('requestAnimationFrame' in window)) {
          node.textContent = target.toFixed(decimals);
          return;
        }

        const duration = 900;
        const start = performance.now();

        const step = (now) => {
          const progress = Math.min((now - start) / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
          node.textContent = (target * eased).toFixed(decimals);
          if (progress < 1) window.requestAnimationFrame(step);
        };

        window.requestAnimationFrame(step);
      });
    },
  };

  /* ======================================================================
     3. Sliders
     ====================================================================== */
  const Sliders = {
    init() {
      document.querySelectorAll('.slider').forEach((slider) => {
        const numberInput = document.getElementById(slider.dataset.sync);
        if (!numberInput) return;

        this.paint(slider);

        slider.addEventListener('input', () => {
          numberInput.value = slider.value;
          this.paint(slider);
        });

        numberInput.addEventListener('input', () => {
          const value = Number.parseFloat(numberInput.value);
          if (Number.isFinite(value)) {
            slider.value = value;
            this.paint(slider);
          }
        });
      });
    },

    /** Colour the track up to the current value. */
    paint(slider) {
      const min = Number.parseFloat(slider.min);
      const max = Number.parseFloat(slider.max);
      const value = Number.parseFloat(slider.value);
      const span = max - min;
      const percent = span > 0 ? ((value - min) / span) * 100 : 0;
      slider.style.background =
        `linear-gradient(90deg, var(--primary) ${percent}%, #cdd9ce ${percent}%)`;
    },
  };

  /* ======================================================================
     4. Prediction
     ====================================================================== */
  const API_ENDPOINT = '/predict';

  /** Agronomic reference model — mirrors features.heuristic_yield(). */
  const FallbackModel = {
    cropWeights: { rice: 1.12, wheat: 1.0, maize: 0.95, soybean: 0.88 },
    seasonWeights: { rabi: 0.92, kharif: 1.08, zaid: 0.98 },
    baseYield: 0.8,

    predict({ rainfall, temperature, soilNpk, irrigationHours, crop, season }) {
      const tempFactor = 1 + (temperature - 20) * 0.012;
      const rainFactor = 1 + Math.min(Math.max(rainfall, 0) / 2000, 1) * 0.35;
      const soilFactor = 1 + (Math.min(Math.max(soilNpk, 0), 100) / 100) * 0.4;
      const irrigationFactor = 1 + Math.min(Math.max(irrigationHours, 0) / 100, 1) * 0.1;

      return (
        this.baseYield *
        (this.cropWeights[crop] || 1) *
        (this.seasonWeights[season] || 1) *
        tempFactor * rainFactor * soilFactor * irrigationFactor
      );
    },
  };

  const Result = {
    box: null,
    value: null,
    unit: null,
    badge: null,
    total: null,

    init() {
      this.box = document.getElementById('result-box');
      this.value = document.getElementById('result');
      this.unit = document.getElementById('result-unit');
      this.badge = document.getElementById('result-source');
      this.total = document.getElementById('total-output');
    },

    render({ value, unit, badge, badgeKind, total }) {
      this.value.textContent = value;
      this.unit.textContent = unit;
      this.badge.textContent = badge;
      this.badge.className = `result-badge${badgeKind ? ` is-${badgeKind}` : ''}`;
      this.total.textContent = total ?? '';

      this.box.classList.remove('is-pop');
      void this.box.offsetWidth;
      this.box.classList.add('is-pop');
    },

    error(message) {
      this.render({ value: '—', unit: '', badge: message, badgeKind: 'error', total: '' });
    },
  };

  function readForm() {
    const numeric = (id) => Number.parseFloat(document.getElementById(id).value);

    // Clear stale validation styling before re-checking.
    document.querySelectorAll('#prediction-form input[type="number"]').forEach((input) => {
      input.removeAttribute('aria-invalid');
    });

    const payload = {
      rainfall: numeric('rainfall'),
      temperature: numeric('temp'),
      soil_npk: numeric('soil-npk'),
      irrigation_hours: numeric('irrigation'),
      farm_size_hectares: numeric('farm-size'),
      crop: document.getElementById('crop').value,
      season: document.getElementById('season').value,
    };

    for (const [key, value] of Object.entries(payload)) {
      if (typeof value === 'number' && !Number.isFinite(value)) {
        const field = document.getElementById(key.replace('_hours', '').replace('_hectares', ''));
        if (field) field.setAttribute('aria-invalid', 'true');
        throw new Error(`Enter a valid number for “${key.replace(/_/g, ' ')}”.`);
      }
    }
    return payload;
  }

  async function predictViaApi(payload) {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body && typeof body.detail === 'string') detail = body.detail;
      } catch {
        /* keep the status-code message */
      }
      throw new Error(detail);
    }
    return response.json();
  }

  async function onSubmit(event) {
    event.preventDefault();

    let payload;
    try {
      payload = readForm();
    } catch (error) {
      Result.error(error.message);
      return;
    }

    const button = document.getElementById('predict-btn');
    button.disabled = true;
    button.classList.add('is-loading');

    try {
      const result = await predictViaApi(payload);
      Result.render({
        value: result.predicted_yield_tph.toFixed(2),
        unit: 'tonnes/hectare',
        badge: `${result.model_used} · ${Math.round(result.confidence * 100)}% conf.`,
        badgeKind: 'ml',
        total: `≈ ${result.estimated_total_tonnes.toFixed(1)} tonnes across ${payload.farm_size_hectares} ha`,
      });
    } catch (apiError) {
      const yieldTph = FallbackModel.predict({
        rainfall: payload.rainfall,
        temperature: payload.temperature,
        soilNpk: payload.soil_npk,
        irrigationHours: payload.irrigation_hours,
        crop: payload.crop,
        season: payload.season,
      });
      Result.render({
        value: yieldTph.toFixed(2),
        unit: 'tonnes/hectare',
        badge: 'offline estimate · API unavailable',
        badgeKind: 'offline',
        total: `≈ ${(yieldTph * payload.farm_size_hectares).toFixed(1)} tonnes across ${payload.farm_size_hectares} ha`,
      });
      console.warn('Prediction API unavailable, used the reference model:', apiError.message);
    } finally {
      button.disabled = false;
      button.classList.remove('is-loading');
    }
  }

  /* ======================================================================
     Boot
     ====================================================================== */
  document.addEventListener('DOMContentLoaded', () => {
    Result.init();
    Motion.init();
    Sliders.init();
    Counters.runWithin(document.querySelector('.hero-stats'));

    const form = document.getElementById('prediction-form');
    if (form) form.addEventListener('submit', onSubmit);

    TabController.init();
  });
})();
