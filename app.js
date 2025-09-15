// ASCII wireframe
// ------------------------------------------------------------
//  +-------------------------------------------------------+
//  | Hydroponic Mix Planner                               |
//  +----------------------+  +---------------------------+
//  | Tap Water Preset     |  | Chemicals                |
//  | (RO, South FL,       |  | [ ] GP3 Grow (N,P,K,...) |
//  |  Custom)             |  | [ ] GP3 Bloom            |
//  |  Ca  Mg  Na  K  Alk  |  | [ ] GP3 Micro            |
//  |  pH  (if Custom)     |  | [ ] Nitric Acid          |
//  | [Save Custom]        |  | + Add Chemical v         |
//  +----------------------+  +---------------------------+
//  +----------------------+  +---------------------------+
//  | Selected Chemicals    |  | Actions                  |
//  |  - GP3 Micro          |  | [Export JSON] [Import]   |
//  |  - Nitric Acid        |  |                          |
//  +----------------------+  +---------------------------+
// ------------------------------------------------------------

(function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const STORAGE_KEYS = {
    chemicals: 'hp_chemicals_v1',
    selected: 'hp_selected_v1',
    waterPreset: 'hp_water_preset_v1',
    customWater: 'hp_custom_water_v1'
  };

  const DEFAULT_CHEMICALS = [
    { name: 'GP3 Grow', per_ml_ppm: { N: 20.0, P: 1.0 * 0.436 * 10.0, K: 6.0 * 0.83 * 10.0, Ca: 0.0, Mg: 5.0, Fe: 0.0 }, alk_change_mg_per_ml: 0.0, ml_bounds: [0.0, 10.0] },
    { name: 'GP3 Bloom', per_ml_ppm: { N: 0.0, P: 5.0 * 0.436 * 10.0, K: 4.0 * 0.83 * 10.0, Ca: 0.0, Mg: 15.0, Fe: 0.0 }, alk_change_mg_per_ml: 0.0, ml_bounds: [0.0, 10.0] },
    { name: 'GP3 Micro', per_ml_ppm: { N: 50.0, P: 0.0, K: 1.0 * 0.83 * 10.0, Ca: 60.0, Mg: 0.0, Fe: 1.0 }, alk_change_mg_per_ml: 0.0, ml_bounds: [0.0, 10.0] },
    { name: 'Nitric Acid', per_ml_ppm: { N: 150.0 }, alk_change_mg_per_ml: -400.0, ml_bounds: [0.0, 5.0] }
  ];

  const WATER_PRESETS = {
    ro_di: { Ca: 0, Mg: 0, Na: 1, K: 0, Alk: 10, pH: 6.0 },
    south_florida: { Ca: 80, Mg: 20, Na: 25, K: 1, Alk: 140, pH: 7.8 }
  };

  function loadState() {
    const chemicals = JSON.parse(localStorage.getItem(STORAGE_KEYS.chemicals) || 'null') || DEFAULT_CHEMICALS;
    const selected = JSON.parse(localStorage.getItem(STORAGE_KEYS.selected) || '[]');
    const waterPreset = localStorage.getItem(STORAGE_KEYS.waterPreset) || 'ro_di';
    const customWater = JSON.parse(localStorage.getItem(STORAGE_KEYS.customWater) || 'null') || WATER_PRESETS.ro_di;
    return { chemicals, selected, waterPreset, customWater };
  }

  function saveState(partial) {
    if (partial.chemicals) localStorage.setItem(STORAGE_KEYS.chemicals, JSON.stringify(partial.chemicals));
    if (partial.selected) localStorage.setItem(STORAGE_KEYS.selected, JSON.stringify(partial.selected));
    if (partial.waterPreset) localStorage.setItem(STORAGE_KEYS.waterPreset, partial.waterPreset);
    if (partial.customWater) localStorage.setItem(STORAGE_KEYS.customWater, JSON.stringify(partial.customWater));
  }

  function renderChemicals(state) {
    const container = $('#chemicals-list');
    container.innerHTML = '';
    state.chemicals.forEach((chem, idx) => {
      const row = document.createElement('div');
      row.className = 'chem-row';

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = state.selected.includes(idx);
      cb.addEventListener('change', () => {
        const selected = new Set(state.selected);
        if (cb.checked) selected.add(idx); else selected.delete(idx);
        state.selected = Array.from(selected);
        saveState({ selected: state.selected });
        renderSelected(state);
      });

      const name = document.createElement('div');
      name.className = 'chem-name';
      name.textContent = chem.name;

      const meta = document.createElement('div');
      meta.className = 'chem-meta';
      const c = chem.per_ml_ppm || {};
      meta.textContent = `N ${c.N||0} | P ${c.P||0} | K ${c.K||0} | Ca ${c.Ca||0} | Mg ${c.Mg||0} | Fe ${c.Fe||0} ppm/ml/L`;

      const actions = document.createElement('div');
      actions.className = 'chem-actions';
      const del = document.createElement('button');
      del.className = 'icon-btn danger';
      del.title = 'Delete chemical';
      del.textContent = 'Delete';
      del.addEventListener('click', () => {
        if (!confirm(`Delete ${chem.name}?`)) return;
        // Reindex selection after deletion
        const newChemicals = state.chemicals.filter((_, i) => i !== idx);
        const mapping = new Map();
        state.chemicals.forEach((_, i) => { if (i !== idx) mapping.set(i, i - (i > idx ? 1 : 0)); });
        const newSelected = state.selected
          .filter((i) => i !== idx)
          .map((i) => mapping.get(i))
          .filter((i) => i != null);
        state.chemicals = newChemicals;
        state.selected = newSelected;
        saveState({ chemicals: state.chemicals, selected: state.selected });
        renderChemicals(state);
        renderSelected(state);
      });
      actions.appendChild(del);

      container.appendChild(cb);
      const block = document.createElement('div');
      block.appendChild(name);
      block.appendChild(meta);
      container.appendChild(block);
      container.appendChild(actions);
    });
  }

  function renderSelected(state) {
    const ul = $('#selected-chemicals');
    ul.innerHTML = '';
    state.selected.forEach((idx) => {
      const chem = state.chemicals[idx];
      if (!chem) return;
      const li = document.createElement('li');
      li.innerHTML = `<div>${chem.name}</div><div class="chem-meta">alk Δ ${chem.alk_change_mg_per_ml||0} mg/L per ml/L · dose ${chem.ml_bounds?.[0]||0}-${chem.ml_bounds?.[1]||0} ml/L</div>`;
      ul.appendChild(li);
    });
  }

  function hookAddChemical(state) {
    const form = $('#add-chem-form');
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const fd = new FormData(form);
      const name = String(fd.get('name')||'').trim();
      if (!name) return alert('Name is required');
      const num = (k) => Number(fd.get(k) || 0);
      const chem = {
        name,
        per_ml_ppm: { N: num('N'), P: num('P'), K: num('K'), Ca: num('Ca'), Mg: num('Mg'), Fe: num('Fe') },
        alk_change_mg_per_ml: Number(fd.get('alk') || 0),
        ml_bounds: [Number(fd.get('min') || 0), Number(fd.get('max') || 10)]
      };
      state.chemicals.push(chem);
      saveState({ chemicals: state.chemicals });
      form.reset();
      $('#add-chem-details').open = false;
      renderChemicals(state);
    });
  }

  function hookWaterPresets(state) {
    const radios = $$('input[name="waterPreset"]');
    const customWrap = $('#custom-water');
    const fields = {
      Ca: $('#water-ca'), Mg: $('#water-mg'), Na: $('#water-na'), K: $('#water-k'), Alk: $('#water-alk'), pH: $('#water-ph')
    };

    function applyFields(obj) {
      fields.Ca.value = obj.Ca;
      fields.Mg.value = obj.Mg;
      fields.Na.value = obj.Na;
      fields.K.value = obj.K;
      fields.Alk.value = obj.Alk;
      fields.pH.value = obj.pH;
    }

    function readFields() {
      return {
        Ca: Number(fields.Ca.value || 0),
        Mg: Number(fields.Mg.value || 0),
        Na: Number(fields.Na.value || 0),
        K: Number(fields.K.value || 0),
        Alk: Number(fields.Alk.value || 0),
        pH: Number(fields.pH.value || 0)
      };
    }

    radios.forEach((r) => {
      r.addEventListener('change', () => {
        state.waterPreset = r.value;
        saveState({ waterPreset: state.waterPreset });
        if (r.value === 'custom') {
          customWrap.style.display = '';
          applyFields(state.customWater);
        } else {
          customWrap.style.display = 'none';
          const preset = WATER_PRESETS[r.value];
          applyFields(preset);
        }
      });
    });

    // Initialize
    const selected = $(`input[name="waterPreset"][value="${state.waterPreset}"]`) || $('input[name="waterPreset"][value="ro_di"]');
    if (selected) selected.checked = true;
    if (state.waterPreset === 'custom') {
      customWrap.style.display = '';
      applyFields(state.customWater);
    } else {
      customWrap.style.display = 'none';
      applyFields(WATER_PRESETS[state.waterPreset] || WATER_PRESETS.ro_di);
    }

    $('#save-custom-water').addEventListener('click', () => {
      const obj = readFields();
      state.customWater = obj;
      state.waterPreset = 'custom';
      saveState({ customWater: obj, waterPreset: 'custom' });
      $(`input[name="waterPreset"][value="custom"]`).checked = true;
      customWrap.style.display = '';
    });
  }

  function hookExportImport(state) {
    $('#export').addEventListener('click', () => {
      const payload = {
        chemicals: state.chemicals,
        selected: state.selected,
        water: { preset: state.waterPreset, custom: state.customWater }
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'hydroponic_mix_planner.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });

    $('#import').addEventListener('click', () => $('#import-file').click());
    $('#import-file').addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        if (Array.isArray(data.chemicals)) state.chemicals = data.chemicals;
        if (Array.isArray(data.selected)) state.selected = data.selected;
        if (data.water) {
          state.waterPreset = data.water.preset || 'ro_di';
          state.customWater = data.water.custom || WATER_PRESETS.ro_di;
        }
        saveState({ chemicals: state.chemicals, selected: state.selected, waterPreset: state.waterPreset, customWater: state.customWater });
        // Refresh UI
        renderChemicals(state);
        renderSelected(state);
        // Update water UI
        const radio = $(`input[name="waterPreset"][value="${state.waterPreset}"]`);
        if (radio) radio.click(); else {
          $(`input[name="waterPreset"][value="ro_di"]`).click();
        }
      } catch (err) {
        alert('Import failed: ' + err.message);
      } finally {
        e.target.value = '';
      }
    });
  }

  function hookReset(state) {
    $('#reset').addEventListener('click', () => {
      if (!confirm('Reset to defaults? This clears your chemicals, selection, and custom water.')) return;
      localStorage.removeItem(STORAGE_KEYS.chemicals);
      localStorage.removeItem(STORAGE_KEYS.selected);
      localStorage.removeItem(STORAGE_KEYS.waterPreset);
      localStorage.removeItem(STORAGE_KEYS.customWater);
      const fresh = loadState();
      Object.assign(state, fresh);
      // Re-render all
      renderChemicals(state);
      renderSelected(state);
      const radio = $(`input[name="waterPreset"][value="${state.waterPreset}"]`);
      if (radio) radio.click();
    });
  }

  // Boot
  const state = loadState();
  renderChemicals(state);
  renderSelected(state);
  hookAddChemical(state);
  hookWaterPresets(state);
  hookExportImport(state);
  hookReset(state);
})();


