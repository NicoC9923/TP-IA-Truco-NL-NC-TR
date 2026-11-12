// The robots' speech, shared by the human game (voz.html) and spectator mode (espectador.html).
//
// The server sends each phrase once in a `voz` list: {id (increasing), asiento (1 | 2),
// frase, perfil} where `perfil` comes from voz_jugador.perfil_voz (preferred voice names,
// pitch, rate). Only what a player says aloud ever reaches this list (ADR 0066).
//
//   const voz = VozRobots(() => vozActiva);
//   voz.procesar(data.voz);   // speaks the new phrases; the first call only takes the last id
//   voz.repetir();            // says the last phrase again
//   voz.reiniciar();          // a new list (e.g. after a server restart): skip what it has
(function () {
  const hayVoz = () => 'speechSynthesis' in window;
  const sinAcento = (t) => String(t || '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
  // Whole words: "male" must not match "Female".
  const palabras = (t) => ` ${sinAcento(t).replace(/[^a-z0-9]+/g, ' ').trim()} `;
  let vocesEs = [];
  function cargarVoces() {
    try { vocesEs = (window.speechSynthesis.getVoices() || []).filter(v => /^es/i.test(v.lang)); } catch (e) { vocesEs = []; }
  }
  if (hayVoz()) {
    cargarVoces();
    try { window.speechSynthesis.addEventListener('voiceschanged', cargarVoces); } catch (e) {}
  }

  // Preferred names first (es-AR, then any Spanish), then any es-AR, then any Spanish voice.
  function elegirVoz(perfil, evitar = null) {
    const prefs = (perfil?.preferencias || []).map(p => palabras(p));
    const todas = evitar && vocesEs.some(v => v !== evitar) ? vocesEs.filter(v => v !== evitar) : vocesEs;
    const ar = todas.filter(v => /^es[-_]AR/i.test(v.lang));
    const buscar = (lista) => {
      for (const p of prefs) { const v = lista.find(x => palabras(x.name).includes(p)); if (v) return v; }
      return null;
    };
    return buscar(ar) || buscar(todas) || ar[0] || todas[0] || null;
  }

  window.VozRobots = function (estaActiva = () => true) {
    const perfilesAsiento = {};
    let ultimoId = null;          // null until the first list: the history is not replayed on load
    let ultima = null;

    function vozDeAsiento(perfil, asiento) {
      if (perfil) perfilesAsiento[asiento] = perfil;
      // Seat 2 keeps a different voice from seat 1 when there is one.
      const evitar = asiento !== 1 && perfilesAsiento[1] ? elegirVoz(perfilesAsiento[1]) : null;
      return elegirVoz(perfil, evitar);
    }
    function hablar(frase, perfil, asiento = 1) {
      if (!frase || !estaActiva() || !hayVoz()) return;
      try {
        const u = new SpeechSynthesisUtterance(frase);
        const voz = vozDeAsiento(perfil, asiento);
        if (voz) u.voice = voz;
        u.lang = voz?.lang || perfil?.idioma || 'es-AR';
        u.pitch = perfil?.tono ?? 1;
        u.rate = perfil?.velocidad ?? 1;
        window.speechSynthesis.speak(u);
      } catch (e) {}
    }
    function procesar(lista) {
      lista = Array.isArray(lista) ? lista : [];
      const maxId = lista.reduce((m, v) => Math.max(m, v.id), 0);
      if (lista.length) ultima = lista[lista.length - 1];
      if (ultimoId === null || maxId < ultimoId) { ultimoId = maxId; return; }
      lista.filter(v => v.id > ultimoId).forEach(v => hablar(v.frase, v.perfil, v.asiento ?? 1));
      ultimoId = maxId;
    }
    return {
      hablar,
      procesar,
      repetir() { if (ultima) hablar(ultima.frase, ultima.perfil, ultima.asiento ?? 1); },
      reiniciar() { ultimoId = null; },
      callar() { if (hayVoz()) try { window.speechSynthesis.cancel(); } catch (e) {} },
    };
  };
})();
