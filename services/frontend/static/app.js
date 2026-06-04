function dashboard() {
    return {
        // ── State ──
        view: 'overview',
        patients: [],
        stats: {},
        selectedId: null,
        wsConnected: false,
        wsError: false,

        // ── Init ──
        init() {
            this.connectWS();
        },

        // ── WebSocket ──
        connectWS() {
            const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${proto}//${window.location.host}/api/ws`;
            let ws = new WebSocket(wsUrl);
            let self = this;

            ws.onopen = () => {
                self.wsConnected = true;
                self.wsError = false;
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'update') {
                        self.patients = data.patients || [];
                        self.stats = data.stats || {};
                    }
                } catch (e) {
                    // ignore parse errors
                }
            };

            ws.onclose = () => {
                self.wsConnected = false;
                self.wsError = true;
                setTimeout(() => self.connectWS(), 3000);
            };

            ws.onerror = () => {
                self.wsError = true;
                ws.close();
            };
        },

        // ── Navigation ──
        selectPatient(id) {
            this.selectedId = id;
            this.view = 'detail';
        },

        goBack() {
            this.selectedId = null;
            this.view = 'overview';
        },

        // ── Getters ──
        get selectedPatient() {
            if (!this.selectedId) return null;
            return this.patients.find(p => p.paziente_id === this.selectedId) || null;
        },

        get sortedPatients() {
            const order = { alto: 0, medio: 1, basso: 2 };
            return [...this.patients].sort((a, b) => {
                return (order[a.livello_rischio] || 9) - (order[b.livello_rischio] || 9);
            });
        },

        // ── Formatting ──
        round(v, d = 0) {
            if (v == null) return '—';
            return Number(v).toFixed(d);
        },

        riskBadgeClass(risk) {
            return risk || 'basso';
        },

        actionClass(action) {
            return (action || 'monitoring').replace(/ /g, '_');
        },

        qBarWidth(val) {
            const normalized = Math.max(0, Math.min(100, ((val + 20) / 40) * 100));
            return normalized + '%';
        },

        qBarClass(val, isMax) {
            if (isMax) return 'pos max';
            return val >= 0 ? 'pos' : 'neg';
        },

        isMaxQ(pid, action) {
            const p = this.patients.find(p2 => p2.paziente_id === pid);
            if (!p || !p.q_values) return false;
            const vals = Object.values(p.q_values);
            const maxVal = Math.max(...vals);
            return p.q_values[action] === maxVal;
        },

        vitalClass(val, min, max) {
            if (val == null) return '';
            return (val >= min && val <= max) ? 'in-range' : 'out-range';
        },

        vitalBarWidth(val, absMin, absMax) {
            const pct = ((val - absMin) / (absMax - absMin)) * 100;
            return Math.max(0, Math.min(100, pct)) + '%';
        },

        vitalBarClass(val, min, max) {
            return (val >= min && val <= max) ? 'in-range' : 'out-range';
        },

        actionBadgeHTML(azione) {
            const labels = {
                monitoring: '🔍 Monitoraggio',
                contatta_medico: '📞 Contatta Medico',
                pronto_soccorso: '🚑 Pronto Soccorso',
                emergenza: '🚨 Emergenza'
            };
            return labels[azione] || azione;
        },

        formatTimestamp() {
            return new Date().toLocaleTimeString();
        },

        get patientDetail() {
            return this.selectedPatient;
        },

        get history() {
            if (!this.selectedId) return [];
            return this.selectedPatient && this.selectedPatient.storico
                ? this.selectedPatient.storico
                : [];
        }
    };
}
