"""Execute the evolution page summaries against the committed measurements."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from servidor_truco import _comparacion_publicada

ROOT = Path(__file__).resolve().parent.parent

NODE_RUNNER = r"""
const fs = require('fs'), vm = require('vm');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
class Element {
  constructor() { this.children = []; this.textContent = ''; this.classList = {add(){}}; }
  append(...children) { this.children.push(...children); }
  setAttribute() {}
  addEventListener() {}
  get text() { return this.textContent + this.children.map(c => c instanceof Element ? c.text : String(c)).join(' '); }
}
const context = {location:{search:''}, URLSearchParams, window:{innerWidth:1000},
 document:{createElement:()=>new Element(), createElementNS:()=>new Element(), createDocumentFragment:()=>new Element()},
 data};
vm.createContext(context);
vm.runInContext(data.source + `
state.comparison = data.comparison;
state.p1.gens = data.p1;
state.run = data.run;
state.p2.acceptance = data.acceptance;
const result = {headline:headline(data.phase).text, acceptance:acceptanceView().text, comparison:comparison().text};
JSON.stringify(result);`, context);
process.stdout.write(vm.runInContext('JSON.stringify(result)', context));
"""


@unittest.skipUnless(shutil.which('node'), 'Node.js is required to execute browser JavaScript')
class EvolutionJavaScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        page = (ROOT / 'proyecto/evolucion.html').read_text(encoding='utf8')
        cls.source = page.split('<script>')[1].split("document.querySelector('nav').addEventListener")[0]
        cls.comparison = _comparacion_publicada(ROOT)
        cls.p1 = [json.loads(line) for line in (ROOT / 'champions/phase1/generations.jsonl').read_text(encoding='utf8').splitlines()]

    def render_summaries(self, run='run3', phase='p2', acceptance=None):
        result = subprocess.run(['node', '-e', NODE_RUNNER], input=json.dumps({
            'source':self.source, 'comparison':self.comparison, 'p1':self.p1,
            'run':run, 'phase':phase, 'acceptance':acceptance or {}}),
            capture_output=True, text=True, encoding='utf8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_published_headline_uses_actual_exam_rates(self):
        rendered = self.render_summaries()
        exam = self.comparison['shipped']['exam']
        for opponent in ('adaptive_utility', 'reactive', 'phase1'):
            rate = exam['shipped'][exam['opponents'].index(opponent)]
            self.assertIn(f'{rate * 100:.1f}'.replace('.', ',') + ' %', rendered['headline'])
        self.assertIn('generación 48', rendered['headline'])

    def test_inconclusive_does_not_claim_no_improvement(self):
        # A positive estimate with an interval crossing the threshold is inconclusive.
        acceptance = {'decision':'inconclusive', 'direct':{'mean':.51,'lower':.48,'upper':.54},
                      'held_out':{'mean':.01,'lower':-.04,'upper':.06},
                      'config':{'candidate':{'generation':38}}}
        rendered = self.render_summaries(run='run2', acceptance=acceptance)
        for key in ('headline', 'acceptance'):
            self.assertIn('no se demostró una mejora', rendered[key])
            self.assertNotIn('no mejoró', rendered[key])

    def test_comparison_uses_latest_measured_generations(self):
        rendered = self.render_summaries(phase='p1')
        self.assertIn('generación 48', rendered['headline'])
        p1 = max((r for r in self.comparison['phase1_reference'] if r['reference_opponent']=='adaptive_utility'), key=lambda r:r['generation'])
        p2 = max((r for r in self.comparison['phase2_reference'] if r['reference_opponent']=='adaptive_utility' and r['adaptation']=='on'), key=lambda r:r['generation'])
        text = rendered['comparison']
        for row in (p1,p2):
            self.assertIn(f"generación {row['generation']}", text)
            self.assertIn(f"{row['win_rate'] * 100:.1f}".replace('.', ',') + ' %', text)
        difference = (p2['win_rate'] - p1['win_rate']) * 100
        self.assertIn(('+' if difference > 0 else '−') + f'{abs(difference):.1f}'.replace('.', ',') + ' pts', text)
