"""The published site's matcher must reproduce the scientific Python matcher."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from indoor_vt.execution.peak_explorer import pair_data
from indoor_vt.execution.viewer import json_value
from indoor_vt.science.matching import MPUrgeMatcher, CrossingAllowedMPUrgeMatcher
from indoor_vt.science.costs import absolute_difference


def test_browser_matcher_parity(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required for browser/Python parity')
    rng = np.random.default_rng(81771)
    fixtures = []
    for k in range(120):
        n, m = rng.integers(0, 27, size=2)
        a = {'ranges': np.round(rng.uniform(.1, 20, n), 2),
             'source_ids': np.arange(n), 'xyz': [1., 2., 1.]}
        b = {'ranges': np.round(rng.uniform(.1, 20, m), 2),
             'source_ids': np.arange(m), 'xyz': [2., 3., 1.5]}
        cls = CrossingAllowedMPUrgeMatcher if k % 3 == 0 else MPUrgeMatcher
        p = {'half_window': int(rng.choice([0, 1, 2, 7])), 'alpha': float(rng.choice([0., .25, .7, 1.])),
             'unify': bool(k % 2), 'triangle_tolerance_m': .2 if k % 4 == 0 else None,
             'matcher': cls.__name__}
        matcher = cls(p['half_window'], p['alpha'], size_unification_norm=absolute_difference if p['unify'] else None,
                      triangle_tolerance_m=p['triangle_tolerance_m'])
        fixtures.append({'a': a, 'b': b, 'p': p, 'expected': pair_data(a, b, matcher, [])})
    path = tmp_path/'cases.json'
    path.write_text(json.dumps(json_value(fixtures), allow_nan=False), encoding='utf-8')
    module = (Path(__file__).parents[2]/'src/indoor_vt/execution/web/static-matcher.mjs').as_uri()
    script = tmp_path/'check.mjs'
    script.write_text("import {readFileSync} from 'node:fs';\n" +
        f"import {{pairData}} from {json.dumps(module)};\n" + """
function compare(actual, expected, path) {
  if(typeof expected==='number') {
    if(Math.abs(actual-expected)>1e-10*Math.max(1,Math.abs(expected)))throw Error(path+': '+actual+' != '+expected);
  } else if(Array.isArray(expected)) {
    if(actual.length!==expected.length)throw Error(path+': length');
    expected.forEach((v,i)=>compare(actual[i],v,path+'['+i+']'));
  } else if(expected && typeof expected==='object') {
    for(const k of Object.keys(expected))compare(actual[k],expected[k],path+'.'+k);
  } else if(actual!==expected)throw Error(path+': literal');
}
for(const [i,c] of JSON.parse(readFileSync(process.argv[2],'utf8')).entries())
  compare(pairData(c.a,c.b,c.p,[]),c.expected,'case '+i);
""", encoding='utf-8')
    subprocess.run([node, str(script), str(path)], check=True, capture_output=True, text=True)
