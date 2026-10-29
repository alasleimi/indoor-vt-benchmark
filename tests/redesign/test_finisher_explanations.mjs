import test from "node:test";
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

// Exercise the actual browser module and KaTeX, without a browser or build step.
const web = new URL("../../src/indoor_vt/execution/web/", import.meta.url);
const source = (await readFile(new URL("finisher-explanations.js", web), "utf8"))
  .replace('"/vendor/katex/katex.mjs"', JSON.stringify(new URL("vendor/katex/katex.mjs", web).href));
const explanation = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const app = await readFile(new URL("app.js", web), "utf8");
const parameters = {neighbors:5, half_window:2, alpha:.7, tolerance_m:.1,
  support_fraction:.04, duplicate_m:.06, scope_radius_m:.75, gap_m:.8,
  group_cost:1, trials:30000};
const method = (family="mpurge", finisher="robust") => ({family, finisher,
  component:"sources", group:"vt", selection:"ospa", source_full_name:"MPUrge + robust range-refinement and peak-consumption finisher",
  code:{"Composition constructors":[{file:"execution/compositions.py",line:44,label:"composed_reconstructor()"}]}});
const receipt = (unify=true) => ({parameters:{...parameters}, frozen:{}, unify,
  id:"saved-receipt", run:"saved-run"});
const steps = html => [...html.matchAll(/<li[^>]+data-step="([^"]+)"/g)].map(x=>x[1]);
const functionSource = name => {
  const start=app.indexOf(`function ${name}(`), end=app.indexOf("\nfunction ",start+1);
  assert.ok(start>=0);
  return app.slice(start,end<0?undefined:end);
};
function renderTab(name,currentMethod,activeReceipt) {
  const local={...explanation,currentMethod,activeReceipt,
    currentRow:{receipt_id:activeReceipt.id},data:{config:{name:"test",storage:{cache_mib:64,ram_mib:16}}},
    esc:s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]),
    selectionExplanation:()=>"Saved selection protocol.",obstacleEvaluationNote:()=>"",
    parameterTable:(_p,title)=>`<h3>${title}</h3>`,panel:{innerHTML:""}};
  return new Function(...Object.keys(local), `${functionSource("codeLinks")}\n${functionSource(name)}\n${name}(panel); return panel.innerHTML;`)(...Object.values(local));
}

test("robust reconstruction explains the full executable order and exact weighted loss",()=>{
  const html=explanation.reconstructionExplanation(method(),receipt());
  assert.deepEqual(steps(html),["neighborhoods","unification","mpurge-dissimilarity",
    "reciprocal-matching","peak-tracks","proposal-least-squares","finisher-association",
    "finisher-ranking","finisher-optimization","finisher-acceptance","finisher-consumption"]);
  assert.match(html,/including rejected crossings/);
  assert.match(html,/Do not recompute components afterward/);
  assert.match(html,/Hold fixed throughout this solve/);
  assert.ok(html.includes(String.raw`\frac{w_i\bigl(\|z-x_i\|_2-\bar y_i\bigr)^2}{\tau^2}`));
  assert.match(html,/stop the entire finisher/);
  assert.doesNotMatch(html,/katex-error/);
});

test("support-first has its own ranking and no coordinate-refinement step",()=>{
  const html=explanation.reconstructionExplanation(method("mpurge","consumption"),receipt());
  assert.ok(html.includes(String.raw`S(q)=n(q)+\frac{g(q)}{N+1}`));
  assert.ok(!steps(html).includes("finisher-optimization"));
  assert.doesNotMatch(html,/soft-L1|median_clip_factor|smooth_weight/);
  assert.match(html,/does not optimize coordinates/);
});

test("unification DP appears only when active; proposer branches stay distinct",()=>{
  const off=explanation.reconstructionExplanation(method(),receipt(false));
  assert.match(off,/No subset is selected/);
  assert.doesNotMatch(off,/D_\{ij\}|dynamic program/);
  for(const family of ["zayets","ransac"]){
    const html=explanation.reconstructionExplanation(method(family),receipt());
    assert.ok(steps(html).some(key=>key.startsWith(family+"-")));
    assert.ok(!steps(html).includes("mpurge-dissimilarity"));
    assert.ok(steps(html).includes("finisher-optimization"));
  }
});

test("Scientific and Code tabs show the same algorithm before separate caching prose",()=>{
  const m=method(),r=receipt();
  const scientific=renderTab("renderMethod",m,r),code=renderTab("renderCode",m,r);
  const algorithm=explanation.reconstructionExplanation(m,r);
  assert.ok(scientific.startsWith(algorithm));
  assert.ok(code.startsWith(algorithm));
  assert.ok(code.indexOf("Execution &amp; caching")>code.indexOf("finisher-consumption"));
  assert.doesNotMatch(scientific,/uses its pairwise peak matcher and connected local groups/);
  assert.doesNotMatch(code,/&amp;#(?:x[0-9a-f]+|\d+);/i);
  assert.match(code,/source\?file=execution%2Fcompositions.py#L44/);
});

test("current values and saved constructor defaults appear once in ordered parameter table",()=>{
  const r=receipt();r.parameters.smooth_weight=.6;
  const html=renderTab("renderParameters",method(),r);
  assert.equal((html.match(/<table/g)||[]).length,1);
  assert.match(html,/<td>0.6 <\/td>/);
  assert.match(html,/fixed constructor default in this saved run/);
  assert.ok(html.indexOf("neighbors")<html.indexOf("smooth_weight"));
  assert.ok(html.indexOf("smooth_weight")<html.indexOf("duplicate_m"));
});

test("pipeline explanations retain MAP and fixed-source SMART-LEA after reconstruction",()=>{
  const m={...method(),component:"localization",group:"pipeline",selection:"stagewise"};
  const r={...receipt(),parameters:{alpha:.7,half_window:2,iterations:4},
    frozen:{source:parameters,map:{alpha:.7,half_window:1,neighbors:3,power:2}}};
  const html=explanation.reconstructionExplanation(m,r);
  assert.deepEqual(steps(html).slice(-2),["map-localization","lea-localization"]);
  assert.match(html,/original MAP coordinate/);
  assert.match(html,/Sources stay fixed throughout localization/);
  assert.doesNotMatch(html,/undefined|katex-error/);
});

test("connected-component conflict pruning is visible in both tabs for every MPUrge composition",()=>{
  const grouping={file:"science/grouping.py",line:113,label:"ConnectedComponentsGrouper()"};
  for(const finishing of [null,"robust","consumption"]){
    for(const component of ["sources","localization"]){
      for(const triangle of (finishing?[false]:[false,true])){
        const m={...method("mpurge",finishing),component,triangle,
          group:component==="sources"?"vt":"pipeline"};
        m.code["Algorithm steps"]=[grouping];
        for(const tab of ["renderMethod","renderCode"]){
          const html=renderTab(tab,m,receipt());
          assert.equal((html.match(/data-grouping="connected-components"/g)||[]).length,1);
          assert.match(html,/<ol class="component-pruning-rule"><li><strong>Highest degree\.<\/strong><\/li><li><strong>Lowest mean incident-edge dissimilarity\.<\/strong><\/li><li><strong>Lowest index<\/strong>/);
          assert.match(html,/only ties proceed to the next rule/);
          assert.match(html,/form connected components first/);
          assert.match(html,/all accepted matching edges incident to it in this scope, before pruning/);
          assert.match(html,/Do not recompute components afterward/);
          assert.ok(html.includes(String.raw`\frac{\sum_{e\in E(v)}\Delta_e}{d(v)}`));
          assert.doesNotMatch(html,/katex-error/);
        }
      }
    }
  }
});

test("the connected-component rule is absent from other groupers and MAP-only stages",()=>{
  const groupLink=name=>({file:"science/grouping.py",line:113,label:name+"()"});
  for(const family of ["ransac","zayets"]){
    const html=explanation.reconstructionExplanation(method(family),receipt());
    assert.doesNotMatch(html,/component-pruning-rule/);
  }
  for(const m of [
    {...method("mpurge",null),component:"initial",code:{"Algorithm steps":[groupLink("ConnectedComponentsGrouper")]}},
    {...method("mpurge",null),component:"localization",code:{"Composition constructors":[{file:"execution/presets.py",label:"mpurge_map()"}]}},
    {...method("mpurge",null),code:{"Algorithm steps":[groupLink("StarGrouper")]}}
  ]){
    assert.equal(explanation.usesConnectedComponentGrouping(m),false);
    assert.equal(explanation.connectedComponentSourceOverview(m),"");
  }
});

test("crossing-allowed descriptions remove the veto and define its limited scope",()=>{
  const m={...method(),allow_crossing:true};
  const html=explanation.reconstructionExplanation(m,receipt());
  assert.match(html,/Accept crossings/);
  assert.match(html,/reconstruction matching only/);
  assert.doesNotMatch(html,/Reject .*if it crosses|including rejected crossings|katex-error/);
  const solo={...m,finisher:null,code:{steps:[{file:'science/grouping.py',label:'ConnectedComponentsGrouper()'}]}};
  assert.match(explanation.connectedComponentSourceOverview(solo),/Crossings allowed/);
  const metrics=explanation.sourceMetricExplanation();
  assert.match(metrics,/separately within each AP/);
  assert.doesNotMatch(metrics,/undefined|katex-error/);
  assert.ok(metrics.includes(String.raw`\sum_a w_a O_a`));
});
