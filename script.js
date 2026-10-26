const $ = (s) => document.querySelector(s);
const methodSelect = $("#methodSelect");
const fieldsEl = $("#fields");
const descEl = $("#methodDescription");
const form = $("#calculatorForm");
const resultSummary = $("#resultSummary");
const resultStats = $("#resultStats");
const resultBadge = $("#resultBadge");
const tableWrap = $("#tableWrap");
const downloadCsv = $("#downloadCsv");
const chartHint = $("#chartHint");
let chart = null;
let lastRows = [];
let lastResult = null;

const methods = {
  bisection: {name:"Bisection Method", group:"Root Finding", desc:"Repeatedly halves an interval where f(a) and f(b) have opposite signs.", fields:[
    ["function","Function f(x)","x^3 - 4*x - 9","Example: x^3 - 4*x - 9"],
    ["a","Left endpoint a","2"],["b","Right endpoint b","3"],["tol","Tolerance","0.0001"],["maxIter","Maximum iterations","100"]]},
  falsePosition:{name:"False Position Method",group:"Root Finding",desc:"Uses a straight line between the interval endpoints to estimate the root.",fields:[
    ["function","Function f(x)","x^3 - 4*x - 9"],["a","Left endpoint a","2"],["b","Right endpoint b","3"],["tol","Tolerance","0.0001"],["maxIter","Maximum iterations","100"]]},
  newton:{name:"Newton–Raphson Method",group:"Root Finding",desc:"Uses the derivative of f(x) to refine a single initial guess.",fields:[
    ["function","Function f(x)","x^3 - 4*x - 9"],["x0","Initial guess x₀","2.5"],["tol","Tolerance","0.0001"],["maxIter","Maximum iterations","50"]]},
  secant:{name:"Secant Method",group:"Root Finding",desc:"Approximates the derivative using two recent estimates; no symbolic derivative is required.",fields:[
    ["function","Function f(x)","x^3 - 4*x - 9"],["x0","First guess x₀","2"],["x1","Second guess x₁","3"],["tol","Tolerance","0.0001"],["maxIter","Maximum iterations","100"]]},
  gaussSeidel:{name:"Gauss–Seidel Method",group:"Linear Systems",desc:"Solves A·x = b by updating each unknown immediately during each iteration.",fields:[
    ["matrix","Coefficient matrix A","4,1,1\n2,5,1\n1,1,3","Rows on separate lines; values separated by commas or spaces"],
    ["vector","Right-hand side b","6, -4, 7","Comma- or space-separated values"],["guess","Initial guess (optional)","0, 0, 0"],["tol","Tolerance","0.0001"],["maxIter","Maximum iterations","100"]]},
  jacobi:{name:"Jacobi Method",group:"Linear Systems",desc:"Solves A·x = b by computing all new unknowns from the previous iteration.",fields:[
    ["matrix","Coefficient matrix A","4,1,1\n2,5,1\n1,1,3","Rows on separate lines; values separated by commas or spaces"],
    ["vector","Right-hand side b","6, -4, 7","Comma- or space-separated values"],["guess","Initial guess (optional)","0, 0, 0"],["tol","Tolerance","0.0001"],["maxIter","Maximum iterations","100"]]},
  newtonForward:{name:"Newton Forward Interpolation",group:"Interpolation",desc:"Uses forward differences; best suited to equally spaced x-values near the beginning of the data.",fields:[
    ["xs","x values","0, 1, 2, 3","Comma-separated"],["ys","y values","1, 2, 9, 28","Must have the same number of values"],["target","Target x","1.5"]]},
  newtonBackward:{name:"Newton Backward Interpolation",group:"Interpolation",desc:"Uses backward differences; best suited to equally spaced x-values near the end of the data.",fields:[
    ["xs","x values","0, 1, 2, 3","Comma-separated"],["ys","y values","1, 2, 9, 28","Must have the same number of values"],["target","Target x","2.5"]]},
  stirling:{name:"Stirling Interpolation",group:"Interpolation",desc:"A central-difference interpolation method for equally spaced data with the target near the center.",fields:[
    ["xs","x values","0, 1, 2, 3, 4","Comma-separated"],["ys","y values","1, 2, 9, 28, 65","Must have the same number of values"],["target","Target x","2.2"]]},
  lagrange:{name:"Lagrange Interpolation",group:"Interpolation",desc:"Builds an interpolation polynomial through the provided data points; x-values need not be equally spaced.",fields:[
    ["xs","x values","0, 1, 2, 3","Comma-separated"],["ys","y values","1, 2, 9, 28","Must have the same number of values"],["target","Target x","1.5"]]}
};

function renderFields() {
  const m = methods[methodSelect.value];
  descEl.innerHTML = `<strong>${m.group}</strong><br>${m.desc}`;
  fieldsEl.innerHTML = m.fields.map(([key,label,value,hint]) => {
    const isArea = ["matrix"].includes(key);
    const type = ["tol"].includes(key) ? "number" : "text";
    const step = key === "tol" ? ' step="any" min="0"' : "";
    return `<div class="field-group"><label for="field-${key}">${label}${hint ? `<span class="hint">· ${hint}</span>` : ""}</label>${isArea
      ? `<textarea id="field-${key}" name="${key}" spellcheck="false">${value}</textarea>`
      : `<input id="field-${key}" name="${key}" type="${type}" value="${value}"${step} ${["maxIter"].includes(key) ? 'min="1"' : ""} required>`}</div>`;
  }).join("");
}
methodSelect.addEventListener("change", renderFields);
renderFields();

function getValues() {
  const values = {};
  methods[methodSelect.value].fields.forEach(([key]) => {
    const el = form.elements.namedItem(key);
    values[key] = el ? el.value.trim() : "";
  });
  return values;
}
function num(v, name) {
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`${name} must be a valid number.`);
  return n;
}
function positive(v, name) {
  const n = num(v, name);
  if (n <= 0) throw new Error(`${name} must be greater than zero.`);
  return n;
}
function iterations(v) {
  const n = Math.floor(num(v, "Maximum iterations"));
  if (n < 1 || n > 10000) throw new Error("Maximum iterations must be between 1 and 10000.");
  return n;
}
function parseList(s, label) {
  const arr = s.trim().split(/[,\s;]+/).filter(Boolean).map(Number);
  if (!arr.length || arr.some(n => !Number.isFinite(n))) throw new Error(`${label} must contain valid numbers.`);
  return arr;
}
function parseMatrix(s) {
  const rows = s.trim().split(/\n|;/).map(r => r.trim()).filter(Boolean).map(r => r.split(/[,\s]+/).filter(Boolean).map(Number));
  if (!rows.length || rows.some(r => r.some(n => !Number.isFinite(n)))) throw new Error("Matrix A contains invalid numbers.");
  const n = rows.length;
  if (rows.some(r => r.length !== n)) throw new Error("Matrix A must be square (same number of rows and columns).");
  return rows;
}
function makeFunction(expr) {
  if (!expr) throw new Error("Please enter a function f(x).");
  let compiled;
  try { compiled = math.compile(expr); compiled.evaluate({x: 0}); }
  catch (e) { throw new Error("Could not read f(x). Check the function syntax, e.g. x^3 - 4*x - 9."); }
  return x => {
    const y = Number(compiled.evaluate({x}));
    if (!Number.isFinite(y)) throw new Error(`f(x) is not finite at x = ${format(x)}.`);
    return y;
  };
}
function format(n) {
  if (typeof n !== "number" || !Number.isFinite(n)) return String(n);
  if (Math.abs(n) < 1e-10 && n !== 0) return n.toExponential(5);
  return Number(n.toPrecision(10)).toString();
}
function makeRow(obj) { return obj; }
function solveRoot(method, v) {
  const f = makeFunction(v.function);
  const tol = positive(v.tol, "Tolerance");
  const max = iterations(v.maxIter);
  const rows = [];
  let root, converged = false;
  if (method === "bisection" || method === "falsePosition") {
    let a = num(v.a, "a"), b = num(v.b, "b"), fa = f(a), fb = f(b);
    if (a === b) throw new Error("Endpoints a and b must be different.");
    if (fa * fb > 0) throw new Error("f(a) and f(b) must have opposite signs. Try another interval.");
    for (let i=1;i<=max;i++) {
      let c = method === "bisection" ? (a+b)/2 : (a*fb-b*fa)/(fb-fa);
      const fc = f(c);
      rows.push({Iteration:i,a:format(a),b:format(b), "f(a)":format(fa), "f(b)":format(fb), x:format(c), "f(x)":format(fc), "Error": i===1 ? "—" : format(Math.abs(c-root))});
      const err = i===1 ? Infinity : Math.abs(c-root);
      root = c;
      if (Math.abs(fc) <= tol || err <= tol) { converged = true; break; }
      if (fa * fc < 0) { b=c; fb=fc; } else { a=c; fa=fc; }
    }
  } else if (method === "newton") {
    let x = num(v.x0, "Initial guess");
    for (let i=1;i<=max;i++) {
      const fx = f(x);
      const h = 1e-6 * Math.max(1,Math.abs(x));
      const d = (f(x+h)-f(x-h))/(2*h);
      if (!Number.isFinite(d) || Math.abs(d)<1e-14) throw new Error("Derivative is zero or too small near the current guess. Try another initial guess.");
      const next = x - fx/d;
      const err = Math.abs(next-x);
      rows.push({Iteration:i,x:format(x),"f(x)":format(fx),"f′(x) ≈":format(d),"Next x":format(next),Error:format(err)});
      x = next; root = x;
      if (Math.abs(f(x)) <= tol || err <= tol) { converged = true; break; }
    }
  } else {
    let x0 = num(v.x0, "First guess"), x1 = num(v.x1, "Second guess");
    for (let i=1;i<=max;i++) {
      const f0=f(x0), f1=f(x1), den=f1-f0;
      if (Math.abs(den)<1e-14) throw new Error("The function values at the two guesses are too close. Try different guesses.");
      const x2=x1-f1*(x1-x0)/den, err=Math.abs(x2-x1);
     
rows.push({Iteration:i,"x₀":format(x0),"x₁":format(x1),"f(x₀)":format(f0),"f(x₁)":format(f1),"Next x":format(x2),Error:format(err)});

      x0=x1; x1=x2; root=x2;
      if (Math.abs(f(root))<=tol || err<=tol) {converged=true;break;}
    }
  }
  return {answer:root, rows, converged, detail:`f(root) = ${format(f(root))}`, graph:{type:"function",f,root}};
}
function solveLinear(method,v) {
  const A=parseMatrix(v.matrix), b=parseList(v.vector,"Vector b");
  const n=A.length;
  if (b.length!==n) throw new Error("Vector b must contain one value for each row of A.");
  if (A.some(row=>row.some(x=>!Number.isFinite(x)))) throw new Error("Matrix A contains invalid values.");
  if (A.some((row,i)=>Math.abs(row[i])<1e-14)) throw new Error("A diagonal coefficient is zero. Rearrange the rows so the diagonal has no zeros.");
  const tol=positive(v.tol,"Tolerance"), max=iterations(v.maxIter);
  let x=v.guess ? parseList(v.guess,"Initial guess") : Array(n).fill(0);
  if (x.length!==n) throw new Error("Initial guess must have the same number of values as the number of unknowns.");
  const rows=[]; let converged=false;
  for(let k=1;k<=max;k++){
    const old=x.slice(), next=x.slice();
    for(let i=0;i<n;i++){
      let sum=0;
      for(let j=0;j<n;j++) if(j!==i) sum += A[i][j]*(method==="gaussSeidel" ? next[j] : old[j]);
      next[i]=(b[i]-sum)/A[i][i];
    }
    const err=Math.max(...next.map((val,i)=>Math.abs(val-old[i])));
    x=next;
    const row={Iteration:k}; x.forEach((val,i)=>row[`x${i+1}`]=format(val)); row["Max error"]=format(err); rows.push(row);
    if(err<=tol){converged=true;break;}
  }
  const residual=Math.max(...A.map((row,i)=>Math.abs(row.reduce((s,a,j)=>s+a*x[j],0)-b[i])));
  return {answer:x.map((val,i)=>`x${i+1} = ${format(val)}`).join("   ·   "),rows,converged,detail:`Maximum residual |Ax − b| = ${format(residual)}`,stats:[["Unknowns",n],["Iterations",rows.length],["Max residual",format(residual)]]};
}
function differenceTable(ys) {
  const table=[ys.slice()];
  while(table[table.length-1].length>1) {
    const prev=table[table.length-1], next=[];
    for(let i=0;i<prev.length-1;i++) next.push(prev[i+1]-prev[i]);
    table.push(next);
  }
  return table;
}
function lagrangeValue(xs,ys,x) {
  let total=0;
  for(let i=0;i<xs.length;i++){
    let term=ys[i];
    for(let j=0;j<xs.length;j++) if(i!==j){
      const den=xs[i]-xs[j];
      if(Math.abs(den)<1e-14) throw new Error("x-values must be distinct.");
      term *= (x-xs[j])/den;
    }
    total+=term;
  }
  return total;
}
function checkInterpolation(v, needsEqual) {
  const xs=parseList(v.xs,"x values"), ys=parseList(v.ys,"y values"), target=num(v.target,"Target x");
  if(xs.length!==ys.length) throw new Error("x values and y values must have the same number of values.");
  if(xs.length<2) throw new Error("Enter at least two data points.");
  if(new Set(xs).size!==xs.length) throw new Error("x-values must be distinct.");
  if(needsEqual){
    const h=xs[1]-xs[0];
    if(Math.abs(h)<1e-14 || xs.some((x,i)=>i>0 && Math.abs((x-xs[i-1])-h)>1e-7*Math.max(1,Math.abs(h)))) throw new Error("This method requires equally spaced x-values.");
  }
  return {xs,ys,target};
}
function solveInterpolation(method,v) {
  const needEqual=method!=="lagrange";
  const {xs,ys,target}=checkInterpolation(v,needEqual);
  let answer, rows=[], detail="";
  const diffs=differenceTable(ys);
  if(method==="lagrange"){
    answer=lagrangeValue(xs,ys,target);
    rows=xs.map((x,i)=>({Point:i+1,x:format(x),y:format(ys[i]),"Contribution":format(ys[i]*xs.reduce((p,xj,j)=>j===i?p:p*((target-xj)/(x-xs[j])),1))}));
    detail=`Interpolated from ${xs.length} data points using the Lagrange polynomial.`;
  } else {
    const h=xs[1]-xs[0], n=xs.length;
    const u=(target-xs[0])/h;
    if(method==="newtonForward"){
      let term=1, sum=ys[0];
      rows.push({Order:0,"Difference":format(diffs[0][0]),"Term contribution":format(sum),"Running total":format(sum)});
      for(let k=1;k<n;k++){
        term *= (u-(k-1))/k;
        const contribution=term*diffs[k][0]; sum+=contribution;
        rows.push({Order:k,"Difference":format(diffs[k][0]),"Term contribution":format(contribution),"Running total":format(sum)});
      }
      answer=sum; detail=`u = (x − x₀) / h = ${format(u)}; h = ${format(h)}.`;
    } else if(method==="newtonBackward"){
      const ub=(target-xs[n-1])/h; let term=1, sum=ys[n-1];
      rows.push({Order:0,"Difference":format(diffs[0][n-1]),"Term contribution":format(sum),"Running total":format(sum)});
      for(let k=1;k<n;k++){
        term *= (ub+(k-1))/k;
        const contribution=term*diffs[k][diffs[k].length-1]; sum+=contribution;
        rows.push({Order:k,"Difference":format(diffs[k][diffs[k].length-1]),"Term contribution":format(contribution),"Running total":format(sum)});
      }
      answer=sum; detail=`u = (x − xₙ) / h = ${format(ub)}; h = ${format(h)}.`;
    } else {
      // Stirling central interpolation is evaluated through the equivalent polynomial
      // to ensure a stable result for any number of equally spaced data points.
      answer=lagrangeValue(xs,ys,target);
      rows=diffs.map((arr,k)=>({DifferenceOrder:k,Values:arr.map(format).join(", ")}));
      detail=`Central-difference data table shown below; the polynomial is evaluated at x = ${format(target)}.`;
    }
  }
  const sorted=xs.map((x,i)=>({x,y:ys[i]})).sort((a,b)=>a.x-b.x);
  return {answer:format(answer),rows,converged:true,detail,stats:[["Data points",xs.length],["Target x",format(target)],["Interpolated y",format(answer)]],graph:{type:"interpolation",xs,ys,target,answer}};
}
function calculate(method,v) {
  if(["bisection","falsePosition","newton","secant"].includes(method)) return solveRoot(method,v);
  if(["gaussSeidel","jacobi"].includes(method)) return solveLinear(method,v);
  return solveInterpolation(method,v);
}
function renderTable(rows) {
  if(!rows || !rows.length){tableWrap.innerHTML='<div class="table-empty">No iteration rows to show.</div>';return;}
  const cols=[...new Set(rows.flatMap(r=>Object.keys(r)))];
  tableWrap.innerHTML=`<table class="data-table"><thead><tr>${cols.map(c=>`<th>${escapeHtml(c)}</th>`).join("")}</tr></thead><tbody>${rows.map(row=>`<tr>${cols.map(c=>`<td>${escapeHtml(row[c]===undefined?"—":String(row[c]))}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function showResult(result, method) {
  lastResult=result; lastRows=result.rows||[];
  resultBadge.textContent=result.converged ? "Completed" : "Iteration limit reached";
  resultBadge.className="result-badge "+(result.converged?"success":"");
  resultSummary.className="result-summary";
  resultSummary.innerHTML=`<div class="answer-label">${methods[method].name} · Approximate answer</div><div class="answer-value">${escapeHtml(result.answer)}</div><div class="answer-detail">${escapeHtml(result.detail||"")}</div>`;
  const stats=result.stats || [["Iterations",result.rows.length],["Tolerance",form.elements.namedItem("tol")?.value || "—"],["Status",result.converged?"Converged":"Not converged"]];
  resultStats.classList.remove("hidden");
  resultStats.innerHTML=stats.map(([label,val])=>`<div class="stat-card"><small>${escapeHtml(label)}</small><b>${escapeHtml(val)}</b></div>`).join("");
  renderTable(result.rows);
  downloadCsv.disabled=!result.rows?.length;
  drawChart(result.graph);
}
function showError(err) {
  resultBadge.textContent="Check input"; resultBadge.className="result-badge error";
  resultSummary.className="result-summary";
  resultSummary.innerHTML=`<div class="answer-label">Unable to calculate</div><div class="answer-value" style="color:var(--red);font-size:18px">Input error</div><div class="answer-detail">${escapeHtml(err.message||String(err))}</div>`;
  resultStats.classList.add("hidden"); tableWrap.innerHTML='<div class="table-empty">Fix the input above and calculate again.</div>';
  downloadCsv.disabled=true; lastRows=[]; lastResult=null;
}
function drawChart(graph) {
  if(chart){chart.destroy();chart=null;}
  const canvas=$("#resultChart");
  if(!graph){chartHint.textContent="This method has no graph to display.";return;}
  let datasets=[], labels=[], type="line";
  if(graph.type==="function"){
    const span=Math.max(5,Math.abs(graph.root)*1.8);
    let lo=graph.root-span/2, hi=graph.root+span/2;
    if(!Number.isFinite(lo)||!Number.isFinite(hi)||lo===hi){lo=-10;hi=10;}
    const pts=[];
    for(let i=0;i<=120;i++){const x=lo+(hi-lo)*i/120;try{const y=graph.f(x);if(Math.abs(y)<1e7)pts.push({x,y});}catch(e){}}
    datasets=[{label:"f(x)",data:pts,borderColor:"#5a9bff",backgroundColor:"#5a9bff22",borderWidth:2,pointRadius:0,tension:.1},{label:"Approximate root",data:[{x:graph.root,y:graph.f(graph.root)}],borderColor:"#54d6a0",backgroundColor:"#54d6a0",pointRadius:5,showLine:false}];
    chartHint.textContent="Function curve with the computed root marked in green.";
  } else if(graph.type==="interpolation"){
    const min=Math.min(...graph.xs),max=Math.max(...graph.xs), pad=(max-min)*.08||1;
    const pts=[]; for(let i=0;i<=100;i++){const x=min-pad+(max-min+2*pad)*i/100;pts.push({x,y:lagrangeValue(graph.xs,graph.ys,x)});}
    datasets=[{label:"Interpolation polynomial",data:pts,borderColor:"#5a9bff",borderWidth:2,pointRadius:0},{label:"Given data",data:graph.xs.map((x,i)=>({x,y:graph.ys[i]})),borderColor:"#f5a5c8",backgroundColor:"#f5a5c8",pointRadius:4,showLine:false},{label:"Target value",data:[{x:graph.target,y:Number(graph.answer)}],borderColor:"#54d6a0",backgroundColor:"#54d6a0",pointRadius:5,showLine:false}];
    chartHint.textContent="Interpolation curve, original data points, and the requested target value.";
  }
  chart=new Chart(canvas,{type,data:{datasets},options:{responsive:true,maintainAspectRatio:false,parsing:false,interaction:{mode:"nearest",intersect:false},plugins:{legend:{labels:{color:"#a9bad2",boxWidth:10,font:{size:10}}},tooltip:{callbacks:{label:(ctx)=>`${ctx.dataset.label}: (${format(ctx.parsed.x)}, ${format(ctx.parsed.y)})`}}},scales:{x:{type:"linear",grid:{color:"#203149"},ticks:{color:"#8498b6",font:{size:9}},title:{display:true,text:"x",color:"#8498b6"}},y:{grid:{color:"#203149"},ticks:{color:"#8498b6",font:{size:9}},title:{display:true,text:"y",color:"#8498b6"}}}}});
}
form.addEventListener("submit",e=>{
  e.preventDefault();
  try{const result=calculate(methodSelect.value,getValues());showResult(result,methodSelect.value);}
  catch(err){showError(err);}
});
$("#resetBtn").addEventListener("click",()=>{renderFields();resultBadge.textContent="Waiting for input";resultBadge.className="result-badge";resultSummary.className="result-summary empty-state";resultSummary.innerHTML='<div class="empty-icon">ƒ(x)</div><h3>Your result will appear here</h3><p>Choose a method, enter your values, then click <b>Calculate</b>.</p>';resultStats.classList.add("hidden");tableWrap.innerHTML='<div class="table-empty">Calculation steps will be listed here.</div>';downloadCsv.disabled=true;lastRows=[];lastResult=null;if(chart){chart.destroy();chart=null;}chartHint.textContent="A graph will appear when the selected method supports visualization.";});
downloadCsv.addEventListener("click",()=>{
  if(!lastRows.length)return;
  const cols=[...new Set(lastRows.flatMap(r=>Object.keys(r)))];
  const csv=[cols,...lastRows.map(r=>cols.map(c=>r[c]===undefined?"":r[c]))].map(row=>row.map(v=>`"${String(v).replace(/"/g,'""')}"`).join(",")).join("\n");
  const blob=new Blob(["\ufeff"+csv],{type:"text/csv;charset=utf-8;"});
  const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=`${methodSelect.value}-results.csv`;a.click();URL.revokeObjectURL(url);
});
document.querySelectorAll(".nav-item").forEach(btn=>btn.addEventListener("click",()=>{
  document.querySelectorAll(".nav-item").forEach(b=>b.classList.toggle("active",b===btn));
  const target=btn.dataset.target;$("#calculator").classList.toggle("hidden",target!=="calculator");$("#guide").classList.toggle("hidden",target!=="guide");
}));
$("#backToCalculator").addEventListener("click",()=>{document.querySelector('[data-target="calculator"]').click();});
