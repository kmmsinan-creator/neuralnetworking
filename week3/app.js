// Globals
let trainData = null, testData = null;
let preprocessedTrainData = null, preprocessedTestData = null;
let model = null, validationData = null, validationLabels = null, validationPredictions = null, testPredictions = null;
let visorOpened = false;

// ==================== Load & Parse ====================
async function loadData() {
  const trainFile = document.getElementById('train-file').files[0];
  const testFile = document.getElementById('test-file').files[0];
  const status = document.getElementById('data-status');

  if (!trainFile || !testFile) {
    alert('Upload both train.csv and test.csv');
    return;
  }

  try {
    const trainText = await readFile(trainFile);
    const testText = await readFile(testFile);
    trainData = parseCSV(trainText);
    testData = parseCSV(testText);
    status.innerHTML = `✅ Data loaded: train=${trainData.length}, test=${testData.length}`;
    document.getElementById('inspect-btn').disabled = false;
  } catch (e) {
    status.innerHTML = `❌ Error: ${e.message}`;
  }
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = e => resolve(e.target.result);
    fr.onerror = () => reject(new Error('File read failed'));
    fr.readAsText(file);
  });
}

function parseCSV(text) {
  return Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true }).data;
}

// ==================== Inspect ====================
function inspectData() {
  const preview = document.getElementById('data-preview');
  preview.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
  preview.appendChild(makeTable(trainData.slice(0, 10)));

  makeVisuals();
  document.getElementById('preprocess-btn').disabled = false;
}

function makeTable(rows) {
  const t = document.createElement('table');
  const head = document.createElement('tr');
  Object.keys(rows[0]).forEach(k => { const th = document.createElement('th'); th.textContent = k; head.appendChild(th); });
  t.appendChild(head);
  rows.forEach(r => {
    const tr = document.createElement('tr');
    Object.values(r).forEach(v => { const td = document.createElement('td'); td.textContent = v ?? 'NULL'; tr.appendChild(td); });
    t.appendChild(tr);
  });
  return t;
}

function makeVisuals() {
  const bySex = {};
  trainData.forEach(r => {
    if (r.Sex) {
      if (!bySex[r.Sex]) bySex[r.Sex] = { s: 0, t: 0 };
      bySex[r.Sex].t++; if (r.Survived === 1) bySex[r.Sex].s++;
    }
  });
  const sexData = Object.entries(bySex).map(([k,v]) => ({ x:k, y:(v.s/v.t)*100 }));
  tfvis.render.barchart({ name: 'Survival Rate by Sex', tab: 'Charts' }, sexData, { xLabel:'Sex', yLabel:'%' });

  const byClass = {};
  trainData.forEach(r => {
    if (r.Pclass) {
      if (!byClass[r.Pclass]) byClass[r.Pclass] = { s:0,t:0 };
      byClass[r.Pclass].t++; if (r.Survived===1) byClass[r.Pclass].s++;
    }
  });
  const classData = Object.entries(byClass).map(([k,v]) => ({ x:`Class ${k}`, y:(v.s/v.t)*100 }));
  tfvis.render.barchart({ name: 'Survival Rate by Class', tab: 'Charts' }, classData, { xLabel:'Class', yLabel:'%' });

  if (!visorOpened) { tfvis.visor().open(); visorOpened = true; }
}

// ==================== Preprocess ====================
function preprocessData() {
  const out = document.getElementById('preprocessing-output');
  try {
    const ageMedian = median(trainData.map(r=>r.Age).filter(v=>v!=null));
    const fareMedian = median(trainData.map(r=>r.Fare).filter(v=>v!=null));
    const embarkedMode = mode(trainData.map(r=>r.Embarked).filter(v=>v));

    const feats = [], labels = [];
    trainData.forEach(r => {
      feats.push(extractFeatures(r, ageMedian, fareMedian, embarkedMode));
      labels.push(Number(r.Survived));
    });
    preprocessedTrainData = { features: tf.tensor2d(feats), labels: tf.tensor1d(labels,'float32') };

    const testFeats = [], ids = [];
    testData.forEach(r => { testFeats.push(extractFeatures(r, ageMedian, fareMedian, embarkedMode)); ids.push(r.PassengerId); });
    preprocessedTestData = { features:testFeats, ids };

    out.innerHTML = `✅ Preprocessing done<br/>Train shape: ${preprocessedTrainData.features.shape}`;
    document.getElementById('create-model-btn').disabled = false;
  } catch(e) {
    out.innerHTML = `❌ ${e.message}`;
  }
}

function extractFeatures(r, ageMedian, fareMedian, embarkedMode) {
  const age = r.Age ?? ageMedian;
  const fare = r.Fare ?? fareMedian;
  const emb = r.Embarked ?? embarkedMode;
  const feats = [
    (age-ageMedian)/ (std(trainData.map(x=>x.Age).filter(v=>v!=null))||1),
    (fare-fareMedian)/ (std(trainData.map(x=>x.Fare).filter(v=>v!=null))||1),
    r.SibSp||0,
    r.Parch||0
  ];
  feats.push(...oneHot(r.Pclass,[1,2,3]));
  feats.push(...oneHot(r.Sex,['male','female']));
  feats.push(...oneHot(emb,['C','Q','S']));
  if(document.getElementById('add-family-features').checked){
    const fam=(r.SibSp||0)+(r.Parch||0)+1; feats.push(fam,fam===1?1:0);
  }
  return feats;
}

// helpers
function median(a){a=[...a].sort((x,y)=>x-y); const m=Math.floor(a.length/2); return a.length%2?a[m]:(a[m-1]+a[m])/2;}
function mode(a){const c={};a.forEach(v=>c[v]=(c[v]||0)+1);return Object.entries(c).sort((a,b)=>b[1]-a[1])[0][0];}
function std(a){if(!a.length) return 1; const m=a.reduce((s,v)=>s+v,0)/a.length; return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/a.length);}
function oneHot(v,cats){return cats.map(c=>c===v?1:0);}

// ==================== Model ====================
function createModel() {
  const sum=document.getElementById('model-summary');
  try {
    const inputDim=preprocessedTrainData.features.shape[1];
    model=tf.sequential();
    model.add(tf.layers.dense({units:16,activation:'relu',inputShape:[inputDim]}));
    model.add(tf.layers.dense({units:8,activation:'relu'}));
    model.add(tf.layers.dense({units:1,activation:'sigmoid'}));
    model.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});
    sum.innerHTML=`✅ Model created<br/>Input dim: ${inputDim}<br/>Params: ${model.countParams()}`;
    document.getElementById('train-btn').disabled=false;
  }catch(e){sum.innerHTML=`❌ ${e.message}`;}
}

// ==================== Train ====================
async function trainModel(){
  const stat=document.getElementById('training-status');
  const total=preprocessedTrainData.features.shape[0], nf=preprocessedTrainData.features.shape[1];
  const split=Math.floor(total*0.8);
  const trainX=preprocessedTrainData.features.slice([0,0],[split,nf]);
  const trainY=preprocessedTrainData.labels.slice([0],[split]);
  const valX=preprocessedTrainData.features.slice([split,0],[total-split,nf]);
  const valY=preprocessedTrainData.labels.slice([split],[total-split]);
  validationData=valX; validationLabels=valY;

  await model.fit(trainX,trainY,{epochs:20,validationData:[valX,valY],
    callbacks: tfvis.show.fitCallbacks({name:'Training Performance'},['loss','acc','val_loss','val_acc'],{callbacks:['onEpochEnd']})
  });
  stat.innerHTML='✅ Training done';
  validationPredictions=model.predict(validationData);
  document.getElementById('threshold-slider').disabled=false;
  document.getElementById('predict-btn').disabled=false;
  updateMetrics();
}

// ==================== Evaluation ====================
function updateMetrics(){
  const thr=parseFloat(document.getElementById('threshold-slider').value);
  document.getElementById('threshold-value').textContent=thr.toFixed(2);
  const preds=validationPredictions.arraySync().map(p=>p[0]);
  const labels=validationLabels.arraySync();
  let tp=0,tn=0,fp=0,fn=0;
  preds.forEach((p,i)=>{const y=labels[i];const yhat=p>=thr?1:0;
    if(yhat===1&&y===1)tp++;else if(yhat===0&&y===0)tn++;
    else if(yhat===1&&y===0)fp++;else fn++;});
  document.getElementById('confusion-matrix').innerHTML=`TP:${tp}, FP:${fp}, TN:${tn}, FN:${fn}`;
  const acc=(tp+tn)/(tp+tn+fp+fn);
  const prec=tp/(tp+fp||1); const rec=tp/(tp+fn||1);
  const f1=2*prec*rec/(prec+rec||1);
  document.getElementById('performance-metrics').innerHTML=`Acc:${(acc*100).toFixed(2)}% Prec:${prec.toFixed(2)} Rec:${rec.toFixed(2)} F1:${f1.toFixed(2)}`;
}

// ==================== Predict & Export ====================
function predict(){
  const testX=tf.tensor2d(preprocessedTestData.features);
  testPredictions=model.predict(testX);
  const vals=testPredictions.arraySync().map(p=>p[0]);
  const out=preprocessedTestData.ids.map((id,i)=>({PassengerId:id,Survived:vals[i]>=0.5?1:0,Prob:vals[i]}));
  const div=document.getElementById('prediction-output');
  div.innerHTML='<h3>First 10 Predictions</h3>'+makeTable(out.slice(0,10));
  document.getElementById('export-btn').disabled=false;
}

function exportResults(){
  const vals=testPredictions.arraySync().map(p=>p[0]);
  let csv='PassengerId,Survived\n';
  preprocessedTestData.ids.forEach((id,i)=>{csv+=`${id},${vals[i]>=0.5?1:0}\n`;});
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='submission.csv';a.click();
  document.getElementById('export-status').innerHTML='✅ Exported submission.csv';
}
