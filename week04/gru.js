class GRUModel {
  constructor({inputShape=[12,30],outputSize=30,learningRate=0.001}={}) {
    this.inputShape = inputShape;
    this.outputSize = outputSize;
    this.learningRate = learningRate;
  }

  build() {
    const inp = tf.input({shape: this.inputShape});
    let x = tf.layers.gru({units:96,returnSequences:true}).apply(inp);
    x = tf.layers.dropout({rate:0.25}).apply(x);
    x = tf.layers.gru({units:48}).apply(x);
    x = tf.layers.dropout({rate:0.25}).apply(x);
    x = tf.layers.dense({units:64,activation:"relu"}).apply(x);
    const out = tf.layers.dense({units:this.outputSize,activation:"sigmoid"}).apply(x);
    this.model = tf.model({inputs:inp,outputs:out});
    this.model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "binaryCrossentropy",
      metrics: ["binaryAccuracy"]
    });
  }

  async fit(X_train,y_train,{epochs=40,batchSize=16,onEpoch}={}) {
    if (!this.model) this.build();
    for (let epoch = 0; epoch < epochs; epoch++) {
      const history = await this.model.fit(X_train,y_train,{
        epochs:1,batchSize,shuffle:true,validationSplit:0.1,verbose:0
      });
      const logs = history.history;
      onEpoch && onEpoch(epoch, {loss: logs.loss[0], binaryAccuracy: logs.binaryAccuracy[0]});
      await tf.nextFrame(); // yield to browser UI
    }
  }

  predict(X){return this.model.predict(X);}

  async computeAccuracy(preds,yTrue,symbols){
    const p=await preds.array(), y=await yTrue.array();
    const S=symbols.length, H=p[0].length/S;
    const acc=symbols.map(()=>({c:0,t:0}));
    for(let i=0;i<p.length;i++){
      for(let s=0;s<S;s++){
        for(let h=0;h<H;h++){
          const idx=s+h*S;
          const pred=p[i][idx]>=0.5?1:0;
          const truth=y[i][idx];
          if(pred===truth) acc[s].c++;
          acc[s].t++;
        }
      }
    }
    return symbols.map((sym,i)=>({symbol:sym,accuracy:acc[i].c/acc[i].t}));
  }
}
