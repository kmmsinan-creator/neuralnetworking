// gru.js
// Defines the GRU-based model.

class GRUModel {
  constructor({inputShape=[12,20],outputSize=30,learningRate=0.001}={}){
    this.inputShape=inputShape;
    this.outputSize=outputSize;
    this.learningRate=learningRate;
  }

  build(){
    const inp=tf.input({shape:this.inputShape});
    let x=tf.layers.gru({units:64,returnSequences:true}).apply(inp);
    x=tf.layers.gru({units:32}).apply(x);
    x=tf.layers.dense({units:32,activation:"relu"}).apply(x);
    const out=tf.layers.dense({units:this.outputSize,activation:"sigmoid"}).apply(x);
    this.model=tf.model({inputs:inp,outputs:out});
    this.model.compile({
      optimizer:tf.train.adam(this.learningRate),
      loss:"binaryCrossentropy",
      metrics:["binaryAccuracy"]
    });
  }

  async fit(X_train,y_train,{epochs=25,batchSize=32,onEpoch}={}){
    if(!this.model) this.build();
    return await this.model.fit(X_train,y_train,{
      epochs,batchSize,shuffle:false,
      callbacks:{onEpochEnd:(e,logs)=>{onEpoch&&onEpoch(e,logs);}}
    });
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
