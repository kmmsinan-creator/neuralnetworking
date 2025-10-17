// gru.js
// GRU network with dropout, regularization, and learning rate scheduling

class GRUModel {
  constructor({inputShape=[12,30],outputSize=30,learningRate=0.001}={}) {
    this.inputShape = inputShape;
    this.outputSize = outputSize;
    this.learningRate = learningRate;
  }

  build() {
    const l2 = tf.regularizers.l2({l2: 1e-4});
    const inp = tf.input({shape: this.inputShape});

    let x = tf.layers.gru({units:128,returnSequences:true,
                           kernelRegularizer:l2,recurrentRegularizer:l2}).apply(inp);
    x = tf.layers.dropout({rate:0.3}).apply(x);
    x = tf.layers.gru({units:64,returnSequences:true,
                       kernelRegularizer:l2,recurrentRegularizer:l2}).apply(x);
    x = tf.layers.dropout({rate:0.3}).apply(x);
    x = tf.layers.gru({units:32,
                       kernelRegularizer:l2,recurrentRegularizer:l2}).apply(x);

    x = tf.layers.dense({units:64,activation:"relu",kernelRegularizer:l2}).apply(x);
    x = tf.layers.dropout({rate:0.2}).apply(x);
    const out = tf.layers.dense({units:this.outputSize,activation:"sigmoid"}).apply(x);

    this.model = tf.model({inputs:inp,outputs:out});
    const opt = tf.train.adam(this.learningRate);
    this.model.compile({optimizer:opt,loss:"binaryCrossentropy",metrics:["binaryAccuracy"]});
  }

  async fit(X_train,y_train,{epochs=60,batchSize=16,onEpoch}={}) {
    if (!this.model) this.build();

    const cb = {
      onEpochEnd: (epoch, logs) => {
        onEpoch && onEpoch(epoch, logs);
        if ((epoch+1)%15===0 && logs.binaryAccuracy < 0.7) {
          const oldLR = this.model.optimizer.learningRate;
          const newLR = oldLR.mul(tf.scalar(0.5));
          this.model.optimizer.learningRate = newLR;
          console.log(`Lowered LR to ${newLR.dataSync()[0].toFixed(6)}`);
        }
      }
    };

    return await this.model.fit(X_train,y_train,{
      epochs,batchSize,shuffle:true,validationSplit:0.1,callbacks:cb
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
