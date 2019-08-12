var NUM_CLASSES = 0;


const webcam = new Webcam(document.getElementById('webcam'));

const controllerDataset = new ControllerDataset(NUM_CLASSES);

let mobilenet;
let model;


async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');


  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}


ui.setExampleHandler(label => {
  tf.tidy(() => {
    const img = webcam.capture();
    controllerDataset.addExample(mobilenet.predict(img), label);

    NUM_CLASSES = totals.filter(total => total > 0).length;

  
    ui.drawThumb(img, label);
  });
  ui.trainStatus('TRAIN');
});


async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  model = tf.sequential({
    layers: [
    
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: tf.initializers.varianceScaling(
            {scale: 1.0, mode: 'fanIn', distribution: 'normal'}),
        useBias: true
      }),
     
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: tf.initializers.varianceScaling(
            {scale: 1.0, mode: 'fanIn', distribution: 'normal'}),
        useBias: false,
        activation: 'softmax'
      })
    ]
  });


  const optimizer = tf.train.adam(ui.getLearningRate());
 
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});


  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }
  let loss = 0;
 
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        document.getElementById('train').className =
            'train-model-button train-status';
        loss = logs.loss.toFixed(5);
        ui.trainStatus('LOSS: ' + loss);
      },
      onTrainEnd: () => {
        if (loss > 1) {
          document.getElementById('user-help-text').innerText =
              'Model is not trained well. Add more samples and train again';
        } else {
          document.getElementById('user-help-text').innerText =
              'Test or download the model. You can even add images for other classes and train again.';
          ui.enableModelTest();
          ui.enableModelDownload();
        }
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
  
      const img = webcam.capture();

      const activation = mobilenet.predict(img);

      const predictions = model.predict(activation);


      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();

    ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('ENCODING...');
  controllerDataset.ys = null;
  controllerDataset.addLabels(NUM_CLASSES);
  ui.trainStatus('TRAINING...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});
document.getElementById('predict').addEventListener('click', () => {
  isPredicting = true;
  predict();
});

document.getElementById('stop-predict').addEventListener('click', () => {
  isPredicting = false;
  predict();
});

async function init() {
  try {
    await webcam.setup();
    console.log('Webcam is on');
  } catch (e) {
    console.log(e);
    document.getElementById('no-webcam').style.display = 'block';
    document.getElementById('webcam-inner-wrapper').className =
        'webcam-inner-wrapper center grey-bg';
    document.getElementById('bottom-section').style.pointerEvents = 'none';
  }

  mobilenet = await loadMobilenet();


  tf.tidy(() => mobilenet.predict(webcam.capture()));

  ui.init();
}

var a = document.createElement('a');
document.body.appendChild(a);
a.style = 'display: none';

document.getElementById('download-model').onclick =
    async () => {
  await model.save('downloads://model');

  var text = controlsCaptured.join(',');
  var blob = new Blob([text], {type: 'text/csv;charset=utf-8'});
  var url = window.URL.createObjectURL(blob);
  a.href = url;
  a.download = 'labels.txt';
  a.click();
  window.URL.revokeObjectURL(url);
}

init();
