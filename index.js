// Training only 1 model for characters testing
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Original
// Updates sizes for better prediction intent
const IMAGE_SIZE_W = 64;
const IMAGE_SIZE_H = 64;
const BATCH_SIZE = 16; // 32
const EPOCHS = 30;
// SETTINGS;
// Heroes directory
const DATA_DIR = './tensorflow/train/Heroes'; // Structure: dataset/A/img1.png, dataset/B/img2.png, etc.
// Equipment directory
// const DATA_DIR = './tensorflow/train/Equipment'; // Structure: dataset/A/img1.png, dataset/B/img2.png, etc.

// Load and preprocess a single image
function loadAndProcessImage(imagePath) {
    const buffer = fs.readFileSync(imagePath);
    const imgTensor = tf.node.decodeImage(buffer, 3); // 3 for RGB
    const resized = tf.image.resizeBilinear(imgTensor, [IMAGE_SIZE_W, IMAGE_SIZE_H]);
    const normalized = resized.toFloat().div(255);
    return normalized;
}

// Load dataset from directory structure
function loadDataset() {
    const imageTensors = [];
    const labels = [];
    const classNames = fs.readdirSync(DATA_DIR);

    classNames.forEach((className, labelIndex) => {
        const classPath = path.join(DATA_DIR, className);
        const files = fs.readdirSync(classPath);
        files.forEach(file => {
            const imgPath = path.join(classPath, file);
            const tensor = loadAndProcessImage(imgPath);
            imageTensors.push(tensor);
            labels.push(labelIndex);
        });
    });

    const xs = tf.stack(imageTensors);
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), classNames.length);

    return { xs, ys, classNames };
}

// Create model
function createModel(numClasses) {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_SIZE_W, IMAGE_SIZE_H, 3], // 3 for RGB
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    // Added conv2d
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu'
    }));
    // Added maxPooling2d
    model.add(tf.layers.maxPooling2d({poolSize: 2}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    // Added dropout
    model.add(tf.layers.dropout({rate: 0.5}));
    model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

// Predict a new image
async function predictImage(model, classNames, imagePath) {
    const tensor = loadAndProcessImage(imagePath).expandDims(0);
    const prediction = model.predict(tensor);
    const predictedIndex = (await prediction.argMax(-1).data())[0];
    console.log(`Prediction: ${classNames[predictedIndex]}`);
}

// Main training + prediction logic
async function main() {
    const { xs, ys, classNames } = loadDataset();

    if (classNames.length < 2) {
        console.error('Error: You need at least two character classes to train the model.');
        return;
    }

    const model = createModel(classNames.length);

    await model.fit(xs, ys, {
        epochs: EPOCHS,
        batchSize: BATCH_SIZE,
        shuffle: true,
        callbacks: tf.callbacks.earlyStopping({ monitor: 'loss', patience: 2 })
    });

    // Save model when needed
    await model.save('file://./models/character_model');

    // Test prediction on new image
    // Hero
    // const testImagePath = './tensorflow/user images/character_card_48.png'; // change to your test image
    // Equipment
    // const testImagePath = './tensorflow/user images/equipment_card_5.png'; // change to your test image
    // if (fs.existsSync(testImagePath)) {
    //     // Character Model
    //     const model = await tf.loadLayersModel('file://./models/character_model/model.json');
    //     // Equipment Model
    //     // const model = await tf.loadLayersModel('file://./models/equipment_model/model.json');
    //     await predictImage(model, classNames, testImagePath);
    // } else {
    //     console.warn(`Test image ${testImagePath} not found.`);
    // }
}

main();