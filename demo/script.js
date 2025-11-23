const CLASS_NAMES = [
    { finnish: "Ahven", english: "Perch" },
    { finnish: "Harjus", english: "Grayling" },
    { finnish: "Hauki", english: "Northern Pike" },
    { finnish: "Kiiski", english: "Ruffe" },
    { finnish: "Kirjolohi", english: "Rainbow Trout" },
    { finnish: "Kuha", english: "Zander (Pike-perch)" },
    { finnish: "Lahna", english: "Bream" },
    { finnish: "Lohi", english: "Salmon" },
    { finnish: "Made", english: "Burbot" },
    { finnish: "Pasuri", english: "Silver Bream" },
    { finnish: "Särki", english: "Roach" },
    { finnish: "Säyne", english: "Ide" },
    { finnish: "Siika", english: "Whitefish" },
    { finnish: "Taimen", english: "Trout" }
];

const statusElement = document.getElementById('status');
const resultElement = document.getElementById('result');
const imageUpload = document.getElementById('image-upload');
const uploadStatusElement = document.getElementById('upload-status');
const imagePreview = document.getElementById('image-preview');
const allProbabilitiesElement = document.getElementById('all-probabilities');

let model;

// loads and warms up the model
async function loadModel() {
    try {
        uploadStatusElement.textContent = 'Loading model...';
        imageUpload.disabled = true; 
        statusElement.textContent = '';
        
        model = await tf.loadGraphModel('web_model_output/model.json');
        
        uploadStatusElement.textContent = 'Warming up model...';
        const zeros = tf.zeros([1, 224, 224, 3]);
        await model.predict(zeros).data();
        zeros.dispose();

        uploadStatusElement.textContent = '';
        imageUpload.disabled = false;
    } catch (error) {
        console.error('Error loading model:', error);
        uploadStatusElement.textContent = 'Error loading model';
        statusElement.textContent = 'Error loading model. Please check console for details.';
    }
}

// makes a prediction on the image
async function predict(imgElement) {
    if (!model) return null;

    const tensor = tf.tidy(() => {
        let img = tf.browser.fromPixels(imgElement);
        img = tf.image.resizeBilinear(img, [224, 224]);
        img = img.expandDims(0);
        return img.toFloat();
    });

    try {
        const predictions = await model.predict(tensor).data();
        const maxIndex = predictions.indexOf(Math.max(...predictions));
        tensor.dispose();
        
        const allProbs = Array.from(predictions).map((prob, i) => ({
            finnishName: CLASS_NAMES[i].finnish,
            englishName: CLASS_NAMES[i].english,
            probability: (prob * 100).toFixed(2)
        }));

        allProbs.sort((a, b) => parseFloat(b.probability) - parseFloat(a.probability)); // sort by probability

        return {
            className: CLASS_NAMES[maxIndex].finnish,
            probability: (predictions[maxIndex] * 100).toFixed(2),
            allProbabilities: allProbs
        };
    } catch (error) {
        console.error(error);
        tensor.dispose();
        return null;
    }
}

imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (!file) return;

    resultElement.textContent = '';
    if (allProbabilitiesElement) {
        allProbabilitiesElement.innerHTML = '';
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        imagePreview.onload = async () => {
            const result = await predict(imagePreview);
            if (result) {
                resultElement.textContent = `${result.className} (${result.probability}%)`;
                
                if (allProbabilitiesElement) {
                    let html = '<table>';
                    html += '<thead><tr><th>Species</th><th>Probability</th></tr></thead><tbody>';
                    
                    result.allProbabilities.forEach(item => {
                        const isTop = item.finnishName === result.className;
                        const rowStyle = isTop ? 'style="font-weight: bold;"' : '';
                        html += `<tr ${rowStyle}><td>${item.finnishName}</td><td>${item.probability}%</td></tr>`;
                    });
                    html += '</tbody></table>';
                    allProbabilitiesElement.innerHTML = html;
                }
            }
        };
    };
    reader.readAsDataURL(file);
});

loadModel();
