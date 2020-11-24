let canvas_width = 180
let canvas_height = 200

const webcamVideo = document.getElementById("webcam-video");
const inputCanvas = document.getElementById("input-canvas");
inputCanvas.width = canvas_width;
inputCanvas.height = canvas_height;

const overlayCanvas = document.getElementById("overlay-canvas");
overlayCanvas.width = canvas_width;
overlayCanvas.height = canvas_height;

const inputCanvasCtx = inputCanvas.getContext("2d");
const overlayCanvasCtx = overlayCanvas.getContext("2d");

function mainLoop() {
    inputCanvasCtx.drawImage(webcamVideo, 0, 0, canvas_width, canvas_height);
    const inputImgData = inputCanvasCtx.getImageData(0, 0, canvas_width, canvas_height);

    // pass the image data to webassembly
    const inputBufImg = Module._malloc(inputImgData.data.length);
    Module.HEAP8.set(inputImgData.data, inputBufImg);

    // detect the face
    const ptr = Module.ccall("recognize_face", "number", ["number", "number", "number"], [inputBufImg, canvas_width, canvas_height]) / Uint32Array.BYTES_PER_ELEMENT;
    const len = Module.HEAPU32[ptr];

    const descriptor = [];
    for (let i = 1; i < len; i += 4) {
        descriptor.push([Module.HEAPU32[ptr + i], Module.HEAPU32[ptr + i + 1], Module.HEAPU32[ptr + i + 2], Module.HEAPU32[ptr + 3]]);
    }

    overlayCanvasCtx.clearRect(0, 0, canvas_width, canvas_height);
    for (const rec of descriptor) {
        overlayCanvasCtx.beginPath()
        overlayCanvasCtx.rect(rec[0], rec[1], rec[2], rec[3])
        overlayCanvasCtx.stroke();
        document.getElementById("detected").innerText = `${JSON.stringify(rec)}`;
    }

    Module._free(inputBufImg);
    Module._free(ptr);

    requestAnimationFrame(mainLoop);
}


window.onload = () => {
    Module.onRuntimeInitialized = () => {
        navigator.mediaDevices.getUserMedia({
            video: true
        }).then(stream => {
            webcamVideo.srcObject = stream;
            webcamVideo.style.display = "none";

            init_shape_model().then(() => {
                return init_net_model()
            }).then(() => {
                requestAnimationFrame(mainLoop);
            })
        }).catch(err => {
            alert(err);
        });
    }
}


const init_shape_model = () => {
    return new Promise((resolve, reject) => {
        const xhrReq = new XMLHttpRequest();
        xhrReq.open("GET", "/models/shape_predictor_5_face_landmarks.dat", true)
        xhrReq.responseType = "arraybuffer";
        xhrReq.onload = (e) => {
            const payload = xhrReq.response;
            if (payload) {

                const model = new Uint8Array(payload);
                const inputBufModel = Module._malloc(model.length);
                Module.HEAPU8.set(model, inputBufModel);
                Module.ccall("init_shape_predictor", null, ["number", "number"], [inputBufModel, model.length]); // This pointer gets freed on the C++ side
                resolve()
            }
        }
        xhrReq.send(null);
    })
}

const init_net_model = () => {
    return new Promise((resolve, reject) => {
        const xhrReq = new XMLHttpRequest();
        xhrReq.open("GET", "/models/dlib_face_recognition_resnet_model_v1.dat", true)
        xhrReq.responseType = "arraybuffer";
        xhrReq.onload = (e) => {
            const payload = xhrReq.response;
            if (payload) {

                const model = new Uint8Array(payload);
                const inputBufModel = Module._malloc(model.length);
                Module.HEAPU8.set(model, inputBufModel);
                Module.ccall("init_resnet_model", null, ["number", "number"], [inputBufModel, model.length]); // This pointer gets freed on the C++ side
                resolve()
            }
        }
        xhrReq.send(null);
    })
}