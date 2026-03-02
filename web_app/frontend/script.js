/* ==========================================
   SignLiveAI — Core JavaScript
   Camera, MediaPipe, Recording, API, TTS
   ========================================== */

// ====== STATE ======
let cameraActive = false;
let recording = false;
let recordedFrames = [];
let currentPrediction = null;
let sentenceWords = [];
let handsModel = null;
let poseModel = null;
let camera = null;
const MIN_FRAMES = 8;
const MAX_FRAMES = 30;

// ====== INIT ======
document.addEventListener("DOMContentLoaded", async () => {
    await checkHealth();
    loadGlosses();
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
});

async function checkHealth() {
    const statusEl = document.getElementById("headerStatus");
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        if (data.status === "healthy") {
            statusEl.innerHTML = `<span class="status-dot ${data.model_loaded ? "online" : "loading"}"></span>
                <span class="status-text">${data.model_loaded ? "Model Ready" : "Model Not Loaded"}</span>`;
        }
    } catch {
        statusEl.innerHTML = `<span class="status-dot offline"></span><span class="status-text">Server Offline</span>`;
    }
}

async function loadGlosses() {
    try {
        const res = await fetch("/api/glosses");
        const data = await res.json();
        const grid = document.getElementById("glossaryGrid");
        document.getElementById("glossCount").textContent = data.count;
        grid.innerHTML = data.glosses.map(g => `<span class="gloss-chip">${g}</span>`).join("");
    } catch {
        console.warn("Could not load glosses");
    }
}

function filterGlosses() {
    const q = document.getElementById("glossSearch").value.toLowerCase();
    document.querySelectorAll(".gloss-chip").forEach(chip => {
        chip.style.display = chip.textContent.toLowerCase().includes(q) ? "" : "none";
    });
}

// ====== CAMERA ======
async function toggleCamera() {
    const btn = document.getElementById("btnToggleCamera");
    const overlay = document.getElementById("cameraOverlay");
    const recordBtn = document.getElementById("btnRecord");

    if (cameraActive) {
        stopCamera();
        btn.innerHTML = '<span class="btn-icon">▶</span> Start Camera';
        overlay.classList.remove("hidden");
        recordBtn.disabled = true;
        cameraActive = false;
        return;
    }

    btn.innerHTML = '<span class="btn-icon">⏳</span> Loading...';
    btn.disabled = true;

    try {
        await initMediaPipe();
        overlay.classList.add("hidden");
        btn.innerHTML = '<span class="btn-icon">⏹</span> Stop Camera';
        btn.disabled = false;
        recordBtn.disabled = false;
        cameraActive = true;
    } catch (err) {
        console.error("Camera error:", err);
        btn.innerHTML = '<span class="btn-icon">▶</span> Start Camera';
        btn.disabled = false;
        alert("Could not access camera. Please allow camera permissions.");
    }
}

async function initMediaPipe() {
    const videoEl = document.getElementById("webcam");
    const canvasEl = document.getElementById("landmarkCanvas");

    // Init Hands
    handsModel = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
    });
    handsModel.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });

    // Init Pose
    poseModel = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`,
    });
    poseModel.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });

    // Store latest results
    let latestHandResults = null;
    let latestPoseResults = null;

    handsModel.onResults((results) => {
        latestHandResults = results;
        drawLandmarks(canvasEl, latestHandResults, latestPoseResults);
        if (recording) captureFrame(latestHandResults, latestPoseResults);
    });

    poseModel.onResults((results) => {
        latestPoseResults = results;
    });

    // Camera
    camera = new Camera(videoEl, {
        onFrame: async () => {
            await handsModel.send({ image: videoEl });
            await poseModel.send({ image: videoEl });
        },
        width: 640,
        height: 480,
    });
    await camera.start();
}

function stopCamera() {
    if (camera) {
        camera.stop();
        camera = null;
    }
    handsModel = null;
    poseModel = null;
}

function drawLandmarks(canvas, handResults, poseResults) {
    const ctx = canvas.getContext("2d");
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw hand landmarks
    if (handResults && handResults.multiHandLandmarks) {
        for (const landmarks of handResults.multiHandLandmarks) {
            // Connections
            drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: "#6366f180", lineWidth: 2 });
            // Points
            for (const lm of landmarks) {
                ctx.beginPath();
                ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
                ctx.fillStyle = "#818cf8";
                ctx.fill();
                ctx.strokeStyle = "#312e81";
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }
    }

    // Draw pose landmarks (upper body only: 0-24)
    if (poseResults && poseResults.poseLandmarks) {
        const upperIndices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24];
        for (const idx of upperIndices) {
            const lm = poseResults.poseLandmarks[idx];
            if (lm) {
                ctx.beginPath();
                ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
                ctx.fillStyle = "#22c55e80";
                ctx.fill();
            }
        }
    }
}

// ====== RECORDING & FRAMES ======
function toggleRecording() {
    const btn = document.getElementById("btnRecord");
    const indicator = document.getElementById("recordingIndicator");
    const textEl = document.getElementById("recordBtnText");

    if (recording) {
        // Stop recording
        recording = false;
        indicator.style.display = "none";
        textEl.textContent = "Start Recording";
        btn.classList.remove("recording");

        // Send for prediction
        if (recordedFrames.length >= MIN_FRAMES) {
            sendPrediction();
        } else {
            showIdle("Not enough frames. Please record a longer sign.");
        }
    } else {
        // Start recording
        recordedFrames = [];
        updateFrameCounter(0);
        recording = true;
        indicator.style.display = "flex";
        textEl.textContent = "Stop Recording";
        btn.classList.add("recording");
        showIdle("Recording... Perform the sign now.");
    }
}

function captureFrame(handResults, poseResults) {
    if (!recording || recordedFrames.length >= MAX_FRAMES) {
        if (recordedFrames.length >= MAX_FRAMES) {
            toggleRecording(); // auto-stop
        }
        return;
    }

    // Build a frame of 75 landmarks (42 hand + 33 pose) with [x, y]
    const frame = [];

    // RIGHT HAND (21 landmarks) and LEFT HAND (21 landmarks)
    // IMPORTANT: Do NOT un-mirror MediaPipe labels. Training data was generated
    // using labels as-is ("Right" → RH, "Left" → LH), so inference must match.
    if (handResults && handResults.multiHandLandmarks && handResults.multiHandedness) {
        let rightHand = null;
        let leftHand = null;
        for (let i = 0; i < handResults.multiHandedness.length; i++) {
            const label = handResults.multiHandedness[i].label;
            // Match training: "Right" label → RH slots, "Left" label → LH slots
            if (label === "Right") rightHand = handResults.multiHandLandmarks[i];
            else leftHand = handResults.multiHandLandmarks[i];
        }

        // RH x0..x20, y0..y20 → landmarks 0..20
        for (let j = 0; j < 21; j++) {
            if (rightHand) frame.push([rightHand[j].x, rightHand[j].y]);
            else frame.push([-2, -2]);
        }
        // LH → landmarks 21..41
        for (let j = 0; j < 21; j++) {
            if (leftHand) frame.push([leftHand[j].x, leftHand[j].y]);
            else frame.push([-2, -2]);
        }
    } else {
        // No hands: 42 padded landmarks (padVal=-2 to match training)
        for (let j = 0; j < 42; j++) frame.push([-2, -2]);
    }

    // POSE (33 landmarks) → landmarks 42..74
    if (poseResults && poseResults.poseLandmarks) {
        for (let j = 0; j < 33; j++) {
            frame.push([poseResults.poseLandmarks[j].x, poseResults.poseLandmarks[j].y]);
        }
    } else {
        for (let j = 0; j < 33; j++) frame.push([-2, -2]);
    }

    recordedFrames.push(frame);
    updateFrameCounter(recordedFrames.length);
}

function updateFrameCounter(count) {
    document.getElementById("frameCount").textContent = count;
    const pct = Math.min((count / MAX_FRAMES) * 100, 100);
    document.getElementById("progressFill").style.width = pct + "%";
}

// ====== PREDICTION ======
async function sendPrediction() {
    showIdle("⏳ Predicting...");

    const body = {
        frames: recordedFrames.map(landmarks => ({ landmarks })),
    };

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        const data = await res.json();

        if (data.success) {
            showPrediction(data);
            currentPrediction = data;
            document.getElementById("btnSpeak").disabled = false;

            // Add to sentence
            addWordToSentence(data.predicted_gloss);
        } else {
            showIdle("⚠️ " + (data.error || "Prediction failed"));
        }
    } catch (err) {
        console.error("Prediction error:", err);
        showIdle("⚠️ Could not reach server");
    }
}

function showPrediction(data) {
    document.getElementById("predictionIdle").style.display = "none";
    const resultEl = document.getElementById("predictionResult");
    resultEl.style.display = "block";

    document.getElementById("resultGloss").textContent = data.predicted_gloss;
    const confPct = Math.round(data.confidence * 100);
    document.getElementById("confidenceFill").style.width = confPct + "%";
    document.getElementById("confidenceText").textContent = confPct + "%";

    // Top 5
    const top5Section = document.getElementById("top5Section");
    const top5List = document.getElementById("top5List");
    if (data.top5 && data.top5.length > 0) {
        top5Section.style.display = "block";
        top5List.innerHTML = data.top5.map((item, i) => `
            <div class="top5-item" style="animation-delay: ${i * 0.05}s">
                <span class="top5-gloss">${item.gloss}</span>
                <span class="top5-conf">${Math.round(item.confidence * 100)}%</span>
            </div>
        `).join("");
    }
}

function showIdle(message) {
    document.getElementById("predictionResult").style.display = "none";
    document.getElementById("top5Section").style.display = "none";
    const idle = document.getElementById("predictionIdle");
    idle.style.display = "block";
    idle.querySelector("p").textContent = message || "Record a sign to see the translation";
}

function clearAll() {
    recordedFrames = [];
    recording = false;
    currentPrediction = null;
    updateFrameCounter(0);
    document.getElementById("recordingIndicator").style.display = "none";
    document.getElementById("recordBtnText").textContent = "Start Recording";
    document.getElementById("btnSpeak").disabled = true;
    showIdle();
}

// ====== SENTENCE BUILDER ======
function addWordToSentence(word) {
    sentenceWords.push(word);
    renderSentence();
}

function renderSentence() {
    const display = document.getElementById("sentenceDisplay");
    if (sentenceWords.length === 0) {
        display.innerHTML = '<span class="sentence-placeholder">Signs will appear here...</span>';
    } else {
        display.innerHTML = sentenceWords.map(w => `<span class="sentence-word">${w}</span>`).join(" ");
    }
}

function undoLastWord() {
    sentenceWords.pop();
    renderSentence();
}

function clearSentence() {
    sentenceWords = [];
    renderSentence();
}

// ====== TEXT-TO-SPEECH ======
function loadVoices() {
    const select = document.getElementById("voiceSelect");
    const voices = window.speechSynthesis.getVoices();
    select.innerHTML = '<option value="">Default Voice</option>';
    voices
        .filter(v => v.lang.startsWith("en"))
        .forEach((v, i) => {
            const opt = document.createElement("option");
            opt.value = i;
            opt.textContent = `${v.name} (${v.lang})`;
            select.appendChild(opt);
        });
}

function speakText(text) {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;

    const selectVal = document.getElementById("voiceSelect").value;
    if (selectVal !== "") {
        const voices = window.speechSynthesis.getVoices().filter(v => v.lang.startsWith("en"));
        if (voices[parseInt(selectVal)]) utterance.voice = voices[parseInt(selectVal)];
    }

    window.speechSynthesis.speak(utterance);
}

function speakResult() {
    if (currentPrediction) speakText(currentPrediction.predicted_gloss);
}

function speakSentence() {
    if (sentenceWords.length > 0) speakText(sentenceWords.join(" "));
}
