import { useState, useReducer, useRef, useEffect, useCallback } from "react";

// ─── Constants ────────────────────────────────────────────────────────────────
const ACCESSORIES = [
    { label: "glasses", icon: "👓", description: "Eyeglasses / Sunglasses" },
    { label: "face mask", icon: "😷", description: "Medical / Face Mask" },
    { label: "hat", icon: "🎩", description: "Hat / Cap" },
    { label: "headphones", icon: "🎧", description: "Headphones / Earphones" },
    { label: "scarf", icon: "🧣", description: "Scarf / Neck wrap" },
    { label: "hoodie", icon: "🧥", description: "Hoodie / Jacket" },
    { label: "earrings", icon: "💎", description: "Earrings / Jewelry" },
    { label: "beanie", icon: "🧢", description: "Beanie / Winter hat" },
];

// Pre-compute labels once — avoids re-allocation every scan
const LABELS = [
    ...ACCESSORIES.map((a) => `a person wearing ${a.label}`),
    "a person with no accessories",
];

const THRESHOLD = 0.40;
const SCAN_INTERVAL = 3000;
const MODEL_URL =
    "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// ─── Model state machine ──────────────────────────────────────────────────────
const INIT_MODEL = { status: "idle", progress: 0, message: "", error: null };

function modelReducer(state, action) {
    switch (action.type) {
        case "LOADING": return { ...INIT_MODEL, status: "loading", progress: action.progress, message: action.message };
        case "PROGRESS": return { ...state, progress: action.progress, message: action.message };
        case "READY": return { ...state, status: "ready", progress: 100, message: "Model ready" };
        case "ERROR": return { ...INIT_MODEL, status: "error", error: action.error };
        case "RESET": return INIT_MODEL;
        default: return state;
    }
}

// ─── Pure helpers ─────────────────────────────────────────────────────────────
const confidenceColor = (score) =>
    score >= 0.8 ? "var(--green)" : score >= 0.65 ? "var(--amber)" : "var(--muted)";

// ─── Component ────────────────────────────────────────────────────────────────
export default function AccessoryDetector() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const pipelineRef = useRef(null);   // holds loaded pipeline
    const streamRef = useRef(null);   // holds MediaStream
    const timerRef = useRef(null);   // holds setInterval id
    const busyRef = useRef(false);  // detection in-flight guard (avoids stale-closure bug)

    const [model, dispatch] = useReducer(modelReducer, INIT_MODEL);
    const [cameraActive, setCameraActive] = useState(false);
    const [detections, setDetections] = useState([]);
    const [scanning, setScanning] = useState(false);
    const [scanCount, setScanCount] = useState(0);

    // ── Load model ──────────────────────────────────────────────────────────────
    const loadModel = useCallback(async () => {
        if (pipelineRef.current) return;

        dispatch({ type: "LOADING", progress: 5, message: "Importing Transformers.js…" });
        try {
            const { pipeline, env } = await import(/* @vite-ignore */ MODEL_URL);
            env.allowLocalModels = false;
            if (env.backends?.onnx?.wasm) {
                env.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/";
            }   

            dispatch({ type: "PROGRESS", progress: 20, message: "Downloading vision model…" });

            pipelineRef.current = await pipeline(
                "zero-shot-image-classification",
                "Xenova/clip-vit-base-patch32",
                {
                    progress_callback: ({ status, loaded, total }) => {
                        if (status === "downloading") {
                            const pct = Math.min(Math.round((loaded / total) * 70) + 20, 90);
                            dispatch({
                                type: "PROGRESS",
                                progress: pct,
                                message: `Downloading… ${Math.round((loaded / total) * 100)} %`,
                            });
                        }
                    },
                }
            );

            dispatch({ type: "READY" });
        } catch (err) {
            console.error("[AccessoryAI] model load failed:", err);
            dispatch({ type: "ERROR", error: "Failed to load model: " + err.message });
        }
    }, []);

    // ── Camera ──────────────────────────────────────────────────────────────────
    const stopCamera = useCallback(() => {
        clearInterval(timerRef.current);
        timerRef.current = null;
        busyRef.current = false;

        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;

        if (videoRef.current) videoRef.current.srcObject = null;
        setCameraActive(false);
        setDetections([]);
        setScanning(false);
    }, []);

    const startCamera = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
            });
            streamRef.current = stream;

            const video = videoRef.current;
            if (!video) { stream.getTracks().forEach((t) => t.stop()); return; }
            video.srcObject = stream;
            await video.play();
            setCameraActive(true);
        } catch (err) {
            dispatch({ type: "ERROR", error: "Camera access denied: " + err.message });
        }
    }, []);

    // ── Detection ────────────────────────────────────────────────────────────────
    const captureFrame = useCallback(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2) return null;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);
        return canvas.toDataURL("image/jpeg", 0.8);
    }, []);

    // Key fix: busyRef (not scanning state) is the guard, so this callback
    // is stable across renders and won't reset the interval on every scan.
    const runDetection = useCallback(async () => {
        if (!pipelineRef.current || busyRef.current) return;
        const dataUrl = captureFrame();
        if (!dataUrl) return;

        busyRef.current = true;
        setScanning(true);
        try {
            const results = await pipelineRef.current(dataUrl, LABELS);

            // Simple & clean: match results to our accessories, filter by threshold, sort.
            const detected = ACCESSORIES
                .map((acc) => {
                    const r = results.find((res) => res.label === `a person wearing ${acc.label}`);
                    return { ...acc, score: r ? r.score : 0 };
                })
                .filter((acc) => acc.score >= THRESHOLD)
                .sort((a, b) => b.score - a.score);

            setDetections(detected);
            setScanCount((c) => c + 1);
        } catch (err) {
            console.error("[AccessoryAI] detection error:", err);
        } finally {
            busyRef.current = false;
            setScanning(false);
        }
    }, [captureFrame]); // stable — no scanning dep

    // Start interval only when camera + model are both ready
    useEffect(() => {
        if (!cameraActive || model.status !== "ready") return;
        runDetection(); // immediate first scan
        timerRef.current = setInterval(runDetection, SCAN_INTERVAL);
        return () => clearInterval(timerRef.current);
    }, [cameraActive, model.status, runDetection]);

    // Cleanup on unmount
    useEffect(() => stopCamera, [stopCamera]);

    // ── Start handler ────────────────────────────────────────────────────────────
    const handleStart = useCallback(async () => {
        dispatch({ type: "RESET" });
        if (!pipelineRef.current) await loadModel();
        await startCamera();
    }, [loadModel, startCamera]);

    // ── Derived flags ────────────────────────────────────────────────────────────
    const isLoading = model.status === "loading";
    const isError = model.status === "error";

    // ── Render ───────────────────────────────────────────────────────────────────
    return (
        <div className="ad-root">
            <header className="ad-header">
                <div className="ad-eyebrow">Vision · Real-time · On-device</div>
                <h1 className="ad-title">AccessoryAI</h1>
                <p className="ad-subtitle">
                    Real-time accessory detection · No backend · No data leaves your device
                </p>
            </header>

            <main className="ad-card">
                {/* ── Camera viewport ── */}
                <div className="ad-viewport">
                    <video ref={videoRef} muted playsInline className={`ad-video${cameraActive ? " ad-video--active" : ""}`} />
                    <canvas ref={canvasRef} hidden />

                    {!cameraActive && !isLoading && (
                        <div className="ad-placeholder">
                            <span className="ad-placeholder__icon" aria-hidden="true">📷</span>
                            <p>Camera feed will appear here</p>
                        </div>
                    )}

                    {isLoading && (
                        <div className="ad-loading">
                            <div className="ad-spinner" aria-hidden="true" />
                            <p className="ad-loading__status">{model.message}</p>
                            <div
                                className="ad-progress"
                                role="progressbar"
                                aria-valuenow={model.progress}
                                aria-valuemin={0}
                                aria-valuemax={100}
                            >
                                <div className="ad-progress__bar" style={{ width: `${model.progress}%` }} />
                            </div>
                            <p className="ad-loading__note">{model.progress}% · ~30 MB first load, cached after</p>
                        </div>
                    )}

                    {cameraActive && (
                        <>
                            <div className="ad-badge ad-badge--live">
                                <span className={`ad-dot${scanning ? " ad-dot--scanning" : ""}`} />
                                <span>{scanning ? "ANALYZING" : "LIVE"}</span>
                            </div>
                            <div className="ad-badge ad-badge--count">#{scanCount} scans</div>
                            {["tl", "tr", "bl", "br"].map((p) => (
                                <div key={p} className={`ad-corner ad-corner--${p}`} aria-hidden="true" />
                            ))}
                        </>
                    )}
                </div>

                {/* ── Controls + Results ── */}
                <section className="ad-panel" aria-label="Controls and detections">
                    {isError && (
                        <div className="ad-error" role="alert">⚠️ {model.error}</div>
                    )}

                    <div className="ad-actions">
                        {!cameraActive ? (
                            <button
                                id="btn-start"
                                className="ad-btn ad-btn--primary"
                                onClick={handleStart}
                                disabled={isLoading}
                            >
                                {isLoading ? "Loading model…" : "▶️ Start Detection"}
                            </button>
                        ) : (
                            <button
                                id="btn-stop"
                                className="ad-btn ad-btn--danger"
                                onClick={stopCamera}
                            >
                                ■ Stop
                            </button>
                        )}
                    </div>

                    <div className="ad-detections">
                        <h2 className="ad-section-label">Detected Accessories</h2>

                        {!cameraActive && detections.length === 0 && (
                            <p className="ad-hint">Start the camera to begin detection.</p>
                        )}
                        {cameraActive && detections.length === 0 && scanCount > 0 && (
                            <p className="ad-hint">✨ No accessories detected</p>
                        )}

                        <ul className="ad-detection-list">
                            {detections.map((det) => (
                                <li key={det.label} className="ad-detection-item">
                                    <span className="ad-detection__icon" aria-hidden="true">{det.icon}</span>
                                    <div className="ad-detection__info">
                                        <span className="ad-detection__name">{det.description}</span>
                                        <div className="ad-bar" role="presentation">
                                            <div
                                                className="ad-bar__fill"
                                                style={{
                                                    width: `${Math.round(det.score * 100)}%`,
                                                    background: confidenceColor(det.score),
                                                }}
                                            />
                                        </div>
                                    </div>
                                    <span
                                        className="ad-detection__score"
                                        style={{ color: confidenceColor(det.score) }}
                                    >
                                        {Math.round(det.score * 100)}%
                                    </span>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {cameraActive && (
                        <div className="ad-tags">
                            <h2 className="ad-section-label">Scanning for</h2>
                            <div className="ad-tag-list">
                                {ACCESSORIES.map((a) => {
                                    const active = detections.some((d) => d.label === a.label);
                                    return (
                                        <span key={a.label} className={`ad-tag${active ? " ad-tag--active" : ""}`}>
                                            {a.icon} {a.label}
                                        </span>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </section>
            </main>

            <footer className="ad-footer">
                Powered by CLIP · Transformers.js · All processing on-device
            </footer>
        </div >
    );
}