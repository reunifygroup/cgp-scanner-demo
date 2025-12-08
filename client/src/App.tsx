import { useState, useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import "./App.css";

interface ScanResult {
    cardId: string;
    cardName: string;
    confidence: number; // cosine similarity in [0, 100]
    debugImage?: string;
}

type EmbeddingMap = Record<string, tf.Tensor1D>;

function App() {
    const [isScanning, setIsScanning] = useState(false);
    const [result, setResult] = useState<ScanResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [modelStatus, setModelStatus] = useState<string>("Loading model...");
    const [isModelLoaded, setIsModelLoaded] = useState(false);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const intervalRef = useRef<number | null>(null);

    // MobileNet feature extractor + precomputed embeddings
    const mobilenetRef = useRef<any>(null); // keep it simple for now
    const embeddingsRef = useRef<EmbeddingMap>({});

    // 🧠 Load MobileNet + embeddings on mount
    useEffect(() => {
        async function loadModelAndEmbeddings() {
            try {
                setModelStatus("Loading feature extractor and embeddings...");

                // Load MobileNet feature extractor
                const mbnet = await mobilenet.load({
                    version: 2,
                    alpha: 1.0,
                });
                mobilenetRef.current = mbnet;
                console.log("✅ MobileNet loaded");

                // Load precomputed embeddings JSON
                const embeddingsResponse = await fetch("/model/card_embeddings.json");
                const embeddingsJson = (await embeddingsResponse.json()) as Record<string, number[]>;

                const embMap: EmbeddingMap = {};
                for (const [cardId, vector] of Object.entries(embeddingsJson)) {
                    const t = tf.tensor1d(vector);
                    // Normalize to unit length for cosine similarity
                    const norm = t.norm();
                    const normalized = t.div(norm) as tf.Tensor1D;
                    embMap[cardId] = normalized;
                    t.dispose();
                    norm.dispose();
                }

                embeddingsRef.current = embMap;

                const numCards = Object.keys(embMap).length;
                setModelStatus(`Model ready · ${numCards} cards`);
                setIsModelLoaded(true);

                console.log("✅ Embeddings loaded for cards:", Object.keys(embMap));
            } catch (err) {
                console.error("❌ Failed to load model/embeddings:", err);
                setError("Failed to load model or embeddings: " + (err as Error).message);
                setModelStatus("Model load failed");
                setIsModelLoaded(false);
            }
        }

        loadModelAndEmbeddings();

        // Cleanup on unmount
        return () => {
            stopScanning();
            // Dispose embedding tensors
            Object.values(embeddingsRef.current).forEach((t) => t.dispose());
            embeddingsRef.current = {};
            mobilenetRef.current = null;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // 🧪 Cosine similarity between two unit vectors
    const cosineSimilarity = (a: tf.Tensor1D, b: tf.Tensor1D): number => {
        const dot = a.dot(b) as tf.Scalar;
        const val = dot.dataSync()[0] as number;
        dot.dispose();
        return val;
    };

    // 📸 Start camera and scanning
    const startScanning = async () => {
        if (!mobilenetRef.current || Object.keys(embeddingsRef.current).length === 0) {
            setError("Model or embeddings not loaded yet. Please wait...");
            return;
        }

        try {
            setError(null);

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "environment",
                    width: { ideal: 720 },
                    height: { ideal: 1000 },
                    aspectRatio: { ideal: 0.72 },
                },
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                streamRef.current = stream;
                setIsScanning(true);

                // Capture frames every 500ms
                intervalRef.current = window.setInterval(() => {
                    captureAndPredict();
                }, 500);
            }
        } catch (err) {
            setError("Failed to access camera: " + (err as Error).message);
        }
    };

    // 🛑 Stop camera and scanning
    const stopScanning = () => {
        if (intervalRef.current !== null) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }

        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }

        setIsScanning(false);
    };

    // 🎯 Capture frame and run embedding-based similarity
    // 🎯 Capture frame and run embedding-based similarity
    // 🎯 Capture frame and run embedding-based similarity
    const captureAndPredict = async () => {
        if (!videoRef.current || !canvasRef.current || !mobilenetRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");

        if (!context || video.readyState !== video.HAVE_ENOUGH_DATA) return;

        try {
            const TARGET_WIDTH = 320;
            const TARGET_HEIGHT = 440;
            const TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT;

            canvas.width = TARGET_WIDTH;
            canvas.height = TARGET_HEIGHT;

            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            if (!videoWidth || !videoHeight) {
                console.warn("⚠️ videoWidth/videoHeight is zero — iOS metadata not ready");
                return;
            }

            const videoAspect = videoWidth / videoHeight;

            let sx = 0;
            let sy = 0;
            let sWidth = videoWidth;
            let sHeight = videoHeight;

            // Center-crop to match target aspect ratio (no stretching)
            if (videoAspect > TARGET_ASPECT) {
                sHeight = videoHeight;
                sWidth = sHeight * TARGET_ASPECT;
                sx = (videoWidth - sWidth) / 2;
                sy = 0;
            } else {
                sWidth = videoWidth;
                sHeight = sWidth / TARGET_ASPECT;
                sx = 0;
                sy = (videoHeight - sHeight) / 2;
            }

            context.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);

            const mbnet = mobilenetRef.current;
            if (!mbnet) return;

            // 🔍 NEW: crop the inner region (approx artwork area) before embedding
            const embTensor = tf.tidy(() => {
                // [H, W, 3]
                const full = tf.browser.fromPixels(canvas);

                const [h, w] = full.shape; // [440, 320]

                // Rough crop: keep central area, removing borders/text boxes
                const topFrac = 0.18;
                const bottomFrac = 0.78;
                const leftFrac = 0.1;
                const rightFrac = 0.9;

                const y1 = Math.floor(h * topFrac);
                const y2 = Math.floor(h * bottomFrac);
                const x1 = Math.floor(w * leftFrac);
                const x2 = Math.floor(w * rightFrac);

                const cropH = y2 - y1;
                const cropW = x2 - x1;

                const cropped = full.slice([y1, x1, 0], [cropH, cropW, 3]); // [cropH, cropW, 3]

                // Resize to 224×224 and normalize 0–1
                const resized = tf.image.resizeBilinear(cropped, [224, 224]);
                const floatImg = resized.toFloat().div(255.0);

                // Call mobilenet.infer on tensor instead of canvas
                return mbnet.infer(floatImg, { embedding: true }) as tf.Tensor;
            });

            const emb1d = embTensor.squeeze() as tf.Tensor1D;

            // Normalize to unit length
            const norm = emb1d.norm();
            const queryEmbedding = emb1d.div(norm) as tf.Tensor1D;

            // Compare with all stored embeddings
            let bestCardId: string | null = null;
            let bestScore = -Infinity;
            const allScores: { cardId: string; score: number }[] = [];

            for (const [cardId, refEmbedding] of Object.entries(embeddingsRef.current)) {
                const score = cosineSimilarity(queryEmbedding, refEmbedding); // [-1, 1]
                allScores.push({ cardId, score });
                if (score > bestScore) {
                    bestScore = score;
                    bestCardId = cardId;
                }
            }

            allScores.sort((a, b) => b.score - a.score);
            const secondBestScore = allScores.length > 1 ? allScores[1].score : -Infinity;

            // Cleanup
            embTensor.dispose();
            emb1d.dispose();
            norm.dispose();
            queryEmbedding.dispose();

            if (!bestCardId) return;

            const confidencePercent = bestScore * 100;

            console.log("🔍 Similarities:", {
                bestCardId,
                bestScore,
                secondBestScore,
                confidencePercent,
                tensors: tf.memory().numTensors,
                allScores: allScores.map((s) => ({
                    cardId: s.cardId,
                    score: s.score,
                    confidence: (s.score * 100).toFixed(1) + "%",
                })),
            });

            const MIN_SIMILARITY = 0.5;
            const MIN_MARGIN = 0.05;

            if (bestScore < MIN_SIMILARITY || bestScore - secondBestScore < MIN_MARGIN) {
                setResult(null);
                return;
            }

            const nameParts = bestCardId.split("_");
            const cardName = nameParts.length > 1 ? nameParts.slice(1).join(" ") : bestCardId;

            const debugImage = canvas.toDataURL("image/png");

            setResult({
                cardId: bestCardId,
                cardName,
                confidence: confidencePercent,
                debugImage,
            });
        } catch (err) {
            console.error("❌ Prediction error:", err);
        }
    };

    return (
        <div className="app">
            <header>
                <h1>CGP Card Scanner</h1>
                <p>AI-powered instant card recognition (embeddings)</p>
                <div className="model-status">{modelStatus}</div>
            </header>

            <main>
                <div className="scanner-container">
                    <div className="video-wrapper">
                        <video ref={videoRef} autoPlay playsInline muted className={isScanning ? "active" : "hidden"} />
                    </div>

                    {/* Hidden canvas for frame capture */}
                    <canvas ref={canvasRef} style={{ display: "none" }} />

                    {/* Control button */}
                    {!isScanning ? (
                        <button onClick={startScanning} className="btn-primary" disabled={!isModelLoaded}>
                            Start Scanner
                        </button>
                    ) : (
                        <button onClick={stopScanning} className="btn-secondary">
                            Stop Scanner
                        </button>
                    )}
                </div>

                {/* Error display */}
                {error && (
                    <div className="error">
                        <strong>Error:</strong> {error}
                    </div>
                )}

                {/* Result display - updates continuously */}
                {result && isScanning && (
                    <div className="result">
                        <div className="result-header">🔄 Continuous Prediction (Embeddings)</div>
                        <div className="result-content">
                            <div className="card-id">{result.cardId}</div>
                            <div className="card-name">{result.cardName}</div>
                            <div className={"card-meta" + (result.confidence > 90 ? " card-meta--high-confidence" : "")}>
                                <span>Similarity: {result.confidence.toFixed(1)}%</span>
                            </div>

                            {result.debugImage && (
                                <div style={{ marginTop: "1rem" }}>
                                    <div
                                        style={{
                                            fontSize: "0.9rem",
                                            color: "#808080",
                                            marginBottom: "0.5rem",
                                        }}>
                                        Image sent to MobileNet (320×440, center-cropped):
                                    </div>
                                    <img
                                        src={result.debugImage}
                                        alt="Debug view"
                                        style={{
                                            border: "1px solid #e0e0e0",
                                            borderRadius: "4px",
                                            maxWidth: "200px",
                                            imageRendering: "auto",
                                        }}
                                    />
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {isScanning && !result && (
                    <div className="scanning-indicator">
                        <div className="spinner"></div>
                        <p>Scanning for cards...</p>
                    </div>
                )}
            </main>
        </div>
    );
}

export default App;
