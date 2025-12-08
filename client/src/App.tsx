import { useState, useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "./App.css";

interface ScanResult {
    cardId: string;
    cardName: string;
    confidence: number;
    debugImage?: string; // Base64 image data for debugging
}

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
    const modelRef = useRef<tf.GraphModel | null>(null);
    const classNamesRef = useRef<string[]>([]);

    // 🧠 Load TensorFlow.js model on mount
    useEffect(() => {
        async function loadModel() {
            try {
                setModelStatus("Loading model...");

                // Load class names
                const classNamesResponse = await fetch("/model/class_names.json");
                classNamesRef.current = await classNamesResponse.json();

                // Load classification model
                const model = await tf.loadGraphModel("/model/model.json");
                modelRef.current = model;

                // Warm up model (portrait: height=440, width=320)
                const dummyInput = tf.zeros([1, 440, 320, 3]); // [batch, height, width, channels]
                model.predict(dummyInput);
                dummyInput.dispose();

                setModelStatus(`Model loaded! ${classNamesRef.current.length} cards ready`);
                setIsModelLoaded(true);
                console.log("✅ Model loaded:", classNamesRef.current);
            } catch (err) {
                setError("Failed to load model: " + (err as Error).message);
                setModelStatus("Model load failed");
                setIsModelLoaded(false);
            }
        }

        loadModel();
    }, []);

    // 📸 Start camera and scanning
    const startScanning = async () => {
        if (!modelRef.current) {
            setError("Model not loaded yet. Please wait...");
            return;
        }

        try {
            setError(null);

            // Request camera access with card-like aspect ratio (portrait)
            // Card ratio: 63mm × 88mm ≈ 0.716:1
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "environment",
                    width: { ideal: 720 }, // Portrait mode
                    height: { ideal: 1000 }, // Card-like ratio
                    aspectRatio: { ideal: 0.72 },
                },
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                streamRef.current = stream;
                setIsScanning(true);

                // Start capturing frames every 500ms
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
        if (intervalRef.current) {
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


    // 🎯 Capture frame and run inference
    const captureAndPredict = async () => {
        if (!videoRef.current || !canvasRef.current || !modelRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");

        if (!context || video.readyState !== video.HAVE_ENOUGH_DATA) return;

        try {
            // Set canvas size to model input size (portrait card)
            canvas.width = 320;
            canvas.height = 440;

            // Resize full camera frame to 320×440
            context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 320, 440);

            // Get image data and convert to tensor
            const imageData = context.getImageData(0, 0, 320, 440);

            // Convert to tensor and normalize
            const tensor = tf.tidy(() => {
                const imageTensor = tf.browser.fromPixels(imageData);
                const normalized = imageTensor.div(255.0);
                const batched = normalized.expandDims(0);
                return batched;
            });

            // Run classification
            const predictions = modelRef.current.predict(tensor) as tf.Tensor;
            const predArray = await predictions.data();

            // Get top prediction
            const maxIndex = predArray.indexOf(Math.max(...Array.from(predArray)));
            const confidence = predArray[maxIndex];
            const cardId = classNamesRef.current[maxIndex];

            // Debug: Show all predictions
            console.log("Predictions:", {
                cardId,
                confidence: (confidence * 100).toFixed(1) + "%",
                tensors: tf.memory().numTensors,
                allConfidences: Array.from(predArray)
                    .map((p, i) => ({
                        card: classNamesRef.current[i],
                        conf: (p * 100).toFixed(1) + "%",
                    }))
                    .sort((a, b) => parseFloat(b.conf) - parseFloat(a.conf)),
            });

            // Clean up tensors
            tensor.dispose();
            predictions.dispose();

            // Show result continuously (no threshold) for debugging
            const cardName = cardId.split("_").slice(1).join(" ");
            const debugImage = canvas.toDataURL("image/png");

            setResult({
                cardId,
                cardName,
                confidence: confidence * 100,
                debugImage,
            });

            // Keep scanning - don't stop (for debugging)
        } catch (err) {
            console.error("Prediction error:", err);
        }
    };

    // 🧹 Cleanup on unmount
    useEffect(() => {
        return () => {
            stopScanning();
            if (modelRef.current) {
                modelRef.current.dispose();
            }
        };
    }, []);

    return (
        <div className="app">
            <header>
                <h1>CGP Card Scanner</h1>
                <p>AI-powered instant card recognition</p>
                <div className="model-status">{modelStatus}</div>
            </header>

            <main>
                <div className="scanner-container">
                    {/* Video stream - full freedom, no guides */}
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
                        <div className="result-header">🔄 Continuous Prediction (Live)</div>
                        <div className="result-content">
                            <div className="card-id">{result.cardId}</div>
                            <div className="card-name">{result.cardName}</div>
                            <div className="card-meta">
                                <span>Confidence: {result.confidence.toFixed(1)}%</span>
                            </div>

                            {/* Debug: Show the actual image sent to AI */}
                            {result.debugImage && (
                                <div style={{ marginTop: "1rem" }}>
                                    <div style={{ fontSize: "0.9rem", color: "#808080", marginBottom: "0.5rem" }}>Image sent to AI (320×440):</div>
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
