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

                // Load Graph Model (Keras 3.x compatible)
                const model = await tf.loadGraphModel("/model/model.json");
                modelRef.current = model;

                // Warm up the model (portrait: height=312, width=224)
                const dummyInput = tf.zeros([1, 312, 224, 3]); // [batch, height, width, channels]
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

    // 🔄 Clear result and restart scanning with full reset
    const scanAgain = async () => {
        // Clear result
        setResult(null);

        // Stop and cleanup current scan completely
        stopScanning();

        // Clear canvas to remove any residual state
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            if (ctx) {
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            }
        }

        // Force TensorFlow.js to cleanup tensors
        tf.engine().startScope();
        tf.engine().endScope();

        // Small delay to ensure complete cleanup
        await new Promise((resolve) => setTimeout(resolve, 100));

        // Log tensor memory for debugging
        console.log("🔄 Scan reset - Tensors in memory:", tf.memory().numTensors);

        // Start fresh scan
        startScanning();
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
            canvas.width = 224;
            canvas.height = 312;

            // Resize full camera frame to 224×312 (no cropping)
            // Camera is already card-shaped (0.72 aspect ratio)
            context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 224, 312);

            // Get image data and convert to tensor
            const imageData = context.getImageData(0, 0, 224, 312);

            // Convert to tensor and normalize
            const tensor = tf.tidy(() => {
                // Convert to tensor
                const imageTensor = tf.browser.fromPixels(imageData);

                // Normalize to [0, 1]
                const normalized = imageTensor.div(255.0);

                // Add batch dimension
                const batched = normalized.expandDims(0);

                return batched;
            });

            // Run inference
            const predictions = modelRef.current.predict(tensor) as tf.Tensor;
            const predArray = await predictions.data();

            // Get top prediction
            const maxIndex = predArray.indexOf(Math.max(...Array.from(predArray)));
            const confidence = predArray[maxIndex];

            // Debug: Show all predictions and memory
            console.log("Predictions:", {
                confidence: (confidence * 100).toFixed(1) + "%",
                card: classNamesRef.current[maxIndex],
                tensors: tf.memory().numTensors, // Monitor tensor count
                allConfidences: Array.from(predArray)
                    .map((p, i) => ({
                        card: classNamesRef.current[i],
                        conf: (p * 100).toFixed(1) + "%",
                    }))
                    .sort((a, b) => parseFloat(b.conf) - parseFloat(a.conf)), // Show all 4 cards
            });

            // Clean up tensors immediately
            tensor.dispose();
            predictions.dispose();

            // Simple high-confidence threshold
            const CONFIDENCE_THRESHOLD = 0.7; // Lock on any card with >80% confidence

            // Lock on first high-confidence detection
            if (confidence > CONFIDENCE_THRESHOLD) {
                const cardId = classNamesRef.current[maxIndex];
                const cardName = cardId.split("_").slice(1).join(" ");

                // Capture the canvas as image for debugging
                const debugImage = canvas.toDataURL("image/png");

                setResult({
                    cardId,
                    cardName,
                    confidence: confidence * 100,
                    debugImage,
                });

                console.log("Card detected:", cardId, `${(confidence * 100).toFixed(1)}%`);
                console.log("Debug image saved - check result display");
                stopScanning();
            }
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

                {/* Result display */}
                {result && (
                    <div className="result">
                        <div className="result-header">Card Detected</div>
                        <div className="result-content">
                            <div className="card-id">{result.cardId}</div>
                            <div className="card-name">{result.cardName}</div>
                            <div className="card-meta">
                                <span>Confidence: {result.confidence.toFixed(1)}%</span>
                            </div>

                            {/* Debug: Show the actual image sent to AI */}
                            {result.debugImage && (
                                <div style={{ marginTop: "1rem" }}>
                                    <div style={{ fontSize: "0.9rem", color: "#808080", marginBottom: "0.5rem" }}>Image sent to AI (224×312):</div>
                                    <img
                                        src={result.debugImage}
                                        alt="Debug view"
                                        style={{
                                            border: "1px solid #3a3a3a",
                                            borderRadius: "4px",
                                            maxWidth: "200px",
                                            imageRendering: "pixelated",
                                        }}
                                    />
                                </div>
                            )}

                            <button onClick={scanAgain} className="btn-primary" style={{ marginTop: "1rem" }}>
                                Scan Again
                            </button>
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
