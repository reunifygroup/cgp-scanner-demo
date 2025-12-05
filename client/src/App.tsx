import { useState, useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "./App.css";

interface ScanResult {
    cardId: string;
    cardName: string;
    confidence: number;
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
    const firstCardIdRef = useRef<string | null>(null);

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

    // 🔄 Clear result and restart scanning
    const scanAgain = () => {
        setResult(null);
        firstCardIdRef.current = null; // Reset first card tracker
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

            // Crop card-shaped region from video center (portrait orientation)
            // This matches training data: card images resized to portrait 312×224 (height×width)
            const cardAspect = 224 / 312; // Width/height ratio for portrait cards (≈0.718)
            const videoAspect = video.videoWidth / video.videoHeight;

            let sx = 0,
                sy = 0,
                sWidth = video.videoWidth,
                sHeight = video.videoHeight;

            if (videoAspect > cardAspect) {
                // Video is wider - crop sides to get card aspect
                sHeight = video.videoHeight;
                sWidth = sHeight * cardAspect;
                sx = (video.videoWidth - sWidth) / 2;
                sy = 0;
            } else {
                // Video is taller - crop top/bottom to get card aspect
                sWidth = video.videoWidth;
                sHeight = sWidth / cardAspect;
                sx = 0;
                sy = (video.videoHeight - sHeight) / 2;
            }

            // Draw portrait card crop, resized to 224×312 (width×height)
            context.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, 224, 312);

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

            // Debug: Show all predictions
            console.log("Predictions:", {
                confidence: (confidence * 100).toFixed(1) + "%",
                card: classNamesRef.current[maxIndex],
                allConfidences: Array.from(predArray).map((p, i) => ({
                    card: classNamesRef.current[i],
                    conf: (p * 100).toFixed(1) + "%",
                })),
            });

            // Clean up tensors
            tensor.dispose();
            predictions.dispose();

            // Show result if confidence is high enough
            // Lower threshold (50%) to catch cards at various positions
            if (confidence > 0.5) {
                const cardId = classNamesRef.current[maxIndex];

                // If this is the first detection, just store it (likely wrong)
                if (firstCardIdRef.current === null) {
                    firstCardIdRef.current = cardId;
                    console.log("First card detected (ignoring):", cardId);
                    return;
                }

                // If we detect a DIFFERENT card, this is likely the real one
                if (cardId !== firstCardIdRef.current) {
                    const cardName = cardId.split("_").slice(1).join(" ");

                    setResult({
                        cardId,
                        cardName,
                        confidence: confidence * 100,
                    });

                    console.log("New card detected! Locking on:", cardId);

                    // Stop scanning after detecting different card
                    stopScanning();
                }
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
