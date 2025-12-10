import { useState, useRef, useEffect } from "react";
import "./App.css";

interface ScanResult {
    cardId: string;
    cardName: string;
    similarity: number; // cosine similarity in [0, 100]
}

interface Hit {
    cardId: string;
    similarity: number;
}

function App() {
    const [isScanning, setIsScanning] = useState(false);
    const [validHit, setValidHit] = useState<ScanResult | null>(null);
    const [hitHistory, setHitHistory] = useState<Hit[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [isReady, setIsReady] = useState(true);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const intervalRef = useRef<number | null>(null);

    // 📸 Start camera and scanning
    const startScanning = async () => {
        try {
            setError(null);
            setValidHit(null); // Clear previous valid hit
            setHitHistory([]); // Clear hit history

            // Request camera access
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

                // Start capturing frames every 1000ms
                intervalRef.current = window.setInterval(() => {
                    captureAndPredict();
                }, 1000);
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

    // 🔄 Scan again - restart everything
    const scanAgain = () => {
        setValidHit(null);
        setHitHistory([]);
        startScanning();
    };

    // ✅ Check if we have a valid hit (3 consecutive hits on same card with >65% similarity)
    const checkForValidHit = (newHitHistory: Hit[]) => {
        if (newHitHistory.length < 3) {
            return null;
        }

        // Get last 3 hits
        const lastThree = newHitHistory.slice(-3);

        // Check if all 3 are the same card
        const firstCardId = lastThree[0].cardId;
        const allSameCard = lastThree.every((hit) => hit.cardId === firstCardId);

        if (!allSameCard) {
            return null;
        }

        // Check if all have >65% similarity
        const allAboveThreshold = lastThree.every((hit) => hit.similarity > 65);

        if (!allAboveThreshold) {
            return null;
        }

        // Valid hit! Return the result
        const cardName = firstCardId.split("_").slice(1).join(" ");
        const avgSimilarity = lastThree.reduce((sum, hit) => sum + hit.similarity, 0) / 3;

        return {
            cardId: firstCardId,
            cardName,
            similarity: avgSimilarity,
        };
    };

    // 🎯 Capture frame and send to API for identification
    const captureAndPredict = async () => {
        if (!videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");

        if (!context || video.readyState !== video.HAVE_ENOUGH_DATA) return;

        try {
            // MobileNet expects 224×224 input
            const TARGET_SIZE = 224;

            // Set canvas to model input size
            canvas.width = TARGET_SIZE;
            canvas.height = TARGET_SIZE;

            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            if (!videoWidth || !videoHeight) {
                console.warn("⚠️ videoWidth/videoHeight is zero — metadata not ready");
                return;
            }

            // Crop to card aspect ratio first (~0.72:1 like training data 320×440)
            const CARD_ASPECT = 320 / 440; // ~0.727
            const videoAspect = videoWidth / videoHeight;

            let sx = 0;
            let sy = 0;
            let sWidth = videoWidth;
            let sHeight = videoHeight;

            if (videoAspect > CARD_ASPECT) {
                // Video is wider → crop left/right
                sHeight = videoHeight;
                sWidth = sHeight * CARD_ASPECT;
                sx = (videoWidth - sWidth) / 2;
                sy = 0;
            } else {
                // Video is taller → crop top/bottom
                sWidth = videoWidth;
                sHeight = sWidth / CARD_ASPECT;
                sx = 0;
                sy = (videoHeight - sHeight) / 2;
            }

            // Draw cropped card region and resize to 224×224
            context.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, TARGET_SIZE, TARGET_SIZE);

            // Convert canvas to base64 image
            const imageData = canvas.toDataURL("image/jpeg", 0.8);

            // Send to API for identification
            const response = await fetch("/api/identify-card", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ image: imageData }),
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const result: ScanResult = await response.json();

            console.log("🔍 Similarity search:", {
                cardId: result.cardId,
                similarity: result.similarity.toFixed(1) + "%",
            });

            // Add to hit history
            const newHit: Hit = {
                cardId: result.cardId,
                similarity: result.similarity,
            };

            setHitHistory((prev) => {
                const updated = [...prev, newHit];

                // Check for valid hit
                const validResult = checkForValidHit(updated);
                if (validResult) {
                    console.log("✅ VALID HIT DETECTED:", validResult);
                    setValidHit(validResult);
                    stopScanning(); // Stop scanning when valid hit is detected
                }

                // Keep only last 10 hits to avoid memory issues
                return updated.slice(-10);
            });
        } catch (err) {
            console.error("❌ Prediction error:", err);
        }
    };

    // 🧹 Cleanup on unmount
    useEffect(() => {
        return () => {
            stopScanning();
        };
    }, []);

    return (
        <div className="app">
            <header>
                <h1>CGP Card Scanner</h1>
                <p>AI-powered instant card recognition (Server-side processing)</p>
            </header>

            <main>
                {/* Error display */}
                {error && (
                    <div className="error">
                        <strong>Error:</strong> {error}
                    </div>
                )}

                {/* Valid hit result - only shown after 3 consecutive matches */}
                {validHit && (
                    <div className="result result--confirmed">
                        <div className="result-header">✅ Card Identified!</div>
                        <div className="result-content">
                            <div className="card-id">{validHit.cardId}</div>
                            <div className="card-name">{validHit.cardName}</div>
                            <div className="card-meta card-meta--high-confidence">
                                <span>Confidence: {validHit.similarity.toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Scanner container - only show when actively scanning */}
                <div className="scanner-container" style={{ display: isScanning ? "flex" : "none" }}>
                    {/* Video stream */}
                    <div className="video-wrapper">
                        <video ref={videoRef} autoPlay playsInline muted className="active" />
                    </div>

                    {/* Hidden canvas for frame capture */}
                    <canvas ref={canvasRef} style={{ display: "none" }} />
                </div>

                {/* Control buttons */}
                {!validHit && !isScanning && (
                    <button onClick={startScanning} className="btn-primary" disabled={!isReady}>
                        Start Scanner
                    </button>
                )}

                {!validHit && isScanning && (
                    <button onClick={stopScanning} className="btn-secondary">
                        Stop Scanner
                    </button>
                )}

                {validHit && (
                    <button onClick={scanAgain} className="btn-primary">
                        Scan Again
                    </button>
                )}

                {/* Scanning indicator - only shown while actively scanning */}
                {isScanning && !validHit && (
                    <div className="scanning-indicator">
                        <div className="spinner"></div>
                        <p>Scanning for cards...</p>
                        <div style={{ fontSize: "0.9rem", color: "#808080", marginTop: "0.5rem" }}>
                            {hitHistory.length > 0 && (
                                <>
                                    Hits: {hitHistory.slice(-3).length}/3
                                    {hitHistory.length >= 3 && <span style={{ marginLeft: "1rem" }}>{hitHistory.slice(-3).every((h) => h.cardId === hitHistory[hitHistory.length - 1].cardId) ? "🎯 Same card detected" : "🔄 Different cards"}</span>}
                                </>
                            )}
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}

export default App;
