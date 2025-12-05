import { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

interface ScanResult {
  cardId: string;
  cardName: string;
  confidence: number;
}

function App() {
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<string>('Loading model...');
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const modelRef = useRef<tf.GraphModel | null>(null);
  const classNamesRef = useRef<string[]>([]);

  // 🧠 Load TensorFlow.js model on mount
  async function loadModel() {
    try {
      setModelStatus('Loading model...');

      // Load class names
      const classNamesResponse = await fetch('/model/class_names.json');
      classNamesRef.current = await classNamesResponse.json();

      // Load Graph Model (Keras 3.x compatible)
      const model = await tf.loadGraphModel('/model/model.json');
      modelRef.current = model;

      // Warm up the model
      const dummyInput = tf.zeros([1, 224, 224, 3]);
      model.predict(dummyInput);
      dummyInput.dispose();

      setModelStatus(`Model loaded! ${classNamesRef.current.length} cards ready`);
      setIsModelLoaded(true);
      console.log('✅ Model loaded:', classNamesRef.current);

    } catch (err) {
      setError('Failed to load model: ' + (err as Error).message);
      setModelStatus('Model load failed');
      setIsModelLoaded(false);
    }
  }

  useEffect(() => {
    loadModel();
  }, []);

  // 📸 Start camera and scanning
  const startScanning = async () => {
    if (!modelRef.current) {
      setError('Model not loaded yet. Please wait...');
      return;
    }

    try {
      setError(null);

      // Request camera access with card-like aspect ratio (portrait)
      // Card ratio: 63mm × 88mm ≈ 0.716:1
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 720 },   // Portrait mode
          height: { ideal: 1000 },  // Card-like ratio
          aspectRatio: { ideal: 0.72 }
        }
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
      setError('Failed to access camera: ' + (err as Error).message);
    }
  };

  // 🛑 Stop camera and scanning
  const stopScanning = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsScanning(false);
    setResult(null);
  };

  // 🎯 Capture frame and run inference
  const captureAndPredict = async () => {
    if (!videoRef.current || !canvasRef.current || !modelRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context || video.readyState !== video.HAVE_ENOUGH_DATA) return;

    try {
      // Set canvas size to model input size
      canvas.width = 224;
      canvas.height = 224;

      // Crop center of video to focus on card area (zoom effect)
      // This matches training data where cards fill the frame
      const videoAspect = video.videoWidth / video.videoHeight;
      const targetAspect = 1; // Square for 224x224

      let sx = 0, sy = 0, sWidth = video.videoWidth, sHeight = video.videoHeight;

      if (videoAspect > targetAspect) {
        // Video is wider - crop sides
        sWidth = video.videoHeight * targetAspect;
        sx = (video.videoWidth - sWidth) / 2;
      } else {
        // Video is taller - crop top/bottom
        sHeight = video.videoWidth / targetAspect;
        sy = (video.videoHeight - sHeight) / 2;
      }

      // Draw cropped and scaled video frame
      context.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, 224, 224);

      // Get image data and convert to tensor
      const imageData = context.getImageData(0, 0, 224, 224);

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
      console.log('Predictions:', {
        confidence: (confidence * 100).toFixed(1) + '%',
        card: classNamesRef.current[maxIndex],
        allConfidences: Array.from(predArray).map((p, i) => ({
          card: classNamesRef.current[i],
          conf: (p * 100).toFixed(1) + '%'
        }))
      });

      // Clean up tensors
      tensor.dispose();
      predictions.dispose();

      // Show result if confidence is high enough
      // Lower threshold (50%) to catch cards at various positions
      if (confidence > 0.50) {
        const cardId = classNamesRef.current[maxIndex];
        const cardName = cardId.split('_').slice(1).join(' ');

        setResult({
          cardId,
          cardName,
          confidence: confidence * 100
        });
      } else {
        // Clear result if confidence drops
        setResult(null);
      }

    } catch (err) {
      console.error('Prediction error:', err);
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
        <h1>🎴 Pokémon Card Scanner</h1>
        <p>AI-powered instant card recognition</p>
        <div className="model-status">{modelStatus}</div>
      </header>

      <main>
        <div className="scanner-container">
          {/* Video stream - full freedom, no guides */}
          <div className="video-wrapper">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={isScanning ? 'active' : 'hidden'}
            />
          </div>

          {/* Hidden canvas for frame capture */}
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {/* Control button */}
          {!isScanning ? (
            <button
              onClick={startScanning}
              className="btn-primary"
              disabled={!isModelLoaded}
            >
              📸 Start Scanner
            </button>
          ) : (
            <button onClick={stopScanning} className="btn-secondary">
              ⏹️ Stop Scanner
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
            <div className="result-header">✅ Card Detected!</div>
            <div className="result-content">
              <div className="card-id">{result.cardId}</div>
              <div className="card-name">{result.cardName}</div>
              <div className="card-meta">
                <span>Confidence: {result.confidence.toFixed(1)}%</span>
              </div>
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

      <footer>
        <p>Powered by TensorFlow.js • 10 cards trained</p>
      </footer>
    </div>
  );
}

export default App;
