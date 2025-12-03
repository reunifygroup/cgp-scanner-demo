import { useState, useRef, useEffect } from 'react';
import './App.css';

interface ScanResult {
  matched: boolean;
  card?: {
    cardId: string;
    cardName: string;
    setId: string;
    distance: number;
    confidence: number;
  };
  message?: string;
}

function App() {
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);

  // 📸 Start camera and scanning
  const startScanning = async () => {
    try {
      setError(null);

      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsScanning(true);

        // Start capturing frames every 500ms
        intervalRef.current = window.setInterval(() => {
          captureAndScan();
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

  // 📷 Capture frame and send to API
  const captureAndScan = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context || video.readyState !== video.HAVE_ENOUGH_DATA) return;

    // Calculate crop area (matching frame guide: 60% width, 5:7 ratio)
    const cropWidth = video.videoWidth * 0.6;
    const cropHeight = cropWidth * 1.4; // 5:7 ratio = 1.4
    const cropX = (video.videoWidth - cropWidth) / 2;
    const cropY = (video.videoHeight - cropHeight) / 2;

    // Set canvas to crop size
    canvas.width = cropWidth;
    canvas.height = cropHeight;

    // Draw only the cropped area
    context.drawImage(
      video,
      cropX, cropY, cropWidth, cropHeight,  // source crop
      0, 0, cropWidth, cropHeight            // destination
    );

    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
      if (!blob) return;

      try {
        // Send to API
        const formData = new FormData();
        formData.append('file', blob, 'scan.png');

        const response = await fetch('http://localhost:3000/api/scan', {
          method: 'POST',
          body: formData
        });

        const data: ScanResult = await response.json();

        // Only update if we found a match
        if (data.matched) {
          setResult(data);
          setError(null);
        }
      } catch (err) {
        console.error('Scan error:', err);
        // Don't show errors for failed scans, just keep trying
      }
    }, 'image/png');
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
        <h1>🎴 Pokémon Card Scanner</h1>
        <p>Point your camera at a Pokémon card to identify it</p>
      </header>

      <main>
        <div className="scanner-container">
          {/* Video stream with frame overlay */}
          <div className="video-wrapper">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={isScanning ? 'active' : 'hidden'}
            />
            {isScanning && (
              <div className="frame-guide">
                <div className="frame-corner tl"></div>
                <div className="frame-corner tr"></div>
                <div className="frame-corner bl"></div>
                <div className="frame-corner br"></div>
                <p className="frame-text">Position card here</p>
              </div>
            )}
          </div>

          {/* Hidden canvas for frame capture */}
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {/* Control button */}
          {!isScanning ? (
            <button onClick={startScanning} className="btn-primary">
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
        {result?.matched && result.card && (
          <div className="result">
            <div className="result-header">✅ Card Detected!</div>
            <div className="result-content">
              <div className="card-id">{result.card.cardId}</div>
              <div className="card-name">{result.card.cardName}</div>
              <div className="card-meta">
                <span>Set: {result.card.setId}</span>
                <span>Confidence: {result.card.confidence.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}

        {isScanning && !result?.matched && (
          <div className="scanning-indicator">
            <div className="spinner"></div>
            <p>Scanning for cards...</p>
          </div>
        )}
      </main>

      <footer>
        <p>TCGdex Scanner • sv09 & sv10 sets loaded</p>
      </footer>
    </div>
  );
}

export default App;
