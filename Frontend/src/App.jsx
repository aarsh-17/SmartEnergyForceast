import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [imageURL, setImageURL] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a CSV file before uploading.");
      return;
    }

    setLoading(true);
    setError("");
    setImageURL(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const blob = await response.blob();
      const imageObjectURL = URL.createObjectURL(blob);
      setImageURL(imageObjectURL);
    } catch (err) {
      console.error("Error uploading file:", err);
      setError("Failed to fetch forecast. Check backend logs or file format.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="Main">
    <div className="App">
      <h2>⚡ Smart Energy Consumption Forecaster</h2>

      <div className="upload-section">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button onClick={handleUpload} disabled={loading}>
          {loading ? "Processing..." : "Generate Forecast"}
        </button>
      </div>

      {error && <p className="error-text">{error}</p>}

      {imageURL && (
        <div className="image-section">
          <h3>Hybrid Model Forecast</h3>
          <p>Note: Larger the dataset better the hybrid will be. For smaller datasets ignore the red curve</p>
          <img
            src={imageURL}
            alt="Hybrid Forecast Plot"
            className="forecast-image"
          />
        </div>
      )}
    </div>
    </div>
  );
}

export default App;
