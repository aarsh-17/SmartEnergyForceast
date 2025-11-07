import React, { useState } from "react";
import ForecastChart from "./ForecastChart";
import "./App.css";
function App() {
  const [forecast, setForecast] = useState([]);
  const [file, setFile] = useState(null);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    console.log(data);
    
    setForecast(data.forecast);
  };

  return (
    <div className="App">
      <h2>Smart Energy Consumption Forecast</h2>
      <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])}/>
      <button onClick={handleUpload}>Predict</button>
      <ForecastChart forecast={forecast} />
    </div>
  );
}

export default App;
