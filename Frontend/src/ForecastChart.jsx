import { Line } from "react-chartjs-2";

import { useEffect, useRef } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function ForecastChart({ forecast }) {
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);

  useEffect(() => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    const ctx = chartRef.current.getContext("2d");
    chartInstanceRef.current = new ChartJS(ctx, {
      type: "line",
      data: {
        labels: forecast.map(f => f.ds),
        datasets: [
          {
            label: "Forecast (kW)",
            data: forecast.map(f => f.yhat),
            borderColor: "royalblue",
            tension: 0.3
          }
        ]
      },
      options: { responsive: true }
    });

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
    };
  }, [forecast]);

  return <canvas ref={chartRef} />;
}

export default ForecastChart;
