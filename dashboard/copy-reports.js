import fs from "fs";
import path from "path";

const source = path.resolve(process.cwd(), "..", "reports");
const dest = path.resolve(process.cwd(), "public", "reports");

function copyIfExists(file) {
  const from = path.join(source, file);
  const to = path.join(dest, file);
  if (fs.existsSync(from)) {
    fs.copyFileSync(from, to);
    console.log(`Copied ${file}`);
  } else {
    console.warn(`Missing ${from}; skipping`);
  }
}

fs.mkdirSync(dest, { recursive: true });
["evaluation_metrics.json", "predictions.csv", "forecast_dec_2025.csv"].forEach(copyIfExists);

