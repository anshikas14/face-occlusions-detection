# AccessoryAI

AccessoryAI is a real-time, on-device vision application that detects facial accessories using your device's camera. Built with React and powered by **Transformers.js**, it runs completely in the browser without any backend servers, ensuring your data never leaves your device.

## ✨ Features

- **Real-Time Detection:** Instantly analyzes camera frames to detect various accessories.
- **100% On-Device:** Uses WebAssembly (WASM) and Transformers.js to run machine learning models entirely in the browser. Zero API calls, zero latency issues, and maximum privacy.
- **Zero-Shot Classification:** Powered by OpenAI's `clip-vit-base-patch32` model, classifying images into predefined categories without requiring custom training.
- **Dynamic UI:** Features a sleek, responsive, and accessible interface with live progress bars, confidence scores, and loading states.

## 🎯 What it Detects

The application currently scans for:
- 👓 Glasses (Eyeglasses / Sunglasses)
- 😷 Face Masks (Medical / Face Mask)
- 🎩 Hats (Hat / Cap)
- 🎧 Headphones (Headphones / Earphones)
- 🧣 Scarves (Scarf / Neck wrap)
- 🧥 Hoodies (Hoodie / Jacket)
- 💎 Earrings (Earrings / Jewelry)
- 🧢 Beanies (Beanie / Winter hat)

## 🛠️ Tech Stack

- **Framework:** React + Vite
- **Machine Learning:** [Transformers.js](https://huggingface.co/docs/transformers.js) (`@xenova/transformers`)
- **Vision Model:** `Xenova/clip-vit-base-patch32` (Zero-Shot Image Classification)
- **Styling:** Custom CSS with a focus on modern web design aesthetics

## 🚀 Getting Started

### Prerequisites

- Node.js (v18 or higher recommended)
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd face-occlusions-detection
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to the URL provided in the terminal (usually `http://localhost:5173`).

> **Note:** The application requires camera permissions to function. The first time you run the application and start detection, it will download the vision model (~30MB) and cache it in your browser for faster subsequent loads.

## 🔒 Privacy & Security

This application prioritizes user privacy. **No images or video frames are ever uploaded to a server.** 
- The camera stream is processed locally on your device.
- The machine learning model runs directly within your browser.
- Once the initial model is downloaded, the application can even function completely offline.

