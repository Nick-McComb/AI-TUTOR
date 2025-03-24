# RAG Chat React Application

This is a React version of the RAG (Retrieval-Augmented Generation) Chat application.

## Features

- Real-time chat with RAG capabilities via WebSocket
- Upload and manage documents (PDF, DOCX, PPTX, TXT)
- Upload and process videos with transcription
- Process YouTube videos with automatic downloading and transcription
- Browse document transcripts and interact with specific segments

## Prerequisites

- Node.js (v14.0.0 or later)
- npm (v6.0.0 or later)
- Backend server running at http://localhost:8000

## Installation

1. Install dependencies:

```bash
npm install
```

2. Start the development server:

```bash
npm start
```

The application will be available at http://localhost:3000.

## Usage

1. Ensure the backend server is running at http://localhost:8000
2. Open the application in your browser
3. Use the tabs to navigate between Chat, Documents, and Upload functionalities
4. Upload documents or videos to create a knowledge base
5. Chat with the AI that has access to the information from your uploaded content

## Backend API

This React frontend communicates with the following backend API endpoints:

- WebSocket connection: `ws://{hostname}:8000/ws/chat`
- Document listing: `GET /api/documents`
- Document upload: `POST /api/documents/upload`
- Video upload: `POST /api/videos/upload`
- YouTube processing: `POST /api/youtube`

## Development

- `npm start`: Runs the app in development mode
- `npm build`: Builds the app for production
- `npm test`: Runs tests
- `npm run eject`: Ejects from Create React App configuration 