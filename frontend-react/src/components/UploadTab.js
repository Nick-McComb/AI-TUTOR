import React, { useState, useRef } from 'react';

const UploadTab = ({ addSystemMessage, refreshDocuments, apiBaseUrl }) => {
  const [uploadType, setUploadType] = useState('document');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [documentFile, setDocumentFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  
  const documentFileInputRef = useRef(null);
  const videoFileInputRef = useRef(null);

  const uploadDocument = async () => {
    if (!documentFile) return;
    
    const formData = new FormData();
    formData.append('file', documentFile);
    
    try {
      addSystemMessage(`Uploading document: ${documentFile.name}`);
      
      const response = await fetch(`${apiBaseUrl}/api/documents/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const data = await response.json();
        addSystemMessage(`Document uploaded successfully. Processing in background.`);
        
        // Reset file input
        setDocumentFile(null);
        if (documentFileInputRef.current) {
          documentFileInputRef.current.value = '';
        }
        
        // Refresh document list
        refreshDocuments();
      } else {
        const errorData = await response.json();
        addSystemMessage(`Error: ${errorData.detail}`);
      }
    } catch (error) {
      console.error('Error uploading document:', error);
      addSystemMessage(`Error: ${error.message}`);
    }
  };
  
  const uploadVideo = async () => {
    if (!videoFile) return;
    
    const formData = new FormData();
    formData.append('file', videoFile);
    
    try {
      addSystemMessage(`Uploading video: ${videoFile.name}`);
      
      const response = await fetch(`${apiBaseUrl}/api/videos/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        await response.json();
        addSystemMessage(`Video uploaded successfully! Processing and transcription will begin shortly.`);
        refreshDocuments();
      } else {
        const error = await response.json();
        addSystemMessage(`Error uploading video: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error uploading video:', error);
      addSystemMessage(`Error uploading video: ${error.message}`);
    }
    
    // Clear file input
    setVideoFile(null);
    if (videoFileInputRef.current) {
      videoFileInputRef.current.value = '';
    }
  };
  
  const processYouTube = async () => {
    if (!youtubeUrl) return;
    
    try {
      addSystemMessage(`Processing YouTube URL: ${youtubeUrl}`);
      
      const response = await fetch(`${apiBaseUrl}/api/youtube`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          url: youtubeUrl
        })
      });
      
      if (response.ok) {
        await response.json();
        addSystemMessage(`YouTube video processing initiated! Downloading and transcription will begin shortly.`);
        refreshDocuments();
      } else {
        const error = await response.json();
        addSystemMessage(`Error processing YouTube URL: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error processing YouTube URL:', error);
      addSystemMessage(`Error processing YouTube URL: ${error.message}`);
    }
    
    // Clear YouTube URL input
    setYoutubeUrl('');
  };

  return (
    <div className="mb-4">
      <h3>Upload Content</h3>
      <div className="btn-group mb-3">
        <button 
          className={`btn ${uploadType === 'document' ? 'btn-primary' : 'btn-outline-primary'}`}
          onClick={() => setUploadType('document')}
        >
          Document
        </button>
        <button 
          className={`btn ${uploadType === 'video' ? 'btn-primary' : 'btn-outline-primary'}`}
          onClick={() => setUploadType('video')}
        >
          Video
        </button>
        <button 
          className={`btn ${uploadType === 'youtube' ? 'btn-primary' : 'btn-outline-primary'}`}
          onClick={() => setUploadType('youtube')}
        >
          YouTube
        </button>
      </div>
      
      {/* Document Upload */}
      {uploadType === 'document' && (
        <div className="card">
          <div className="card-body">
            <h5 className="card-title">Upload Document</h5>
            <p className="card-text">Upload PDF, DOCX, PPTX, or TXT files to use as knowledge base for the chat.</p>
            <div className="mb-3">
              <label htmlFor="documentFileInput" className="form-label">Select Document</label>
              <input 
                className="form-control" 
                type="file" 
                id="documentFileInput" 
                ref={documentFileInputRef}
                onChange={(e) => setDocumentFile(e.target.files[0])} 
                accept=".pdf,.docx,.pptx,.txt"
              />
            </div>
            <button 
              className="btn btn-primary" 
              onClick={uploadDocument} 
              disabled={!documentFile}
            >
              Upload Document
            </button>
          </div>
        </div>
      )}
      
      {/* Video Upload */}
      {uploadType === 'video' && (
        <div className="card">
          <div className="card-body">
            <h5 className="card-title">Upload Video</h5>
            <p className="card-text">Upload a video file to be transcribed and used as knowledge base for the chat.</p>
            <div className="mb-3">
              <label htmlFor="videoFileInput" className="form-label">Select Video</label>
              <input 
                className="form-control" 
                type="file" 
                id="videoFileInput" 
                ref={videoFileInputRef}
                onChange={(e) => setVideoFile(e.target.files[0])} 
                accept="video/*"
              />
            </div>
            <button 
              className="btn btn-primary" 
              onClick={uploadVideo} 
              disabled={!videoFile}
            >
              Upload Video
            </button>
          </div>
        </div>
      )}
      
      {/* YouTube URL */}
      {uploadType === 'youtube' && (
        <div className="card">
          <div className="card-body">
            <h5 className="card-title">Process YouTube Video</h5>
            <p className="card-text">Enter a YouTube URL to download, transcribe, and use as knowledge base for the chat.</p>
            <div className="mb-3">
              <label htmlFor="youtubeUrlInput" className="form-label">YouTube URL</label>
              <input 
                type="text" 
                className="form-control" 
                id="youtubeUrlInput" 
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)} 
                placeholder="https://www.youtube.com/watch?v=..."
              />
            </div>
            <button 
              className="btn btn-primary" 
              onClick={processYouTube} 
              disabled={!youtubeUrl}
            >
              Process YouTube Video
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadTab; 