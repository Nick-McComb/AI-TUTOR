import React from 'react';

const DocumentsTab = ({ documents, processingCount, jumpToTranscriptSegment, deleteDocument }) => {
  const formatDate = (isoString) => {
    return new Date(isoString).toLocaleString();
  };

  const formatDuration = (seconds) => {
    if (!seconds) return '';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  return (
    <div>
      <h3>Your Documents</h3>
      
      {processingCount > 0 && (
        <div className="alert alert-info">
          {processingCount} document(s) currently processing. Status will update automatically.
        </div>
      )}
      
      {documents.length === 0 && (
        <div className="alert alert-warning">
          No documents uploaded yet. Go to the Upload tab to add documents.
        </div>
      )}
      
      <div className="row">
        {documents.map(doc => (
          <div key={doc.id} className="col-md-6 mb-3">
            <div className="card document-card">
              <div className="card-header d-flex justify-content-between align-items-center">
                <div>{doc.filename}</div>
                <div className="d-flex align-items-center">
                  <span className={`document-status status-${doc.status} me-2`}>{doc.status}</span>
                  <button 
                    className="btn btn-sm btn-danger" 
                    onClick={() => deleteDocument(doc.id)}
                    disabled={['processing', 'downloading', 'transcribing'].includes(doc.status)}
                    title={['processing', 'downloading', 'transcribing'].includes(doc.status) ? 
                          "Cannot delete while processing" : "Delete document"}
                  >
                    <i className="bi bi-trash"></i>
                  </button>
                </div>
              </div>
              <div className="card-body">
                <div><strong>Type:</strong> {doc.document_type}</div>
                <div><strong>Uploaded:</strong> {formatDate(doc.upload_time)}</div>
                {doc.duration && <div><strong>Duration:</strong> {formatDuration(doc.duration)}</div>}
                
                {/* YouTube thumbnail if available */}
                {doc.youtube_thumbnail && (
                  <div className="mt-2">
                    <img src={doc.youtube_thumbnail} className="video-thumbnail" alt="Video thumbnail" />
                    <div className="mt-1"><strong>Title:</strong> {doc.youtube_title}</div>
                  </div>
                )}
                
                {/* Transcript segments if available */}
                {doc.segments && doc.segments.length > 0 && (
                  <div className="mt-3">
                    <h6>Transcript Segments:</h6>
                    <div className="transcript-container" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                      {doc.segments.map((segment, idx) => (
                        <div 
                          key={idx} 
                          className="transcript-segment py-1 cursor-pointer" 
                          onClick={() => jumpToTranscriptSegment(segment)}
                        >
                          <span className="timestamp">[{formatDuration(segment.start)}]</span>
                          <span>{segment.text}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DocumentsTab; 