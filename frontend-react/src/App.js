import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import ChatTab from './components/ChatTab';
import DocumentsTab from './components/DocumentsTab';
import UploadTab from './components/UploadTab';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState('');
  const [processingCount, setProcessingCount] = useState(0);
  
  const ws = useRef(null);
  const refreshIntervalRef = useRef(null);

  // Connect to WebSocket when component mounts
  useEffect(() => {
    connectWebSocket();
    loadDocuments();
    
    refreshIntervalRef.current = setInterval(() => {
      if (processingCount > 0) {
        loadDocuments();
      }
    }, 5000);
    
    // Cleanup on unmount
    return () => {
      disconnectWebSocket();
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, []); // Empty dependency array means this effect runs once on mount

  // Watch for changes in processing documents
  useEffect(() => {
    const processingDocs = documents.filter(doc => 
      ['processing', 'uploaded', 'pending', 'downloading', 'transcribing'].includes(doc.status)
    );
    setProcessingCount(processingDocs.length);
  }, [documents]);

  const connectWebSocket = () => {
    ws.current = new WebSocket(`ws://${window.location.hostname}:8000/ws/chat`);
    
    ws.current.onopen = () => {
      setIsConnected(true);
      setConnectionStatus('Connected');
      addSystemMessage('Connected to RAG Chat WebSocket');
    };
    
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === "start") {
        setIsStreaming(true);
        setCurrentStreamingMessage('');
      } else if (data.type === "stream") {
        setCurrentStreamingMessage(prev => prev + data.content);
        scrollToBottom();
      } else if (data.type === "end") {
        setCurrentStreamingMessage(prevMessage => {
          setIsStreaming(false);
          addMessage(prevMessage, 'ai');
          return '';
        });
      } else if (data.type === "system") {
        addSystemMessage(data.content);
      } else if (data.type === "document_update") {
        // Update document in the list
        setDocuments(prevDocs => {
          const updatedDocs = [...prevDocs];
          const index = updatedDocs.findIndex(doc => doc.id === data.document.id);
          if (index !== -1) {
            updatedDocs[index] = data.document;
          } else {
            updatedDocs.push(data.document);
          }
          return updatedDocs;
        });
      } else if (data.type === "document_deleted") {
        // Remove document from the list
        setDocuments(prevDocs => prevDocs.filter(doc => doc.id !== data.document_id));
        addSystemMessage(`Document ${data.document_id} was deleted`);
      }
    };
    
    ws.current.onclose = () => {
      setIsConnected(false);
      setConnectionStatus('Disconnected');
      addSystemMessage('Disconnected from RAG Chat WebSocket');
    };
    
    ws.current.onerror = () => {
      addSystemMessage('WebSocket error occurred');
    };
  };
  
  const disconnectWebSocket = () => {
    if (ws.current) {
      ws.current.close();
    }
  };
  
  const addMessage = (text, sender) => {
    setMessages(prev => [...prev, {
      text,
      sender,
      time: new Date().toLocaleTimeString()
    }]);
    scrollToBottom();
  };
  
  const addSystemMessage = (text) => {
    setMessages(prev => [...prev, {
      text,
      sender: 'system',
      time: new Date().toLocaleTimeString()
    }]);
    scrollToBottom();
  };
  
  const sendMessage = () => {
    if (!newMessage.trim() || !isConnected) return;
    
    // Add the user message to chat
    addMessage(newMessage, 'user');
    
    // Send via WebSocket
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        type: "message",
        content: newMessage
      }));
    }
    
    setNewMessage('');
  };
  
  const clearChat = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        type: "command",
        command: "clear"
      }));
      setMessages([]);
      addSystemMessage("Chat history cleared");
    }
  };
  
  const deleteDocument = async (documentId) => {
    try {
      // First attempt to delete via WebSocket
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({
          type: "command",
          command: "delete_document",
          document_id: documentId
        }));
        return;
      }
      
      // Fallback to REST API if WebSocket is not available
      const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        // Remove from local state
        setDocuments(prevDocs => prevDocs.filter(doc => doc.id !== documentId));
        addSystemMessage(`Document ${documentId} deleted successfully`);
      } else {
        const errorData = await response.json();
        addSystemMessage(`Error: ${errorData.detail || 'Failed to delete document'}`);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      addSystemMessage(`Error: ${error.message}`);
    }
  };
  
  const scrollToBottom = () => {
    setTimeout(() => {
      const container = document.querySelector('.chat-container');
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
    }, 50);
  };
  
  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents`);
      if (response.ok) {
        const data = await response.json();
        setDocuments(data);
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };
  
  const jumpToTranscriptSegment = (segment) => {
    // Add a message to reference the transcript
    const message = `Can you tell me more about this part: "${segment.text}"`;
    addMessage(message, 'user');
    
    // Send via WebSocket
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        type: "message",
        content: message
      }));
    }
    
    // Switch to chat tab
    setActiveTab('chat');
  };

  return (
    <div className="container mt-4">
      <h1 className="mb-4">RAG Chat Application</h1>
      
      <div className={`alert ${isConnected ? 'alert-success' : 'alert-danger'}`}>
        Connection Status: {connectionStatus}
      </div>
      
      <ul className="nav nav-tabs mb-4">
        <li className="nav-item">
          <a 
            className={`nav-link ${activeTab === 'chat' ? 'active' : ''}`} 
            href="#" 
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('chat');
            }}
          >
            Chat
          </a>
        </li>
        <li className="nav-item">
          <a 
            className={`nav-link ${activeTab === 'documents' ? 'active' : ''}`} 
            href="#" 
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('documents');
            }}
          >
            Documents
          </a>
        </li>
        <li className="nav-item">
          <a 
            className={`nav-link ${activeTab === 'upload' ? 'active' : ''}`} 
            href="#" 
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('upload');
            }}
          >
            Upload
          </a>
        </li>
      </ul>
      
      {activeTab === 'chat' && (
        <ChatTab 
          messages={messages}
          newMessage={newMessage}
          setNewMessage={setNewMessage}
          sendMessage={sendMessage}
          clearChat={clearChat}
          isConnected={isConnected}
          isStreaming={isStreaming}
          currentStreamingMessage={currentStreamingMessage}
        />
      )}
      
      {activeTab === 'documents' && (
        <DocumentsTab 
          documents={documents}
          processingCount={processingCount}
          jumpToTranscriptSegment={jumpToTranscriptSegment}
          deleteDocument={deleteDocument}
        />
      )}
      
      {activeTab === 'upload' && (
        <UploadTab 
          addSystemMessage={addSystemMessage} 
          refreshDocuments={loadDocuments} 
          apiBaseUrl={API_BASE_URL}
        />
      )}
    </div>
  );
}

export default App; 