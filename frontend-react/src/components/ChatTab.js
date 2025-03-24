import React, { useEffect, useRef } from 'react';

const ChatTab = ({ 
  messages, 
  newMessage, 
  setNewMessage, 
  sendMessage, 
  clearChat, 
  isConnected, 
  isStreaming, 
  currentStreamingMessage 
}) => {
  const chatContainerRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, currentStreamingMessage]);

  return (
    <div>
      <div className="chat-container" ref={chatContainerRef}>
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`message ${message.sender === 'user' ? 'user-message' : message.sender === 'ai' ? 'ai-message' : 'system-message'}`}
          >
            {message.sender !== 'system' && (
              <div className="message-header small text-muted mb-1">
                {message.sender === 'user' ? 'You' : 'AI'} - {message.time}
              </div>
            )}
            <div>{message.text}</div>
          </div>
        ))}
        
        {isStreaming && (
          <div className="message ai-message">
            <div className="message-header small text-muted mb-1">
              AI - {new Date().toLocaleTimeString()}
            </div>
            <div>
              {currentStreamingMessage}
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="input-group mb-3">
        <input 
          type="text" 
          className="form-control" 
          value={newMessage} 
          onChange={(e) => setNewMessage(e.target.value)} 
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()} 
          placeholder="Type your message..." 
          disabled={!isConnected || isStreaming}
        />
        <button 
          className="btn btn-primary" 
          onClick={sendMessage} 
          disabled={!isConnected || isStreaming}
        >
          Send
        </button>
        <button 
          className="btn btn-secondary" 
          onClick={clearChat} 
          disabled={!isConnected || isStreaming}
        >
          Clear Chat
        </button>
      </div>
    </div>
  );
};

export default ChatTab; 