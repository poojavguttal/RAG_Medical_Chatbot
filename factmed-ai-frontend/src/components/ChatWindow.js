import React, { useState } from 'react';
import './ChatWindow.css';

const ChatWindow = ({ messages, onSendMessage }) => {
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (input.trim() === "") return;
    onSendMessage(input.trim());
    setInput("");
  };

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.sender}`}>
            <span className="message-text">{msg.text}</span>
          </div>
        ))}
      </div>
      <form className="chat-input-container" onSubmit={(e) => { e.preventDefault(); handleSend(); }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message here..."
        />
        <button type="submit" className="send-button">Send</button>
      </form>
    </div>
  );
};

export default ChatWindow;
