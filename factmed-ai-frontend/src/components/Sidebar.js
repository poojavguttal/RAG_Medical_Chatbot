import React from 'react';
import './Sidebar.css';


const Sidebar = ({chats, currentChatIndex, onNewChat, onSelectChat, onClearAll})=>{
    return (
        <div className="sidebar">
            <h2 className='sidebar-title'>FACTMED.AI</h2>
            <button className="new-chat-button" onClick={onNewChat}> + New Chat</button>
            <div className="chat-history-header">
        <p>Your conversations</p>
        <button className="clear-btn" onClick={onClearAll}>Clear All</button>
      </div>

      <div className="chat-list">
        {chats.map((chat, index) => (
          <div className={`chat-item ${index === currentChatIndex ? 'active' : ''}`}
          key={index} onClick={()=>onSelectChat(index)}>
            <span className="chat-icon">💬</span>
            <span className="chat-title">{chat.title}</span>
          </div>
        ))}
      </div>
    </div>
    
    );
}
export default Sidebar;