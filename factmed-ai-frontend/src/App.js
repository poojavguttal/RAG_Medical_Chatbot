import './App.css';
import React, {useState} from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';

function App() {
  const [chats, setChats] = useState([]);
  const [activeMessages, setActiveMessages] = useState([]);
  const [currentChatIndex, setCurrentChatIndex] = useState(null);
 
  const handleNewChat = () => {
    const newChat = { title: "New Chat", messages: [] };
    setChats(prev => [...prev, newChat]);
    setCurrentChatIndex(chats.length); // new index
    setActiveMessages([]); // clear current messages
  };
  
  const handleSelectChat = (index) => {
    setCurrentChatIndex(index);
    setActiveMessages(chats[index].messages || []);
  };

  const handleClearAllChats = () => {
    setChats([]);
    setActiveMessages([]);
    setCurrentChatIndex(null);
  };
  
  

  const handleSendMessage = async (text) => {
    const userMessage = { sender: 'user', text };
    
    // Show user message + loading message
    const interimMessages = [...activeMessages, userMessage, { sender: 'bot', text: 'Typing...' }];
    setActiveMessages(interimMessages);
  
    try {
      const response = await fetch("https://2c33-34-125-135-95.ngrok-free.app/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
  
      const data = await response.json();
      const botMessage = { sender: 'bot', text: data.answer || "No answer returned." };
  
      // Remove the 'Typing...' placeholder
      const interim = [...interimMessages];
      interim.pop(); // remove 'Typing...'
      const updatedMessages = [...interim, botMessage];
  
      // Update chat
      const updatedChats = [...chats];
      if (currentChatIndex === null) {
        const newChat = { title: text, messages: updatedMessages };
        updatedChats.push(newChat);
        setChats(updatedChats);
        setCurrentChatIndex(updatedChats.length - 1);
      } else {
        if (text && updatedChats[currentChatIndex]?.title === "New Chat") {
          updatedChats[currentChatIndex].title = text;
        }
        updatedChats[currentChatIndex].messages = updatedMessages;
        setChats(updatedChats);
      }
  
      setActiveMessages(updatedMessages);
    } catch (error) {
      console.error("Error fetching:", error);
      const interim = [...interimMessages];
      interim.pop(); // remove 'Typing...'
      const updatedMessages = [...interim, { sender: 'bot', text: '⚠️ Server error. Try again.' }];
      setActiveMessages(updatedMessages);
    }
  };
  
  
  return (
    <div className="App">
      <Sidebar chats={chats}   currentChatIndex={currentChatIndex} onNewChat={handleNewChat} onSelectChat={handleSelectChat} onClearAll={handleClearAllChats}/>
      <div className='app-content'>
       <ChatWindow messages={activeMessages} onSendMessage={handleSendMessage}/>
      </div>
    </div>
  );
}

export default App;
