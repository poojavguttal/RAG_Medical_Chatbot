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
  
  

  const handleSendMessage = (text) => {
    const userMessage = { sender: 'user', text };
    const botMessage = { sender: 'bot', text: `You said: ${text}` };
    const updatedMessages = [...activeMessages, userMessage, botMessage];
  
    // ✅ If no chat exists yet, create one automatically
    if (currentChatIndex === null) {
      const newChat = {
        title: text, // use first message as title
        messages: updatedMessages,
      };
      const updatedChats = [...chats, newChat];
      setChats(updatedChats);
      setCurrentChatIndex(updatedChats.length - 1);
    } else {
      // ✅ Update existing chat title and messages
      const updatedChats = [...chats];
      
      if (text && updatedChats[currentChatIndex]?.title === "New Chat") {
        updatedChats[currentChatIndex].title = text;
      }
  
      updatedChats[currentChatIndex].messages = updatedMessages;
      setChats(updatedChats);
    }
  
    setActiveMessages(updatedMessages);
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
