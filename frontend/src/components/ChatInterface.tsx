import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  Avatar,
  Button,
  Chip,
  CircularProgress,
  Fade,
  Tooltip,
  Snackbar,
  Alert,
  Zoom,
} from '@mui/material';
import {
  Send,
  Person,
  SmartToy,
  Refresh,
  ContentCopy,
  ThumbUp,
  ThumbDown,
} from '@mui/icons-material';
import { ApiService } from '../services/ApiService';

interface ChatInterfaceProps {
  sessionId: string;
  onNewSession: () => void;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  liked?: boolean;
  disliked?: boolean;
}

const suggestions = [
  'What is this document about?',
  'Summarize the key points',
  'Find specific information',
  'Compare different sections'
];

const ChatInterface: React.FC<ChatInterfaceProps> = React.memo(({ sessionId, onNewSession }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [snackbar, setSnackbar] = useState<{open: boolean, message: string, severity: 'success' | 'info' | 'warning' | 'error'}>({
    open: false,
    message: '',
    severity: 'success'
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Focus input on mount
    inputRef.current?.focus();
  }, []);

  const handleSendMessage = useCallback(async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError('');

    try {
      const response = await ApiService.sendMessage({
        session_id: sessionId,
        message: userMessage.content,
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to send message. Please try again.');
      console.error('Chat error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, isLoading, sessionId]);

  const handleKeyPress = useCallback((event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  }, [handleSendMessage]);

  const copyToClipboard = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setSnackbar({
        open: true,
        message: 'Message copied to clipboard!',
        severity: 'success'
      });
    } catch (err) {
      console.error('Copy failed:', err);
      setSnackbar({
        open: true,
        message: 'Failed to copy message',
        severity: 'error'
      });
    }
  }, []);

  const formatTime = useCallback((date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }, []);

  const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  }, []);

  // Memoize expensive calculations
  const userMessageCount = useMemo(() => {
    return messages.filter(m => m.role === 'user').length;
  }, [messages]);

  const handleSuggestionClick = useCallback((suggestion: string) => {
    setInputValue(suggestion);
  }, []);

  const handleLikeMessage = useCallback((messageId: string) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, liked: !msg.liked, disliked: false }
        : msg
    ));
    setSnackbar({
      open: true,
      message: 'Thanks for your feedback!',
      severity: 'success'
    });
  }, []);

  const handleDislikeMessage = useCallback((messageId: string) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, disliked: !msg.disliked, liked: false }
        : msg
    ));
    setSnackbar({
      open: true,
      message: 'Feedback noted. We\'ll improve!',
      severity: 'info'
    });
  }, []);

  const handleCloseSnackbar = useCallback(() => {
    setSnackbar(prev => ({ ...prev, open: false }));
  }, []);

  const MessageBubble = React.memo<{ 
    message: Message; 
    index: number; 
    onCopy: (text: string) => void; 
    formatTime: (date: Date) => string;
    onLike: (messageId: string) => void;
    onDislike: (messageId: string) => void;
  }>(({ message, index, onCopy, formatTime, onLike, onDislike }) => (
    <Box>
      <Box
        sx={{
          display: 'flex',
          mb: 3,
          alignItems: 'flex-start',
          flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
        }}
      >
        <Avatar
          sx={{
            bgcolor: message.role === 'user' ? 'primary.main' : 'secondary.main',
            mx: 1,
            width: 40,
            height: 40,
          }}
        >
          {message.role === 'user' ? <Person /> : <SmartToy />}
        </Avatar>
        
        <Paper
          elevation={1}
          sx={{
            p: 2,
            maxWidth: '70%',
            bgcolor: message.role === 'user' ? 'primary.main' : 'background.paper',
            color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
            borderRadius: 2,
            position: 'relative',
          }}
        >
          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
            {message.content}
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
            <Typography 
              variant="caption" 
              sx={{ 
                opacity: 0.7,
                color: message.role === 'user' ? 'inherit' : 'text.secondary'
              }}
            >
              {formatTime(message.timestamp)}
            </Typography>
            
            {message.role === 'assistant' && (
              <Box sx={{ display: 'flex', gap: 0.5 }}>
                <Tooltip title="Copy message">
                  <IconButton 
                    size="small" 
                    onClick={() => onCopy(message.content)}
                    sx={{ 
                      color: 'text.secondary',
                      '&:hover': { 
                        color: 'primary.main',
                        backgroundColor: 'action.hover'
                      }
                    }}
                  >
                    <ContentCopy fontSize="small" />
                  </IconButton>
                </Tooltip>
                <Tooltip title={message.liked ? "Remove like" : "Good response"}>
                  <IconButton 
                    size="small" 
                    onClick={() => onLike(message.id)}
                    sx={{ 
                      color: message.liked ? 'success.main' : 'text.secondary',
                      '&:hover': { 
                        color: 'success.main',
                        backgroundColor: 'action.hover'
                      }
                    }}
                  >
                    <ThumbUp fontSize="small" />
                  </IconButton>
                </Tooltip>
                <Tooltip title={message.disliked ? "Remove dislike" : "Poor response"}>
                  <IconButton 
                    size="small" 
                    onClick={() => onDislike(message.id)}
                    sx={{ 
                      color: message.disliked ? 'error.main' : 'text.secondary',
                      '&:hover': { 
                        color: 'error.main',
                        backgroundColor: 'action.hover'
                      }
                    }}
                  >
                    <ThumbDown fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            )}
          </Box>
        </Paper>
      </Box>
    </Box>
  ));

  return (
    <Box sx={{ height: '80vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper 
        elevation={1} 
        sx={{ 
          p: 2, 
          mb: 2, 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          borderRadius: 2,
        }}
      >
        <Box>
          <Typography variant="h6" gutterBottom>
            Chat with Your Documents
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Chip 
              label={`Session: ${sessionId.slice(0, 8)}...`} 
              size="small" 
              color="primary" 
              variant="outlined" 
            />
            <Chip 
              label={`${userMessageCount} messages`} 
              size="small" 
              variant="outlined" 
            />
          </Box>
        </Box>
        <Tooltip title="Start a new chat session">
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={onNewSession}
            size="small"
            sx={{
              '&:hover': {
                backgroundColor: 'primary.main',
                color: 'white',
                transform: 'translateY(-1px)',
              },
              transition: 'all 0.2s ease-in-out',
            }}
          >
            New Session
          </Button>
        </Tooltip>
      </Paper>

      {/* Messages Container */}
      <Paper 
        elevation={1} 
        sx={{ 
          flex: 1, 
          p: 2, 
          overflow: 'auto',
          borderRadius: 2,
          bgcolor: 'background.default',
        }}
      >
        {messages.length === 0 ? (
          <Box>
            <Box 
              sx={{ 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center', 
                justifyContent: 'center',
                height: '100%',
                textAlign: 'center',
              }}
            >
              <SmartToy sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Ready to Chat!
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Ask me anything about your uploaded documents
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
                {suggestions.map((suggestion, index) => (
                  <Zoom in={true} style={{ transitionDelay: `${index * 100}ms` }} key={index}>
                    <Chip
                      label={suggestion}
                      variant="outlined"
                      clickable
                      onClick={() => handleSuggestionClick(suggestion)}
                      sx={{ 
                        m: 0.5,
                        '&:hover': {
                          backgroundColor: 'primary.main',
                          color: 'white',
                          borderColor: 'primary.main',
                          transform: 'translateY(-2px)',
                        },
                        transition: 'all 0.2s ease-in-out',
                      }}
                    />
                  </Zoom>
                ))}
              </Box>
            </Box>
          </Box>
        ) : (
          <Box>
            {messages.map((message, index) => (
              <MessageBubble 
                key={message.id} 
                message={message} 
                index={index}
                onCopy={copyToClipboard}
                formatTime={formatTime}
                onLike={handleLikeMessage}
                onDislike={handleDislikeMessage}
              />
            ))}
          </Box>
        )}

        {isLoading && (
          <Fade in={isLoading}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Avatar sx={{ bgcolor: 'secondary.main', mx: 1 }}>
                <SmartToy />
              </Avatar>
              <Paper elevation={1} sx={{ p: 2, borderRadius: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Thinking...
                  </Typography>
                </Box>
              </Paper>
            </Box>
          </Fade>
        )}

        {error && (
          <Fade in={!!error}>
            <Paper 
              elevation={1} 
              sx={{ 
                p: 2, 
                bgcolor: 'error.light', 
                color: 'error.contrastText',
                borderRadius: 2,
                mb: 2,
              }}
            >
              <Typography variant="body2">
                {error}
              </Typography>
            </Paper>
          </Fade>
        )}

        <div ref={messagesEndRef} />
      </Paper>

      {/* Input Area */}
      <Paper 
        elevation={1} 
        sx={{ 
          p: 2, 
          mt: 2,
          borderRadius: 2,
        }}
      >
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
          <TextField
            ref={inputRef}
            fullWidth
            multiline
            maxRows={4}
            value={inputValue}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            placeholder="Ask about your documents..."
            disabled={isLoading}
            variant="outlined"
            size="small"
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
                '&:hover': {
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'primary.main',
                  },
                },
                '&.Mui-focused': {
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderWidth: 2,
                  },
                },
              },
              '& .MuiInputBase-input': {
                '&::placeholder': {
                  color: 'text.secondary',
                  opacity: 0.7,
                },
              },
            }}
          />
          <Tooltip title={isLoading ? "Sending..." : "Send message"}>
            <span>
              <IconButton
                color="primary"
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                sx={{ 
                  bgcolor: !inputValue.trim() || isLoading ? 'action.disabledBackground' : 'primary.main',
                  color: !inputValue.trim() || isLoading ? 'action.disabled' : 'white',
                  '&:hover': {
                    bgcolor: !inputValue.trim() || isLoading ? 'action.disabledBackground' : 'primary.dark',
                    transform: !inputValue.trim() || isLoading ? 'none' : 'scale(1.05)',
                  },
                  '&:disabled': {
                    bgcolor: 'action.disabledBackground',
                    color: 'action.disabled',
                  },
                  transition: 'all 0.2s ease-in-out',
                }}
              >
                {isLoading ? (
                  <CircularProgress size={20} color="inherit" />
                ) : (
                  <Send />
                )}
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      </Paper>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
});

export default ChatInterface;
